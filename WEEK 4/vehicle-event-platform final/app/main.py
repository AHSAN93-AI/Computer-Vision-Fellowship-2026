"""
app/main.py — FastAPI application entry point.

Wires together:
  - Config loading
  - Database init
  - Pipeline construction
  - API routers
  - Static file serving
  - Background DB writer (consumes the async queue)
  - WebSocket live-feed endpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config, setup_logging, get_config
from app.database import repository as db
from app.events.alerts import build_alert_engine
from app.pipeline import VideoPipeline
from app.api import analytics as analytics_api
from app.api import events as events_api
from app.api import video as video_api
from app.api import test_runner as test_runner_api
from app.api.video import zones_router
from app.api.websocket import ws_manager

logger = logging.getLogger(__name__)

# ─── Global singletons ───────────────────────────────────────────────────────

_pipeline: Optional[VideoPipeline] = None
_db_queue: Optional[asyncio.Queue] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def get_pipeline() -> Optional[VideoPipeline]:
    return _pipeline


def get_db_queue() -> Optional[asyncio.Queue]:
    return _db_queue


def get_loop() -> Optional[asyncio.AbstractEventLoop]:
    return _loop


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and configure the FastAPI app."""
    cfg = load_config()
    setup_logging(cfg.logging)

    app = FastAPI(
        title="Vehicle Event Detection & Alerting Platform",
        description="Intelligent vehicle monitoring: detection, tracking, zone analysis, rule engine, and event management.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins in dev; restrict in prod via config/env
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register API routers ─────────────────────────────────────────────────
    app.include_router(video_api.router)
    app.include_router(zones_router)
    app.include_router(events_api.router)
    app.include_router(analytics_api.router)
    app.include_router(test_runner_api.router)

    # ── Serve static files ───────────────────────────────────────────────────
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/history.html", include_in_schema=False)
        async def serve_history():
            return FileResponse(str(static_dir / "history.html"))

    # ── Serve evidence files (frames, crops, clips) referenced by events ────
    evidence_dir = Path(cfg.evidence.storage_path)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/evidence", StaticFiles(directory=str(evidence_dir)), name="evidence")

    # ── WebSocket endpoint ───────────────────────────────────────────────────
    @app.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket):
        """WebSocket endpoint for live frame + metrics streaming."""
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive; pipeline pushes frames via broadcast
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Accept ping messages gracefully
                if data == "ping":
                    await websocket.send_text("pong")
        except asyncio.TimeoutError:
            pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket error: {e}")
        finally:
            await ws_manager.disconnect(websocket)

    # ── Lifespan events ──────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        global _pipeline, _db_queue, _loop

        _loop = asyncio.get_running_loop()
        _db_queue = asyncio.Queue()

        # Init database
        await db.init_db()

        # Build alert engine (converts AlertChannelConfig → real AlertChannel instances)
        alert_engine = build_alert_engine(
            channel_configs=cfg.alerts.channels,
        )

        # Frame-ready callback: broadcast via WebSocket
        async def on_frame_ready(frame_b64: str, live_data: dict):
            await ws_manager.broadcast_frame(frame_b64, live_data)

        # Build pipeline
        _pipeline = VideoPipeline(
            config=cfg,
            alert_engine=alert_engine,
            loop=_loop,
            on_frame_ready=on_frame_ready,
        )
        _pipeline.set_loop(_loop)

        # Inject pipeline into API routers
        video_api.set_pipeline(_pipeline)
        events_api.set_pipeline(_pipeline)
        analytics_api.set_pipeline(_pipeline)

        # Start background DB writer
        _loop.create_task(_db_writer(_db_queue))

        logger.info("Application started. Dashboard at http://localhost:8000")

    @app.on_event("shutdown")
    async def on_shutdown():
        global _pipeline
        if _pipeline and _pipeline.state.running:
            _pipeline.stop()
        logger.info("Application shutdown complete.")

    return app


# ─── Background DB writer ─────────────────────────────────────────────────────

async def _db_writer(queue: asyncio.Queue) -> None:
    """
    Consumes (operation, event) tuples from the queue and writes to DB.
    Runs in the asyncio event loop so it doesn't block the pipeline thread.
    """
    logger.info("DB writer started")
    while True:
        try:
            item = await queue.get()
            if item is None:  # poison pill
                break
            op, event = item
            if op == "create":
                await db.create_event(event)
            elif op == "update":
                await db.update_event(event)
            queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"DB writer error: {e}")


# ─── Entry point ─────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("APP_DEBUG", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )
