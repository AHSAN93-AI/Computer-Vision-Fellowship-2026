"""
app/api/video.py — Video source management API endpoints.

Handles:
  - Video file upload and start processing
  - Webcam start/stop
  - RTSP stream connect/disconnect
  - Pipeline state queries
  - Zone configuration endpoints
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])
zones_router = APIRouter(prefix="/api/zones", tags=["zones"])

_pipeline = None

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and start processing it."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # Validate extension
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            400,
            f"Unsupported file format: {ext}. Allowed: {', '.join(allowed_exts)}"
        )

    # Save uploaded file
    save_path = UPLOAD_DIR / f"{int(time.time())}_{file.filename}"
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    # Start processing
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    try:
        _pipeline.start_file(str(save_path))
        return {
            "status": "started",
            "source": "file",
            "file": file.filename,
            "path": str(save_path),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to start processing: {e}")


@router.post("/webcam/start")
async def start_webcam(device_index: int = Query(0, ge=0)):
    """Start processing from webcam."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    try:
        _pipeline.start_webcam(device_index)
        return {
            "status": "started",
            "source": "webcam",
            "device_index": device_index,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to start webcam: {e}")


@router.post("/rtsp/start")
async def start_rtsp(url: str = Query(...)):
    """Start processing from RTSP stream."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    if not url.startswith("rtsp://"):
        raise HTTPException(400, "URL must start with rtsp://")

    try:
        _pipeline.start_rtsp(url)
        return {
            "status": "started",
            "source": "rtsp",
            "url": url,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to connect to RTSP: {e}")


@router.post("/stop")
async def stop_video():
    """Stop the current video processing."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    try:
        _pipeline.stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(500, f"Failed to stop: {e}")


@router.get("/status")
async def get_status():
    """Get current pipeline status."""
    if _pipeline is None:
        return {
            "running": False,
            "source": None,
            "frame_id": 0,
            "fps": 0,
        }

    state = _pipeline.state
    return {
        "running": state.running,
        "source": state.source_type,
        "source_id": state.source_id,
        "frame_id": state.frame_id,
        "fps": round(state.fps, 1),
        "active_tracks": state.active_tracks,
        "active_events": state.active_events,
    }


# ─── Zone endpoints ─────────────────────────────────────────────────────────

@zones_router.get("")
async def list_zones():
    """List all configured zones."""
    if _pipeline is None:
        return {"zones": []}

    zones = _pipeline.get_zone_configs()
    return {
        "zones": [
            {
                "id": z.id,
                "name": z.name,
                "type": z.type,
                "polygon": z.polygon,
                "line": z.line,
                "max_capacity": z.max_capacity,
                "color": z.color,
                "monitored_classes": z.monitored_classes,
            }
            for z in zones
        ]
    }


@zones_router.get("/{zone_id}")
async def get_zone(zone_id: str):
    """Get a single zone by ID."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    zone = _pipeline.get_zone_config(zone_id)
    if zone is None:
        raise HTTPException(404, f"Zone '{zone_id}' not found")

    return {
        "id": zone.id,
        "name": zone.name,
        "type": zone.type,
        "polygon": zone.polygon,
        "line": zone.line,
        "max_capacity": zone.max_capacity,
        "color": zone.color,
        "monitored_classes": zone.monitored_classes,
    }
