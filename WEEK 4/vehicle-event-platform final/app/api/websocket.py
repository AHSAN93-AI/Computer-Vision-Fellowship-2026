"""
app/api/websocket.py — WebSocket connection manager.

Manages connected WebSocket clients and broadcasts frames + live data.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List

# pyrefly: ignore [missing-import]
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for live dashboard streaming.

    Handles:
    - Client connect/disconnect
    - Frame + metrics broadcast to all connected clients
    - Alert broadcast
    - Graceful error handling (dead connections cleaned up)
    """

    def __init__(self):
        self._connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self._connections)}")

    async def broadcast_frame(self, frame_b64: str, live_data: dict) -> None:
        """Broadcast a frame (base64 JPEG) + live metrics to all clients."""
        if not self._connections:
            return

        message = json.dumps({
            "type": "frame",
            "frame": frame_b64,
            "data": live_data,
        })

        dead: List[WebSocket] = []
        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                if ws in self._connections:
                    self._connections.remove(ws)

    async def broadcast_alert(self, alert_data: dict) -> None:
        """Broadcast an alert notification to all clients."""
        if not self._connections:
            return

        message = json.dumps({
            "type": "alert",
            "data": alert_data,
        })

        dead: List[WebSocket] = []
        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                if ws in self._connections:
                    self._connections.remove(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


# Global singleton
ws_manager = WebSocketManager()
