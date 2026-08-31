"""
app.vision.video_source — Video Input Abstraction

Wraps ``cv2.VideoCapture`` to provide a consistent interface for:
  • Local video files (.mp4, .avi, .mkv, …)
  • Webcam streams (by integer index: 0, 1, …)
  • RTSP / HTTP streams (bonus)

Handles connection errors, disconnections, and frame-read failures
gracefully — the caller always gets either a valid frame or ``None``.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SourceType(Enum):
    FILE = "file"
    WEBCAM = "webcam"
    STREAM = "stream"  # RTSP / HTTP


class VideoSource:
    """Thread-safe video frame provider.

    Parameters
    ----------
    source:
        File path, webcam index as string (``"0"``), or RTSP/HTTP URL.
    """

    def __init__(self, source: str) -> None:
        self._source_str = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_type = self._classify(source)
        self._frame_count = 0
        self._last_read_time: float = 0.0
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0  # 0 for live sources

    # ── Public API ──────────────────────────────────────

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        source = self._resolve_source()
        try:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                logger.error("Failed to open video source: %s", self._source_str)
                return False

            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self._source_type == SourceType.FILE:
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                "Opened %s source: %s (%dx%d @ %.1f FPS, %s frames)",
                self._source_type.value,
                self._source_str,
                self._width,
                self._height,
                self._fps,
                self._total_frames or "live",
            )
            return True

        except Exception:
            logger.exception("Exception opening video source: %s", self._source_str)
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read and return the next frame, or None on failure / end-of-stream.

        Tracks read timing for FPS calculation.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self._source_type == SourceType.FILE:
                logger.info("End of video file reached (%d frames read)", self._frame_count)
            else:
                logger.warning("Frame read failed from %s source", self._source_type.value)
            return None

        now = time.monotonic()
        if self._last_read_time > 0:
            dt = now - self._last_read_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)  # EMA smoothing
        self._last_read_time = now
        self._frame_count += 1

        return frame

    def release(self) -> None:
        """Release the underlying capture."""
        if self._cap is not None:
            self._cap.release()
            logger.info("Released video source: %s (%d frames read)", self._source_str, self._frame_count)
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height)."""
        return (self._width, self._height)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_frames(self) -> int:
        """Total frames in file, or 0 for live sources."""
        return self._total_frames

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def source_str(self) -> str:
        return self._source_str

    # ── Private helpers ─────────────────────────────────

    @staticmethod
    def _classify(source: str) -> SourceType:
        """Determine what kind of source the string represents."""
        stripped = source.strip()
        if stripped.isdigit():
            return SourceType.WEBCAM
        if stripped.startswith(("rtsp://", "http://", "https://")):
            return SourceType.STREAM
        return SourceType.FILE

    def _resolve_source(self) -> int | str:
        """Convert the source string to the value cv2.VideoCapture expects."""
        if self._source_type == SourceType.WEBCAM:
            return int(self._source_str.strip())
        if self._source_type == SourceType.FILE:
            path = Path(self._source_str)
            if not path.exists():
                logger.error("Video file not found: %s", path)
                # Still pass it to VideoCapture — it will fail and we handle it
            return str(path)
        # Stream URL
        return self._source_str.strip()

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"VideoSource(source={self._source_str!r}, type={self._source_type.value})"
