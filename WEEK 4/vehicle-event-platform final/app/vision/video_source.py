"""
app/vision/video_source.py — Video source abstraction.

Supports: uploaded video file, webcam, RTSP stream.
Each source yields (frame: np.ndarray, metadata: FrameMetadata).
Handles errors gracefully: invalid file, camera disconnect, video end.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSourceError(Exception):
    """Raised when a video source cannot be opened or is invalid."""
    pass


@dataclass
class FrameMetadata:
    source_id: str
    frame_number: int
    timestamp: float              # epoch seconds
    fps: float
    width: int
    height: int
    source_type: str              # "file", "webcam", "rtsp"


class VideoSource(ABC):
    """Abstract base class for all video sources."""

    def __init__(self, source_id: str, max_fps: int = 30, frame_skip: int = 0):
        self.source_id = source_id
        self.max_fps = max_fps
        self.frame_skip = frame_skip
        self._frame_number = 0
        self._cap: Optional[cv2.VideoCapture] = None

    @abstractmethod
    def open(self) -> None:
        """Open the video source. Raises VideoSourceError on failure."""
        ...

    @abstractmethod
    def source_type(self) -> str:
        ...

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single raw frame from the source."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame

    def get_native_fps(self) -> float:
        if self._cap is None:
            return 25.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    def frames(self) -> Generator[Tuple[np.ndarray, FrameMetadata], None, None]:
        """
        Generator that yields (frame, metadata) tuples.
        Handles frame skipping and FPS throttling.
        """
        self.open()
        native_fps = self.get_native_fps()
        min_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0
        last_yield_time = 0.0
        skip_counter = 0

        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    if not self._handle_read_failure():
                        break
                    continue

                self._frame_number += 1

                # Frame skipping
                if self.frame_skip > 0:
                    skip_counter += 1
                    if skip_counter <= self.frame_skip:
                        continue
                    skip_counter = 0

                # FPS throttling
                now = time.time()
                elapsed = now - last_yield_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_yield_time = time.time()

                h, w = frame.shape[:2]
                meta = FrameMetadata(
                    source_id=self.source_id,
                    frame_number=self._frame_number,
                    timestamp=time.time(),
                    fps=native_fps,
                    width=w,
                    height=h,
                    source_type=self.source_type(),
                )
                yield frame, meta
        finally:
            self.release()

    def _handle_read_failure(self) -> bool:
        """Called when read() returns False. Return True to continue, False to stop."""
        return False

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info(f"Video source '{self.source_id}' released")


class FileVideoSource(VideoSource):
    """Video source backed by a file path (mp4, avi, etc.)."""

    def __init__(self, file_path: str, source_id: str = "file",
                 max_fps: int = 30, frame_skip: int = 0):
        super().__init__(source_id=source_id, max_fps=max_fps, frame_skip=frame_skip)
        self._file_path = Path(file_path)

    def source_type(self) -> str:
        return "file"

    def open(self) -> None:
        if not self._file_path.exists():
            raise VideoSourceError(f"Video file not found: {self._file_path}")
        self._cap = cv2.VideoCapture(str(self._file_path))
        if not self._cap.isOpened():
            raise VideoSourceError(
                f"Could not open video file: {self._file_path} "
                "(file may be corrupt or unsupported format)"
            )
        logger.info(
            f"Opened file source: {self._file_path} "
            f"({self.get_native_fps():.1f} fps)"
        )

    def _handle_read_failure(self) -> bool:
        """File ended — stop gracefully."""
        logger.info(f"File source '{self.source_id}' reached end of video")
        return False


class WebcamVideoSource(VideoSource):
    """Video source backed by a webcam device index."""

    MAX_RETRIES = 5
    RETRY_DELAY_S = 2.0

    def __init__(self, device_index: int = 0, source_id: str = "webcam",
                 max_fps: int = 30, frame_skip: int = 0):
        super().__init__(source_id=source_id, max_fps=max_fps, frame_skip=frame_skip)
        self._device_index = device_index
        self._consecutive_failures = 0

    def source_type(self) -> str:
        return "webcam"

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise VideoSourceError(
                f"Could not open webcam at index {self._device_index}. "
                "Check that the camera is connected and not in use by another application."
            )
        # Set preferred resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info(f"Opened webcam device {self._device_index}")

    def _handle_read_failure(self) -> bool:
        """Webcam disconnect — retry with backoff."""
        self._consecutive_failures += 1
        if self._consecutive_failures > self.MAX_RETRIES:
            logger.error(
                f"Webcam '{self.source_id}' failed {self.MAX_RETRIES} consecutive times — stopping"
            )
            return False
        logger.warning(
            f"Webcam read failed (attempt {self._consecutive_failures}/{self.MAX_RETRIES}), "
            f"retrying in {self.RETRY_DELAY_S}s..."
        )
        time.sleep(self.RETRY_DELAY_S)
        # Attempt re-open
        try:
            if self._cap:
                self._cap.release()
            self.open()
            self._consecutive_failures = 0
        except VideoSourceError as e:
            logger.warning(f"Webcam re-open failed: {e}")
        return True


class RTSPVideoSource(VideoSource):
    """
    Video source backed by an RTSP stream URL.

    NOTE: RTSP support is included at configuration level.
    Not tested against a live RTSP stream in this build (no live camera available).
    Reconnects on drop with exponential backoff.
    """

    MAX_RETRIES = 10
    BASE_RETRY_DELAY_S = 1.0
    MAX_RETRY_DELAY_S = 30.0

    def __init__(self, rtsp_url: str, source_id: str = "rtsp",
                 max_fps: int = 30, frame_skip: int = 0):
        super().__init__(source_id=source_id, max_fps=max_fps, frame_skip=frame_skip)
        if not rtsp_url:
            raise VideoSourceError("RTSP URL cannot be empty")
        self._rtsp_url = rtsp_url
        self._consecutive_failures = 0

    def source_type(self) -> str:
        return "rtsp"

    def open(self) -> None:
        # Use TCP transport for reliability
        os_env = {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "rtsp_transport;tcp"}
        self._cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Could not open RTSP stream: {self._rtsp_url}")
        logger.info(f"Opened RTSP stream: {self._rtsp_url}")

    def _handle_read_failure(self) -> bool:
        self._consecutive_failures += 1
        if self._consecutive_failures > self.MAX_RETRIES:
            logger.error(f"RTSP '{self.source_id}' failed {self.MAX_RETRIES} times — stopping")
            return False
        delay = min(
            self.BASE_RETRY_DELAY_S * (2 ** (self._consecutive_failures - 1)),
            self.MAX_RETRY_DELAY_S,
        )
        logger.warning(f"RTSP read failed, reconnecting in {delay:.1f}s...")
        time.sleep(delay)
        try:
            if self._cap:
                self._cap.release()
            self.open()
            self._consecutive_failures = 0
        except VideoSourceError as e:
            logger.warning(f"RTSP reconnect failed: {e}")
        return True


def create_video_source(
    source_type: str,
    file_path: Optional[str] = None,
    webcam_index: int = 0,
    rtsp_url: str = "",
    max_fps: int = 30,
    frame_skip: int = 0,
) -> VideoSource:
    """Factory function to create the appropriate VideoSource."""
    if source_type == "file":
        if not file_path:
            raise VideoSourceError("file_path required for file source")
        return FileVideoSource(file_path, source_id=Path(file_path).name,
                               max_fps=max_fps, frame_skip=frame_skip)
    elif source_type == "webcam":
        return WebcamVideoSource(device_index=webcam_index,
                                 max_fps=max_fps, frame_skip=frame_skip)
    elif source_type == "rtsp":
        return RTSPVideoSource(rtsp_url=rtsp_url, max_fps=max_fps, frame_skip=frame_skip)
    else:
        raise VideoSourceError(f"Unknown source type: '{source_type}'. Use 'file', 'webcam', or 'rtsp'")
