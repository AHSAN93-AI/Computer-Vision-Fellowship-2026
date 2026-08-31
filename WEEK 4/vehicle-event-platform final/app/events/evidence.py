"""
app/events/evidence.py — Evidence capture and storage.

On event creation, saves:
  - Full frame JPEG
  - Cropped vehicle image JPEG
  - Event metadata JSON
  - Optional: short video clip from rolling buffer

Evidence stored at: {storage_path}/{event_id}/
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceResult:
    """Paths to saved evidence files."""
    event_id: str
    full_frame_path: Optional[str] = None
    crop_path: Optional[str] = None
    metadata_path: Optional[str] = None
    clip_path: Optional[str] = None
    evidence_dir: Optional[str] = None

    @property
    def primary_path(self) -> Optional[str]:
        return self.evidence_dir


class FrameBuffer:
    """Rolling frame buffer for pre-event clip capture."""

    def __init__(self, max_frames: int = 90):
        self._buffer: deque = deque(maxlen=max_frames)

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        try:
            self._buffer.append((frame.copy(), timestamp))
        except Exception as e:
            logger.debug(f"FrameBuffer.add_frame error: {e}")

    def get_frames(self) -> list:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


class EvidenceSaver:
    """
    Saves evidence files for events.
    Handles storage failures gracefully (logs error, returns None paths).
    """

    def __init__(
        self,
        storage_path: str = "./evidence",
        save_full_frame: bool = True,
        save_crop: bool = True,
        save_metadata: bool = True,
        frame_buffer: Optional[FrameBuffer] = None,
        jpeg_quality: int = 85,
    ):
        self._storage_path = Path(storage_path)
        self._save_full = save_full_frame
        self._save_crop = save_crop
        self._save_meta = save_metadata
        self._frame_buffer = frame_buffer
        self._jpeg_quality = jpeg_quality

        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Evidence storage path unavailable: {e}")

    def save(
        self,
        event: object,     # Event from event_manager
        frame: np.ndarray,
        violation: object = None,  # RuleViolation
    ) -> Optional[str]:
        """
        Save evidence for an event.
        Returns the evidence directory path, or None on failure.
        """
        try:
            event_dir = self._storage_path / event.event_id
            event_dir.mkdir(parents=True, exist_ok=True)

            result = EvidenceResult(
                event_id=event.event_id,
                evidence_dir=str(event_dir),
            )

            # Full frame
            if self._save_full and frame is not None:
                frame_path = event_dir / "frame.jpg"
                if not self._save_jpg(frame, frame_path):
                    result.full_frame_path = None
                else:
                    result.full_frame_path = str(frame_path)

            # Crop (if we have bbox info)
            if self._save_crop and frame is not None and violation is not None:
                crop = self._extract_crop(frame, event, violation)
                if crop is not None:
                    crop_path = event_dir / "crop.jpg"
                    if self._save_jpg(crop, crop_path):
                        result.crop_path = str(crop_path)

            # Metadata JSON
            if self._save_meta:
                meta_path = event_dir / "metadata.json"
                self._save_metadata(event, meta_path)
                result.metadata_path = str(meta_path)

            logger.debug(f"Evidence saved: {event_dir}")
            return str(event_dir)

        except OSError as e:
            logger.error(f"Evidence storage failure (disk full or path invalid?): {e}")
            return None
        except Exception as e:
            logger.error(f"Evidence save error for event {event.event_id}: {e}")
            return None

    def _save_jpg(self, image: np.ndarray, path: Path) -> bool:
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            success, buf = cv2.imencode(".jpg", image, encode_params)
            if success:
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                return True
        except Exception as e:
            logger.warning(f"Failed to save JPEG {path}: {e}")
        return False

    def _extract_crop(
        self,
        frame: np.ndarray,
        event: object,
        violation: object,
    ) -> Optional[np.ndarray]:
        """Extract cropped vehicle image from frame."""
        try:
            # Try to get bbox from the tracked vehicle if available
            # violation may have track_id we can look up — use a 50px padding
            h, w = frame.shape[:2]
            # Default: center crop (fallback)
            x1, y1, x2, y2 = 0, 0, w, h

            # If violation has metadata with bbox
            if hasattr(violation, "metadata") and "bbox" in violation.metadata:
                bbox = violation.metadata["bbox"]
                x1, y1, x2, y2 = bbox

            # Apply 20px padding, clamp to frame
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2].copy()
        except Exception as e:
            logger.debug(f"Crop extraction failed: {e}")
        return None

    def _save_metadata(self, event: object, path: Path) -> None:
        try:
            meta = event.to_dict() if hasattr(event, "to_dict") else {}
            meta["saved_at"] = time.time()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Metadata save failed: {e}")
