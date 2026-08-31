"""
app.events.evidence — Evidence Frame Capture (§4.16)

Captures and saves annotated frames as JPEG images when key events
occur (fall, posture risk, activity changes).

For falls, captures three frames where practical:
  1. Pre-event (from the sequence buffer)
  2. Event frame
  3. Post-event (captured a few seconds later by the pipeline)

File naming: ``{event_id}_{person_id}_{timestamp}_{label}.jpg``

Handles storage failures gracefully — logs a warning and continues
without crashing the pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


class EvidenceCapture:
    """Manages saving evidence frames to disk."""

    def __init__(self, evidence_dir: Optional[str] = None) -> None:
        settings = get_settings()
        self._dir = Path(evidence_dir) if evidence_dir else settings.evidence_abs_path
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("Evidence directory: %s", self._dir)

    def save_frame(
        self,
        frame: np.ndarray,
        event_id: str,
        person_id: int,
        label: str,
        suffix: str = "",
    ) -> Optional[str]:
        """Save a full frame as JPEG evidence.

        Parameters
        ----------
        frame:
            BGR image to save.
        event_id:
            Event identifier.
        person_id:
            Tracked person ID.
        label:
            Activity/event label (e.g. ``"fall"``, ``"posture_risk"``).
        suffix:
            Optional suffix (e.g. ``"_pre"``, ``"_post"``).

        Returns
        -------
        Absolute path to the saved file, or None on failure.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{event_id}_p{person_id:03d}_{ts}_{label}{suffix}.jpg"
        filepath = self._dir / filename

        try:
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.debug("Evidence saved: %s", filepath)
            return str(filepath)
        except Exception:
            logger.warning("Failed to save evidence: %s", filepath, exc_info=True)
            return None

    def save_person_crop(
        self,
        frame: np.ndarray,
        bbox: tuple,
        event_id: str,
        person_id: int,
        label: str,
    ) -> Optional[str]:
        """Save a cropped region around the detected person.

        Parameters
        ----------
        bbox:
            (x1, y1, x2, y2) bounding box in pixels.

        Returns
        -------
        Path to saved crop, or None on failure.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        # Add 10% padding
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("Empty crop for person #%d", person_id)
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{event_id}_p{person_id:03d}_{ts}_{label}_crop.jpg"
        filepath = self._dir / filename

        try:
            cv2.imwrite(str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.debug("Evidence crop saved: %s", filepath)
            return str(filepath)
        except Exception:
            logger.warning("Failed to save evidence crop: %s", filepath, exc_info=True)
            return None

    def save_fall_sequence(
        self,
        pre_frame: Optional[np.ndarray],
        event_frame: np.ndarray,
        event_id: str,
        person_id: int,
    ) -> list[Optional[str]]:
        """Save pre-event and event frames for a fall.

        Post-event frame is captured later by the pipeline.

        Returns
        -------
        List of saved file paths [pre_path, event_path].
        """
        paths = []
        if pre_frame is not None:
            paths.append(self.save_frame(pre_frame, event_id, person_id, "fall", "_pre"))
        else:
            paths.append(None)
        paths.append(self.save_frame(event_frame, event_id, person_id, "fall", "_event"))
        return paths
