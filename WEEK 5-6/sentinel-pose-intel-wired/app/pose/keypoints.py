"""
app.pose.keypoints — COCO Keypoint Definitions & Utilities

The YOLO-Pose model outputs 17 keypoints in standard COCO order.
This module provides:
  • Named constants so the rest of the codebase never hard-codes indices.
  • A ``Keypoint`` / ``PersonKeypoints`` dataclass for structured access.
  • Helpers for confidence filtering and missing-keypoint checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── COCO keypoint indices ──────────────────────────────
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

NUM_KEYPOINTS = 17

KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton connections for drawing (pairs of keypoint indices)
SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]

# ── Colour palette for skeleton rendering ──────────────
# BGR format for OpenCV
KEYPOINT_COLOR = (0, 255, 170)       # green-cyan
SKELETON_COLOR = (0, 200, 140)       # slightly darker
BBOX_COLOR_NORMAL = (0, 255, 170)    # green-cyan
BBOX_COLOR_ALERT = (0, 0, 255)       # red
LABEL_BG_COLOR = (20, 20, 20)        # near-black


@dataclass
class Keypoint:
    """A single body keypoint with pixel coordinates and confidence."""

    x: float              # pixel x-coordinate in original frame
    y: float              # pixel y-coordinate in original frame
    confidence: float     # model confidence 0..1
    index: int            # COCO index (0–16)
    name: str             # human-readable name

    @property
    def is_visible(self) -> bool:
        """True if confidence is above zero (pre-filtering has not zeroed it)."""
        return self.confidence > 0.0 and not (np.isnan(self.x) or np.isnan(self.y))

    def as_tuple(self) -> Tuple[float, float]:
        """Return (x, y) for drawing."""
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        """Return (x, y) rounded to int for OpenCV drawing."""
        return (int(round(self.x)), int(round(self.y)))


@dataclass
class PersonKeypoints:
    """All 17 COCO keypoints for one detected person.

    Keypoints that fell below the confidence threshold during filtering
    have their x/y set to ``NaN`` and confidence set to 0.0.
    """

    keypoints: List[Keypoint] = field(default_factory=list)
    # The original bounding box from detection [x1, y1, x2, y2] in pixels
    bbox: Optional[Tuple[float, float, float, float]] = None
    detection_confidence: float = 0.0
    # Track ID assigned by the tracker (None before tracking runs)
    track_id: Optional[int] = None

    def __getitem__(self, index: int) -> Keypoint:
        return self.keypoints[index]

    def __len__(self) -> int:
        return len(self.keypoints)

    @property
    def visible_count(self) -> int:
        """Number of keypoints currently marked as visible."""
        return sum(1 for kp in self.keypoints if kp.is_visible)

    @property
    def is_valid(self) -> bool:
        """True if enough keypoints are visible for activity analysis."""
        # Imported lazily to avoid circular import at module load
        from app.config import get_settings
        return self.visible_count >= get_settings().min_visible_keypoints

    def get(self, index: int) -> Optional[Keypoint]:
        """Return keypoint if visible, else None."""
        if 0 <= index < len(self.keypoints) and self.keypoints[index].is_visible:
            return self.keypoints[index]
        return None

    def midpoint(self, idx_a: int, idx_b: int) -> Optional[Tuple[float, float]]:
        """Return the midpoint of two keypoints, or None if either is missing."""
        a = self.get(idx_a)
        b = self.get(idx_b)
        if a is None or b is None:
            return None
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    @property
    def mid_hip(self) -> Optional[Tuple[float, float]]:
        return self.midpoint(LEFT_HIP, RIGHT_HIP)

    @property
    def mid_shoulder(self) -> Optional[Tuple[float, float]]:
        return self.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)

    def to_array(self) -> np.ndarray:
        """Return shape (17, 3) array of [x, y, confidence]."""
        arr = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for kp in self.keypoints:
            arr[kp.index] = [kp.x, kp.y, kp.confidence]
        return arr


def filter_keypoints_by_confidence(
    raw_xy: np.ndarray,
    raw_conf: np.ndarray,
    threshold: float,
) -> PersonKeypoints:
    """Build a ``PersonKeypoints`` from raw model output, zeroing low-confidence points.

    Parameters
    ----------
    raw_xy:
        Shape (17, 2) array of (x, y) pixel coordinates.
    raw_conf:
        Shape (17,) confidence array.
    threshold:
        Keypoints with confidence < threshold get x/y = NaN, conf = 0.

    Returns
    -------
    PersonKeypoints with all 17 keypoints populated.
    """
    kps: List[Keypoint] = []
    for i in range(NUM_KEYPOINTS):
        conf = float(raw_conf[i])
        if conf >= threshold:
            kps.append(Keypoint(
                x=float(raw_xy[i, 0]),
                y=float(raw_xy[i, 1]),
                confidence=conf,
                index=i,
                name=KEYPOINT_NAMES[i],
            ))
        else:
            kps.append(Keypoint(
                x=float("nan"),
                y=float("nan"),
                confidence=0.0,
                index=i,
                name=KEYPOINT_NAMES[i],
            ))
    return PersonKeypoints(keypoints=kps)
