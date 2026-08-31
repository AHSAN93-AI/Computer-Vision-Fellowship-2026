"""
app.pose.angles — Joint-Angle Calculations (§4.6)

Reusable functions for computing angles between three keypoints.
All functions return ``None`` when any required keypoint is missing
or below the confidence threshold — they never fabricate a value.

Angle convention: the angle at the *middle* point of three keypoints,
measured in degrees (0–180).  For example, ``elbow_angle(shoulder,
elbow, wrist)`` returns the angle at the elbow.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from app.pose.keypoints import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
    PersonKeypoints,
)

logger = logging.getLogger(__name__)


def calculate_angle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    """Compute the angle at ``p2`` formed by the line segments p1-p2 and p3-p2.

    Parameters
    ----------
    p1, p2, p3 : (x, y)
        Three 2-D points.  The angle is measured **at p2**.

    Returns
    -------
    Angle in degrees, range [0, 180].
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0

    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _angle_from_indices(
    pk: PersonKeypoints, idx_a: int, idx_b: int, idx_c: int,
) -> Optional[float]:
    """Compute angle at keypoint ``idx_b`` using points a-b-c.

    Returns None if any keypoint is not visible.
    """
    a = pk.get(idx_a)
    b = pk.get(idx_b)
    c = pk.get(idx_c)
    if a is None or b is None or c is None:
        return None
    return calculate_angle(a.as_tuple(), b.as_tuple(), c.as_tuple())


# ── Named angle functions ──────────────────────────────

def left_elbow_angle(pk: PersonKeypoints) -> Optional[float]:
    """Angle at left elbow (shoulder → elbow → wrist)."""
    return _angle_from_indices(pk, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)


def right_elbow_angle(pk: PersonKeypoints) -> Optional[float]:
    """Angle at right elbow (shoulder → elbow → wrist)."""
    return _angle_from_indices(pk, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)


def left_knee_angle(pk: PersonKeypoints) -> Optional[float]:
    """Angle at left knee (hip → knee → ankle)."""
    return _angle_from_indices(pk, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)


def right_knee_angle(pk: PersonKeypoints) -> Optional[float]:
    """Angle at right knee (hip → knee → ankle)."""
    return _angle_from_indices(pk, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)


def hip_angle(pk: PersonKeypoints) -> Optional[float]:
    """Angle at hip midpoint: mid-shoulder → mid-hip → mid-knee.

    Uses the midpoint of both hips and both shoulders.
    For the lower vector, uses the midpoint of both knees.
    Returns None if any midpoint cannot be computed.
    """
    mid_s = pk.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
    mid_h = pk.midpoint(LEFT_HIP, RIGHT_HIP)
    mid_k = pk.midpoint(LEFT_KNEE, RIGHT_KNEE)
    if mid_s is None or mid_h is None or mid_k is None:
        return None
    return calculate_angle(mid_s, mid_h, mid_k)


def torso_angle(pk: PersonKeypoints) -> Optional[float]:
    """Angle between the torso line (mid-hip → mid-shoulder) and vertical.

    Returns 0° when perfectly upright, 90° when horizontal.
    Useful for standing/sitting/bending/fall classification.
    """
    mid_s = pk.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
    mid_h = pk.midpoint(LEFT_HIP, RIGHT_HIP)
    if mid_s is None or mid_h is None:
        return None

    # Vector from hip to shoulder
    dx = mid_s[0] - mid_h[0]
    dy = mid_s[1] - mid_h[1]  # Note: in image coords, y increases downward

    # Vertical reference: straight up = (0, -1) in image coords
    # angle = arctan2(|dx|, -dy)  → 0° when upright, 90° when horizontal
    angle = math.degrees(math.atan2(abs(dx), -dy))
    return max(0.0, min(180.0, angle))


def torso_length(pk: PersonKeypoints) -> Optional[float]:
    """Euclidean distance from mid-hip to mid-shoulder (pixels).

    Used as a scaling reference for normalisation.
    """
    mid_s = pk.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
    mid_h = pk.midpoint(LEFT_HIP, RIGHT_HIP)
    if mid_s is None or mid_h is None:
        return None
    return float(math.hypot(mid_s[0] - mid_h[0], mid_s[1] - mid_h[1]))


def shoulder_width(pk: PersonKeypoints) -> Optional[float]:
    """Distance between left and right shoulder (pixels)."""
    ls = pk.get(LEFT_SHOULDER)
    rs = pk.get(RIGHT_SHOULDER)
    if ls is None or rs is None:
        return None
    return float(math.hypot(ls.x - rs.x, ls.y - rs.y))


def head_to_hip_vertical(pk: PersonKeypoints) -> Optional[float]:
    """Vertical distance from nose to mid-hip (pixels, positive = nose above hip).

    In image coordinates, y increases downward, so this returns
    ``mid_hip_y - nose_y`` (positive when nose is above the hip).
    Useful for fall detection (head-to-ground analysis).
    """
    nose = pk.get(NOSE)
    mid_h = pk.midpoint(LEFT_HIP, RIGHT_HIP)
    if nose is None or mid_h is None:
        return None
    return mid_h[1] - nose.y  # positive when head is above hips


def bbox_aspect_ratio(pk: PersonKeypoints) -> Optional[float]:
    """Width / height ratio of the bounding box.

    Upright person → ratio < 1 (taller than wide).
    Fallen person → ratio > 1 (wider than tall).
    """
    if pk.bbox is None:
        return None
    x1, y1, x2, y2 = pk.bbox
    w = x2 - x1
    h = y2 - y1
    if h < 1:
        return None
    return w / h
