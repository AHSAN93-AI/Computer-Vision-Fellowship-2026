"""
app.pose.normalization — Pose Normalisation (§4.7)

Makes activity logic invariant to camera distance, resolution, and person height
by transforming raw pixel keypoints into a canonical coordinate system:

1. **Origin**: midpoint of left and right hip → (0, 0).
2. **Scale**: divide all coordinates by the torso length (mid-hip to
   mid-shoulder distance).  If torso is not visible, fall back to
   shoulder width.

After normalisation, a person standing close to the camera and one far
away produce keypoint coordinates of roughly the same magnitude,
so fixed thresholds in activity rules work regardless of scale.

**Why this matters**:
  • Without normalisation, "walking velocity > 0.015" would fire for
    a far-away person with tiny pixel displacements but fail for a
    nearby person making large pixel movements.
  • Torso-length scaling is preferred over bounding-box scaling because
    bounding boxes change shape with pose (a raised arm extends the box
    but doesn't change the person's actual proportions).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from app.pose.keypoints import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NUM_KEYPOINTS,
    PersonKeypoints,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from app.pose import angles as angle_utils

logger = logging.getLogger(__name__)


def normalize_keypoints(pk: PersonKeypoints) -> Optional[np.ndarray]:
    """Normalise keypoints to a hip-centred, torso-scaled coordinate system.

    Parameters
    ----------
    pk:
        Raw ``PersonKeypoints`` in pixel coordinates.

    Returns
    -------
    numpy array of shape ``(17, 2)`` in normalised coordinates, or
    ``None`` if normalisation is not possible (both hips and both
    shoulders are missing).  Missing keypoints are set to ``(NaN, NaN)``.
    """
    # 1. Compute origin (mid-hip)
    origin = pk.midpoint(LEFT_HIP, RIGHT_HIP)
    if origin is None:
        # Fallback: try mid-shoulder as origin
        origin = pk.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
        if origin is None:
            return None

    ox, oy = origin

    # 2. Compute scale factor
    scale = angle_utils.torso_length(pk)
    if scale is None or scale < 1.0:
        # Fallback: shoulder width
        scale = angle_utils.shoulder_width(pk)
        if scale is None or scale < 1.0:
            # Cannot normalise meaningfully — return None
            return None

    # 3. Translate and scale
    result = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
    for kp in pk.keypoints:
        if kp.is_visible:
            result[kp.index, 0] = (kp.x - ox) / scale
            result[kp.index, 1] = (kp.y - oy) / scale

    return result



def compute_velocity_from_raw(
    hip_prev: Optional[Tuple[float, float]],
    hip_curr: Optional[Tuple[float, float]],
    scale: Optional[float],
) -> Optional[float]:
    """Normalised hip velocity between two consecutive frames.

    Parameters
    ----------
    hip_prev, hip_curr:
        Mid-hip pixel positions from the previous and current frame.
    scale:
        Current torso length (or shoulder width) in pixels.

    Returns
    -------
    Normalised displacement (dimensionless).  A value of 0.015 means
    the hip moved ~1.5% of the torso length in one frame.
    """
    if hip_prev is None or hip_curr is None or scale is None or scale < 1.0:
        return None
    dx = hip_curr[0] - hip_prev[0]
    dy = hip_curr[1] - hip_prev[1]
    return math.hypot(dx, dy) / scale
