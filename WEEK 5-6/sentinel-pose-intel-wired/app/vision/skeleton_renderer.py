"""
app.vision.skeleton_renderer — Skeleton Drawing Utility

Draws pose skeletons, bounding boxes, and labels on video frames.
Separated from ``pose_estimator.py`` so that the pipeline can draw
overlays without importing the YOLO model class.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from app.pose.keypoints import (
    BBOX_COLOR_ALERT,
    BBOX_COLOR_NORMAL,
    KEYPOINT_COLOR,
    LABEL_BG_COLOR,
    SKELETON_COLOR,
    SKELETON_CONNECTIONS,
    PersonKeypoints,
)

logger = logging.getLogger(__name__)


def draw_skeletons(
    frame: np.ndarray,
    persons: List[PersonKeypoints],
    *,
    alert_person_ids: Optional[set] = None,
    draw_bbox: bool = True,
    draw_keypoints: bool = True,
    draw_bones: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw pose skeletons on a frame (in-place for speed).

    Parameters
    ----------
    frame:
        BGR image to draw on.
    persons:
        Detected persons with keypoints.
    alert_person_ids:
        Set of track IDs currently in alert state -- drawn with alert colour.
    draw_bbox, draw_keypoints, draw_bones, draw_labels:
        Toggles for each visual element.

    Returns
    -------
    The same frame reference (modified in-place).
    """
    alert_ids = alert_person_ids or set()

    for pk in persons:
        is_alert = pk.track_id is not None and pk.track_id in alert_ids
        bbox_color = BBOX_COLOR_ALERT if is_alert else BBOX_COLOR_NORMAL

        # -- Bounding box --
        if draw_bbox and pk.bbox is not None:
            x1, y1, x2, y2 = pk.bbox
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                bbox_color,
                2,
            )

        # -- Label (track ID + confidence) --
        if draw_labels and pk.bbox is not None:
            label_parts = []
            if pk.track_id is not None:
                label_parts.append(f"#{pk.track_id:03d}")
            label_parts.append(f"{pk.detection_confidence:.0%}")
            label = " ".join(label_parts)

            x1, y1 = int(pk.bbox[0]), int(pk.bbox[1])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), LABEL_BG_COLOR, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1, cv2.LINE_AA,
            )

        # -- Skeleton bones --
        if draw_bones:
            for idx_a, idx_b in SKELETON_CONNECTIONS:
                kp_a = pk.get(idx_a)
                kp_b = pk.get(idx_b)
                if kp_a is not None and kp_b is not None:
                    cv2.line(
                        frame,
                        kp_a.as_int_tuple(),
                        kp_b.as_int_tuple(),
                        SKELETON_COLOR,
                        2,
                        cv2.LINE_AA,
                    )

        # -- Keypoint circles --
        if draw_keypoints:
            for kp in pk.keypoints:
                if kp.is_visible:
                    cv2.circle(frame, kp.as_int_tuple(), 4, KEYPOINT_COLOR, -1, cv2.LINE_AA)

    return frame
