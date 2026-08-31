"""
app.vision.pose_estimator — Pose Estimation Interface & YOLO Implementation

Defines a ``PoseEstimator`` protocol so the backend can swap between
YOLO-Pose and (future) MediaPipe without changing downstream code.

The default implementation, ``YoloPoseEstimator``, wraps the Ultralytics
YOLO-Pose API.  It:
  • Loads the model once on construction (auto-downloads if needed).
  • Runs inference on a single frame.
  • Applies keypoint confidence filtering (§4.3).
  • Returns a list of ``PersonKeypoints`` — one per detected person.
  • Provides a ``draw_skeleton()`` helper for the overlay toggle (§4.2).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np

from app.config import get_settings
from app.pose.keypoints import (
    NUM_KEYPOINTS,
    PersonKeypoints,
    filter_keypoints_by_confidence,
)

# Re-export draw_skeletons for backwards compatibility with existing imports.
# The implementation now lives in skeleton_renderer.py to avoid coupling
# the drawing utility to the YOLO model class.
from app.vision.skeleton_renderer import draw_skeletons  # noqa: F401

logger = logging.getLogger(__name__)


# -- Protocol (interface) --
@runtime_checkable
class PoseEstimator(Protocol):
    """Abstract interface so MediaPipe or another backend can be swapped in."""

    def estimate(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """Run pose estimation on a single BGR frame.

        Returns one ``PersonKeypoints`` per detected person.
        """
        ...

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        ...


# -- YOLO-Pose implementation --

class YoloPoseEstimator:
    """Wraps Ultralytics YOLO-Pose for single-frame pose estimation.

    Parameters
    ----------
    model_path:
        Ultralytics model name or local ``.pt`` path.
        Defaults to the value in ``config.pose_model``.
    device:
        ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
        Auto-detected if not provided.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO  # deferred to keep import time low

        settings = get_settings()
        self._model_path = model_path or settings.pose_model
        self._device = device or "cpu"
        self._conf_threshold = settings.detection_confidence_threshold
        self._kp_threshold = settings.keypoint_confidence_threshold
        self._inference_time_ms: float = 0.0

        logger.info("Loading YOLO-Pose model: %s (device=%s)", self._model_path, self._device)
        try:
            self._model = YOLO(self._model_path)
            logger.info("YOLO-Pose model loaded successfully")
        except Exception:
            logger.exception("Failed to load YOLO-Pose model: %s", self._model_path)
            raise

    def estimate(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """Run pose estimation on a single BGR frame.

        Returns one ``PersonKeypoints`` per detected person whose
        detection confidence is above the configured threshold.
        Keypoints below the keypoint confidence threshold are marked
        as missing (NaN coordinates, confidence = 0).
        """
        t0 = time.perf_counter()

        results = self._model(
            frame,
            conf=self._conf_threshold,
            verbose=False,
            device=self._device,
        )

        self._inference_time_ms = (time.perf_counter() - t0) * 1000.0

        persons: List[PersonKeypoints] = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            kp_data = result.keypoints.data.cpu().numpy()    # (N, 17, 3)
            boxes = result.boxes.xyxy.cpu().numpy()          # (N, 4)
            confs = result.boxes.conf.cpu().numpy()          # (N,)

            for i in range(len(kp_data)):
                raw_xy = kp_data[i, :, :2]       # (17, 2)
                raw_conf = kp_data[i, :, 2]       # (17,)
                det_conf = float(confs[i])

                pk = filter_keypoints_by_confidence(
                    raw_xy, raw_conf, self._kp_threshold
                )
                pk.bbox = (
                    float(boxes[i, 0]),
                    float(boxes[i, 1]),
                    float(boxes[i, 2]),
                    float(boxes[i, 3]),
                )
                pk.detection_confidence = det_conf
                persons.append(pk)

        return persons

    @property
    def inference_time_ms(self) -> float:
        return self._inference_time_ms

    @property
    def model_name(self) -> str:
        return self._model_path
