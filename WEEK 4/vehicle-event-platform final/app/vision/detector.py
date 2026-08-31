"""
app/vision/detector.py — Vehicle detection module.

Pure detection: frame in → List[Detection] out.
No zone/rule/event knowledge. Uses pretrained COCO YOLO model filtered to
vehicle classes (car, truck, bus, motorcycle).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO class name → class ID mapping (subset we care about)
COCO_VEHICLE_CLASSES: Dict[str, int] = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

# Reverse: class_id → class_name
COCO_ID_TO_NAME: Dict[int, str] = {v: k for k, v in COCO_VEHICLE_CLASSES.items()}


class ModelLoadError(Exception):
    """Raised when the YOLO model cannot be loaded."""
    pass


@dataclass
class Detection:
    """A single object detection from one frame."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) absolute pixels
    class_id: int
    class_name: str
    confidence: float
    frame_id: int
    timestamp: float

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


@dataclass
class DetectionStats:
    """Per-frame detection statistics."""
    frame_id: int
    inference_ms: float
    num_detections: int
    num_filtered: int   # after class filter
    timestamp: float


class VehicleDetector:
    """
    Wraps Ultralytics YOLO for vehicle detection.

    Responsibilities:
    - Load and cache the YOLO model
    - Run inference on a frame
    - Filter detections to configured vehicle classes
    - Return raw Detection objects (no downstream logic)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.40,
        classes: Optional[List[str]] = None,
        input_resolution: Tuple[int, int] = (640, 640),
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.input_resolution = input_resolution

        # Resolve which COCO class IDs to keep
        target_classes = classes or list(COCO_VEHICLE_CLASSES.keys())
        self._target_class_ids: List[int] = []
        for cls_name in target_classes:
            cls_name_lower = cls_name.lower()
            if cls_name_lower in COCO_VEHICLE_CLASSES:
                self._target_class_ids.append(COCO_VEHICLE_CLASSES[cls_name_lower])
            else:
                logger.warning(f"Unknown class '{cls_name}' — ignored")

        self._model = None
        self._last_stats: Optional[DetectionStats] = None
        self._total_frames = 0

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model. Raises ModelLoadError on failure."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics package not installed. Run: pip install ultralytics"
            ) from e

        model_p = Path(self.model_path)
        # If not an absolute path, try relative to CWD; ultralytics will auto-download
        logger.info(f"Loading YOLO model: {self.model_path} on device={self.device}")
        try:
            self._model = YOLO(str(model_p))
            # Warm-up pass with a blank frame to load weights into memory
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                conf=self.confidence,
                classes=self._target_class_ids,
                device=self.device,
                verbose=False,
                imgsz=self.input_resolution[0],
            )
            logger.info(f"YOLO model loaded successfully: {self.model_path}")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YOLO model '{self.model_path}': {e}\n"
                "Ensure the model file exists or internet is available for auto-download."
            ) from e

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Tuple[List[Detection], DetectionStats]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3)
            frame_id: monotonic frame counter from the source

        Returns:
            (detections, stats)  — detections is [] on failure, not an exception
        """
        if self._model is None:
            logger.error("Model not loaded — skipping detection")
            stats = DetectionStats(
                frame_id=frame_id, inference_ms=0, num_detections=0,
                num_filtered=0, timestamp=time.time()
            )
            return [], stats

        t0 = time.perf_counter()
        timestamp = time.time()
        self._total_frames += 1

        try:
            results = self._model.predict(
                frame,
                conf=self.confidence,
                classes=self._target_class_ids,
                device=self.device,
                verbose=False,
                imgsz=self.input_resolution[0],
            )
        except Exception as e:
            logger.warning(f"Detection inference failed on frame {frame_id}: {e} — skipping")
            stats = DetectionStats(
                frame_id=frame_id, inference_ms=0, num_detections=0,
                num_filtered=0, timestamp=timestamp
            )
            return [], stats

        inference_ms = (time.perf_counter() - t0) * 1000
        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    try:
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = (
                            int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        )
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        cls_name = COCO_ID_TO_NAME.get(cls_id, f"class_{cls_id}")

                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf,
                            frame_id=frame_id,
                            timestamp=timestamp,
                        ))
                    except Exception as e:
                        logger.debug(f"Error parsing detection box {i}: {e}")
                        continue

        stats = DetectionStats(
            frame_id=frame_id,
            inference_ms=inference_ms,
            num_detections=len(detections),
            num_filtered=0,  # already filtered by YOLO class list
            timestamp=timestamp,
        )
        self._last_stats = stats
        return detections, stats

    @property
    def last_stats(self) -> Optional[DetectionStats]:
        return self._last_stats

    @property
    def target_class_ids(self) -> List[int]:
        return self._target_class_ids

    def update_confidence(self, confidence: float) -> None:
        """Update confidence threshold at runtime."""
        self.confidence = max(0.01, min(1.0, confidence))
        logger.info(f"Detection confidence updated to {self.confidence}")
