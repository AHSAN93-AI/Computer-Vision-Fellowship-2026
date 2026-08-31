"""
app/vision/tracker.py — Multi-object tracking with ByteTrack.

Maintains persistent track IDs across frames.
Per-track state: ID, class, bbox, centroid, history, first/last seen, dwell info.
Uses supervision.ByteTrack (no ReID model needed).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.vision.detector import Detection

logger = logging.getLogger(__name__)

# Max position history entries per track
MAX_POSITION_HISTORY = 30


@dataclass
class TrackedVehicle:
    """
    A vehicle with a persistent track ID across frames.

    position_history: deque of (x, y) centroids, newest last.
    """
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2)
    confidence: float
    first_seen: float                    # epoch timestamp
    last_seen: float                     # epoch timestamp
    frame_id: int

    # Zone state (populated by ZoneManager)
    current_zone: Optional[str] = None
    previous_zone: Optional[str] = None
    zone_entry_time: Optional[float] = None

    # Position history for movement analysis
    position_history: deque = field(default_factory=lambda: deque(maxlen=MAX_POSITION_HISTORY))

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen

    def is_stationary(self, px_threshold: float = 10.0, window: int = 15) -> bool:
        """
        Returns True if the centroid has moved less than px_threshold pixels
        over the last `window` frames in the history.
        """
        if len(self.position_history) < min(window, 2):
            return False
        recent = list(self.position_history)[-window:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        return (x_range < px_threshold) and (y_range < px_threshold)

    def movement_vector(self, window: int = 10) -> Tuple[float, float]:
        """Returns (dx, dy) between oldest and newest point in the last `window` frames."""
        if len(self.position_history) < 2:
            return (0.0, 0.0)
        recent = list(self.position_history)[-window:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx, dy)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "confidence": round(self.confidence, 3),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "age_seconds": round(self.age_seconds, 2),
            "current_zone": self.current_zone,
            "is_stationary": self.is_stationary(),
        }


class VehicleTracker:
    """
    Wraps supervision.ByteTrack to maintain persistent vehicle tracks.

    Converts YOLO detections → supervision Detections → ByteTrack → TrackedVehicles.
    Maintains a registry of known tracks with their full state.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold

        self._tracker = None
        self._track_registry: Dict[int, TrackedVehicle] = {}  # track_id → TrackedVehicle
        self._init_tracker()

    def _init_tracker(self) -> None:
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
            )
            logger.info("ByteTrack initialized successfully")
        except ImportError as e:
            logger.error(f"supervision not installed: {e}. Run: pip install supervision")
            self._tracker = None
        except Exception as e:
            logger.error(f"Failed to initialize ByteTrack: {e}")
            self._tracker = None

    def update(
        self, detections: List[Detection], frame: np.ndarray, frame_id: int
    ) -> List[TrackedVehicle]:
        """
        Update tracker with new detections. Returns currently tracked vehicles.

        On tracker failure: logs error, resets tracker, returns [] for this frame.
        """
        if self._tracker is None:
            logger.warning("Tracker not available — attempting re-init")
            self._init_tracker()
            if self._tracker is None:
                return []

        if not detections:
            # Still update tracker with empty detections to decrement lost buffers
            try:
                import supervision as sv
                empty = sv.Detections.empty()
                self._tracker.update_with_detections(empty)
            except Exception:
                pass
            return []

        try:
            import supervision as sv

            # Build supervision Detections from our Detection list
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            confidences = np.array([d.confidence for d in detections], dtype=np.float32)
            class_ids = np.array([d.class_id for d in detections], dtype=int)

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids,
            )

            # Run ByteTrack update
            tracked = self._tracker.update_with_detections(sv_detections)

        except Exception as e:
            logger.error(f"ByteTrack update failed on frame {frame_id}: {e} — resetting tracker")
            try:
                self._init_tracker()
            except Exception:
                pass
            return []

        now = time.time()
        result: List[TrackedVehicle] = []

        if tracked is None or len(tracked) == 0:
            return []

        for i in range(len(tracked)):
            try:
                xyxy_i = tracked.xyxy[i]
                x1, y1, x2, y2 = int(xyxy_i[0]), int(xyxy_i[1]), int(xyxy_i[2]), int(xyxy_i[3])
                bbox = (x1, y1, x2, y2)

                track_id = int(tracked.tracker_id[i])
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                from app.vision.detector import COCO_ID_TO_NAME
                cls_name = COCO_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if track_id in self._track_registry:
                    # Update existing track
                    tv = self._track_registry[track_id]
                    tv.bbox = bbox
                    tv.confidence = conf
                    tv.last_seen = now
                    tv.frame_id = frame_id
                    tv.position_history.append((cx, cy))
                else:
                    # New track
                    tv = TrackedVehicle(
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        bbox=bbox,
                        confidence=conf,
                        first_seen=now,
                        last_seen=now,
                        frame_id=frame_id,
                        position_history=deque([(cx, cy)], maxlen=MAX_POSITION_HISTORY),
                    )
                    self._track_registry[track_id] = tv
                    logger.debug(f"New track: ID={track_id} class={cls_name}")

                result.append(tv)
            except Exception as e:
                logger.debug(f"Error processing tracked detection {i}: {e}")
                continue

        # Prune stale tracks from registry (not seen for > lost_track_buffer equivalent)
        stale_threshold = now - (self.lost_track_buffer / 30.0)  # approx seconds
        stale_ids = [
            tid for tid, tv in self._track_registry.items()
            if tv.last_seen < stale_threshold
        ]
        for tid in stale_ids:
            logger.debug(f"Pruning stale track: ID={tid}")
            del self._track_registry[tid]

        return result

    def get_track(self, track_id: int) -> Optional[TrackedVehicle]:
        return self._track_registry.get(track_id)

    def get_all_tracks(self) -> List[TrackedVehicle]:
        return list(self._track_registry.values())

    def reset(self) -> None:
        """Reset tracker state (e.g., on new video source)."""
        self._track_registry.clear()
        self._init_tracker()
        logger.info("Tracker reset")

    @property
    def active_track_count(self) -> int:
        return len(self._track_registry)
