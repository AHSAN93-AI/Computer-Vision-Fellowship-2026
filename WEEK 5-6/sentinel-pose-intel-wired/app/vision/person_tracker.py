"""
app.vision.person_tracker — Multi-Person Tracking with Persistent IDs

Uses Ultralytics' built-in ByteTrack / BoT-SORT integration so we get
persistent person IDs without adding a second heavy dependency.

The tracker works by calling ``model.track()`` instead of ``model()``.
This module wraps that call and maintains a ``TrackedPerson`` state
object per ID that accumulates pose history, activity state, and
alert status (§4.4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.pose.keypoints import (
    PersonKeypoints,
    filter_keypoints_by_confidence,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    """Per-person state object (§4.4).

    Fields
    ------
    person_id : int
        Persistent ID assigned by the tracker.
    bbox : tuple
        Bounding box (x1, y1, x2, y2) in pixels.
    detection_confidence : float
        Detection confidence from the model.
    first_seen : float
        Monotonic timestamp when this person was first detected.
    last_seen : float
        Monotonic timestamp of the most recent detection.
    current_activity : str
        Name of the current recognised activity ("Unknown" initially).
    previous_activity : str
        The activity before the current one.
    activity_start_time : float
        Monotonic time when ``current_activity`` began.
    active_alerts : list
        Currently active alert IDs for this person.
    last_keypoints : PersonKeypoints or None
        Most recent keypoints (raw, before normalisation).
    frames_since_seen : int
        Counts up each frame the person is not detected; reset to 0 on detection.
    is_active : bool
        False when the person has been lost for > track_loss_timeout_frames.
    """

    person_id: int
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    detection_confidence: float = 0.0
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    current_activity: str = "Unknown"
    previous_activity: str = "Unknown"
    activity_start_time: float = field(default_factory=time.monotonic)
    active_alerts: List[str] = field(default_factory=list)
    last_keypoints: Optional[PersonKeypoints] = None
    frames_since_seen: int = 0
    is_active: bool = True
    frame_number: int = 0


class PersonTracker:
    """Manages tracked persons across frames.

    On each frame, call ``update()`` with the current frame to get
    detections with persistent IDs.  Internally it uses
    ``model.track(persist=True)`` and maintains ``TrackedPerson``
    state objects.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        from ultralytics import YOLO

        settings = get_settings()
        self._model_path = model_path or settings.pose_model
        self._conf_threshold = settings.detection_confidence_threshold
        self._kp_threshold = settings.keypoint_confidence_threshold
        self._tracker_type = settings.tracker_type
        self._track_loss_timeout = settings.track_loss_timeout_frames

        self._model = YOLO(self._model_path)
        self._persons: Dict[int, TrackedPerson] = {}
        self._frame_number: int = 0
        self._inference_time_ms: float = 0.0

        logger.info(
            "PersonTracker initialised (model=%s, tracker=%s, loss_timeout=%d frames)",
            self._model_path, self._tracker_type, self._track_loss_timeout,
        )

    def update(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """Run detection + tracking on one frame.

        Returns a list of ``PersonKeypoints`` (one per tracked person),
        each with ``track_id`` set to the persistent ID.

        Also updates internal ``TrackedPerson`` state objects — access
        them via ``get_person()`` or ``active_persons``.
        """
        self._frame_number += 1
        t0 = time.perf_counter()

        tracker_yaml = f"{self._tracker_type}.yaml"
        results = self._model.track(
            frame,
            conf=self._conf_threshold,
            persist=True,
            tracker=tracker_yaml,
            verbose=False,
        )

        self._inference_time_ms = (time.perf_counter() - t0) * 1000.0

        # Mark all existing persons as "not seen this frame"
        for p in self._persons.values():
            p.frames_since_seen += 1

        detected_persons: List[PersonKeypoints] = []
        now = time.monotonic()

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            kp_data = result.keypoints.data.cpu().numpy()    # (N, 17, 3)
            boxes = result.boxes.xyxy.cpu().numpy()          # (N, 4)
            confs = result.boxes.conf.cpu().numpy()          # (N,)

            # Track IDs — may be None if tracking fails for a frame
            track_ids = None
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)  # (N,)

            for i in range(len(kp_data)):
                raw_xy = kp_data[i, :, :2]
                raw_conf = kp_data[i, :, 2]
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

                # Assign track ID
                tid: Optional[int] = None
                if track_ids is not None and i < len(track_ids):
                    tid = int(track_ids[i])
                pk.track_id = tid

                # Update or create TrackedPerson state
                if tid is not None:
                    if tid not in self._persons:
                        self._persons[tid] = TrackedPerson(
                            person_id=tid,
                            first_seen=now,
                        )
                        logger.debug("New person tracked: #%03d", tid)

                    tp = self._persons[tid]
                    tp.bbox = pk.bbox
                    tp.detection_confidence = det_conf
                    tp.last_seen = now
                    tp.last_keypoints = pk
                    tp.frames_since_seen = 0
                    tp.is_active = True
                    tp.frame_number = self._frame_number

                detected_persons.append(pk)

        # -- Pass 1: Deactivate lost persons --
        # After `track_loss_timeout` frames without a detection, mark the
        # person as inactive so they no longer appear in activity processing.
        # We keep them in memory for a while in case the tracker re-acquires
        # the same ID (e.g. brief occlusion).
        for tp in self._persons.values():
            if tp.frames_since_seen > self._track_loss_timeout:
                tp.is_active = False

        # -- Pass 2: Delete stale tracks --
        # After 2x the timeout, the track is truly gone and unlikely to
        # be re-acquired.  Remove it to free memory and prevent the dict
        # from growing unboundedly during long sessions.
        stale_ids = [
            tid for tid, tp in self._persons.items()
            if tp.frames_since_seen > self._track_loss_timeout * 2
        ]
        for tid in stale_ids:
            logger.debug("Removing stale track: #%03d", tid)
            del self._persons[tid]

        return detected_persons

    def get_person(self, track_id: int) -> Optional[TrackedPerson]:
        """Get the state object for a tracked person by ID."""
        return self._persons.get(track_id)

    @property
    def active_persons(self) -> Dict[int, TrackedPerson]:
        """All currently active (not lost) persons."""
        return {tid: tp for tid, tp in self._persons.items() if tp.is_active}

    @property
    def all_persons(self) -> Dict[int, TrackedPerson]:
        """All persons including recently lost ones."""
        return dict(self._persons)

    @property
    def inference_time_ms(self) -> float:
        return self._inference_time_ms

    @property
    def frame_number(self) -> int:
        return self._frame_number
