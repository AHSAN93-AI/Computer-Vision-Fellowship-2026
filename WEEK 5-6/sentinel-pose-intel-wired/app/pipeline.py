"""
app.pipeline — Main Processing Pipeline Orchestrator

Ties together all components into the frame-by-frame pipeline:

  Video Source → Person Detection/Tracking → Pose Estimation →
  Sequence Buffer → Activity Recognition → Alert Engine →
  Database + Evidence Capture → Dashboard (via callback)

Runs in a background thread so the FastAPI event loop is not blocked.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.config import get_settings
from app.database.db import Database
from app.timeutil import monotonic_to_wall
from app.events.activity_manager import ActivityManager, PersonActivityState
from app.events.alerts import Alert, AlertEngine
from app.events.evidence import EvidenceCapture
from app.pose.keypoints import PersonKeypoints
from app.vision.fall_classifier import FallClassifier
from app.vision.person_tracker import PersonTracker, TrackedPerson
from app.vision.skeleton_renderer import draw_skeletons
from app.vision.video_source import VideoSource

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Snapshot of the pipeline's current state for the dashboard."""

    is_running: bool = False
    fps: float = 0.0
    inference_time_ms: float = 0.0
    frame_number: int = 0
    active_people: int = 0
    total_activities: int = 0
    total_falls: int = 0
    total_posture_risks: int = 0
    total_reps: int = 0
    source_info: str = ""
    persons: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class Pipeline:
    """Main processing pipeline.

    Usage::

        pipeline = Pipeline()
        pipeline.start("path/to/video.mp4")
        # ... later ...
        pipeline.stop()
    """

    def __init__(self) -> None:
        self._tracker: Optional[PersonTracker] = None
        self._activity_mgr: Optional[ActivityManager] = None
        self._alert_engine: Optional[AlertEngine] = None
        self._evidence: Optional[EvidenceCapture] = None
        self._db: Optional[Database] = None
        self._video: Optional[VideoSource] = None
        self._fall_classifier: FallClassifier = FallClassifier()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._status = PipelineStatus()
        self._lock = threading.Lock()

        # Callbacks for the dashboard
        self._on_frame: Optional[Callable[[np.ndarray, PipelineStatus], None]] = None
        self._on_alert: Optional[Callable[[Alert], None]] = None

        # Draw settings
        self._draw_skeleton = True
        self._draw_bbox = True

        # Stats
        self._total_falls = 0
        self._total_posture_risks = 0

    @property
    def status(self) -> PipelineStatus:
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        return self._status.is_running

    @property
    def db(self) -> Optional[Database]:
        """Public accessor for the database instance."""
        return self._db

    @property
    def alert_engine(self) -> Optional[AlertEngine]:
        """Public accessor for the alert engine."""
        return self._alert_engine

    def set_callbacks(
        self,
        on_frame: Optional[Callable] = None,
        on_alert: Optional[Callable] = None,
    ) -> None:
        self._on_frame = on_frame
        self._on_alert = on_alert

    def set_draw_options(self, skeleton: bool = True, bbox: bool = True) -> None:
        self._draw_skeleton = skeleton
        self._draw_bbox = bbox

    def start(self, source: Optional[str] = None) -> bool:
        """Start the pipeline in a background thread.

        Parameters
        ----------
        source:
            Video source string.  Defaults to config value.

        Returns
        -------
        True if started successfully.
        """
        if self._status.is_running:
            logger.warning("Pipeline already running")
            return False

        settings = get_settings()
        source_str = source or settings.video_source

        # Initialise components
        try:
            self._db = Database()
            self._db.connect()

            self._evidence = EvidenceCapture()
            self._alert_engine = AlertEngine(self._db, on_alert=self._on_alert)
            self._activity_mgr = ActivityManager()
            self._tracker = PersonTracker()
            self._video = VideoSource(source_str)

            if not self._video.open():
                self._status.error = f"Cannot open video source: {source_str}"
                logger.error(self._status.error)
                return False

        except Exception as e:
            self._status.error = f"Pipeline init failed: {e}"
            logger.exception(self._status.error)
            return False

        self._stop_event.clear()
        self._status = PipelineStatus(is_running=True, source_info=source_str)
        self._total_falls = 0
        self._total_posture_risks = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pipeline")
        self._thread.start()
        logger.info("Pipeline started with source: %s", source_str)
        return True

    def stop(self) -> None:
        """Stop the pipeline."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._video:
            self._video.release()
        if self._db:
            self._db.close()
        with self._lock:
            self._status.is_running = False
        logger.info("Pipeline stopped")

    def _run_loop(self) -> None:
        """Main frame-processing loop (runs in background thread)."""
        settings = get_settings()
        frame_time_target = 1.0 / 30.0  # target ~30 FPS processing
        fps_counter = _FPSCounter()

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                # 1. Read frame
                frame = self._video.read_frame()
                if frame is None:
                    if self._video.source_type.value == "file":
                        logger.info("Video file ended")
                        break
                    else:
                        logger.warning("Frame read failed, retrying...")
                        time.sleep(0.1)
                        continue

                # 2. Detection + Tracking
                persons = self._tracker.update(frame)
                frame_num = self._tracker.frame_number

                # 3. Per-person activity processing
                alert_person_ids: set = set()
                person_data: Dict[int, Dict[str, Any]] = {}

                for pk in persons:
                    if pk.track_id is None:
                        continue

                    tid = pk.track_id
                    tp = self._tracker.get_person(tid)
                    if tp is None:
                        continue

                    # Run activity recognition
                    pas = self._activity_mgr.process_person(tid, pk, frame_num)

                    # Update tracked person state
                    tp.current_activity = pas.current_activity
                    tp.previous_activity = pas.previous_activity
                    tp.activity_start_time = pas.activity_start_time

                    # Check for fall alerts — ensemble of the rule-based
                    # state machine (pose geometry over time) and the CNN
                    # classifier (single-frame appearance), each with its
                    # own temporal confirmation.
                    fall_sm = self._activity_mgr.get_fall_state(tid)
                    rule_active = bool(fall_sm and fall_sm.is_active)
                    rule_confidence = fall_sm.confidence if fall_sm else 0.0

                    cnn_label, cnn_frame_conf = "Unknown", 0.0
                    cnn_confirmed, cnn_confidence = False, 0.0
                    if pk.bbox:
                        cnn_result = self._fall_classifier.observe(tid, frame, pk.bbox)
                        if cnn_result:
                            cnn_label, cnn_frame_conf = cnn_result
                        cnn_confirmed, cnn_confidence = self._fall_classifier.is_confirmed(tid)

                    is_fall = rule_active or cnn_confirmed
                    combined_confidence = max(rule_confidence, cnn_confidence)

                    if is_fall:
                        alert_person_ids.add(tid)
                        # Capture evidence
                        ev_path = None
                        if self._evidence and pk.bbox:
                            ev_path = self._evidence.save_frame(
                                frame, f"fall_{tid}", tid, "fall"
                            )
                        alert = self._alert_engine.check_fall(
                            person_id=tid,
                            is_fall_active=True,
                            confidence=combined_confidence,
                            evidence_path=ev_path,
                            camera_source=settings.video_source,
                        )
                        if alert:
                            self._total_falls += 1

                    # Check ergonomic alerts
                    ergo = pas.ergonomic_monitor
                    if ergo.is_bend_risk or ergo.is_crouch_risk:
                        ev_path = None
                        if self._evidence and pk.bbox:
                            ev_path = self._evidence.save_frame(
                                frame, f"ergo_{tid}", tid, "posture_risk"
                            )
                        alert = self._alert_engine.check_ergonomic(
                            person_id=tid,
                            is_bend_risk=ergo.is_bend_risk,
                            is_crouch_risk=ergo.is_crouch_risk,
                            bend_duration=ergo.bend_duration,
                            crouch_duration=ergo.crouch_duration,
                            evidence_path=ev_path,
                            camera_source=settings.video_source,
                        )
                        if alert:
                            self._total_posture_risks += 1

                    # Check inactivity
                    activity_dur = time.monotonic() - pas.activity_start_time
                    self._alert_engine.check_inactivity(
                        person_id=tid,
                        current_activity=pas.current_activity,
                        activity_duration=activity_dur,
                        camera_source=settings.video_source,
                    )

                    # Store activity events in DB for completed timeline entries
                    while pas.timeline:
                        event = pas.timeline.pop(0)
                        try:
                            self._db.insert_activity_event(
                                person_id=event.person_id,
                                activity_type=event.activity_type,
                                start_time=monotonic_to_wall(event.start_time),
                                end_time=monotonic_to_wall(event.end_time) if event.end_time is not None else None,
                                duration=event.duration,
                                confidence=event.confidence,
                                source_id=settings.video_source,
                            )
                        except Exception:
                            logger.warning("Failed to insert activity event", exc_info=True)

                    # Build person summary for dashboard
                    person_data[tid] = {
                        "person_id": tid,
                        "activity": pas.current_activity,
                        "activity_display": pas.current_activity_display,
                        "confidence": pas.current_confidence,
                        "squat_count": pas.squat_count,
                        "bbox": pk.bbox,
                        "ergo_bend_risk": ergo.is_bend_risk,
                        "ergo_crouch_risk": ergo.is_crouch_risk,
                        "activity_duration": activity_dur,
                        "fall_rule_active": rule_active,
                        "fall_cnn_label": cnn_label,
                        "fall_cnn_confidence": round(cnn_frame_conf, 3),
                        "fall_cnn_confirmed": cnn_confirmed,
                    }

                # 4. Draw overlays
                annotated = frame.copy()
                if self._draw_skeleton or self._draw_bbox:
                    draw_skeletons(
                        annotated,
                        persons,
                        alert_person_ids=alert_person_ids,
                        draw_bbox=self._draw_bbox,
                        draw_keypoints=self._draw_skeleton,
                        draw_bones=self._draw_skeleton,
                        draw_labels=True,
                    )

                # Draw activity labels on frame
                for pk in persons:
                    if pk.track_id is not None and pk.bbox is not None and pk.track_id in person_data:
                        pd = person_data[pk.track_id]
                        label = f"{pd['activity_display']}"
                        if pd['squat_count'] > 0:
                            label += f" (reps:{pd['squat_count']})"
                        x1, y2 = int(pk.bbox[0]), int(pk.bbox[3])
                        cv2.putText(
                            annotated, label, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 170), 1, cv2.LINE_AA,
                        )

                # 5. Update status
                fps_counter.tick()
                with self._lock:
                    self._status.fps = fps_counter.fps
                    self._status.inference_time_ms = self._tracker.inference_time_ms
                    self._status.frame_number = frame_num
                    self._status.active_people = len(self._tracker.active_persons)
                    active_ids = set(self._tracker.active_persons.keys())
                    for stale_tid in self._fall_classifier.tracked_ids():
                        if stale_tid not in active_ids:
                            self._fall_classifier.forget(stale_tid)
                    self._status.persons = person_data
                    self._status.total_falls = self._total_falls
                    self._status.total_posture_risks = self._total_posture_risks
                    self._status.active_alerts = [
                        {
                            "alert_id": a.alert_id,
                            "person_id": a.person_id,
                            "alert_type": a.alert_type,
                            "severity": a.severity,
                            "status": a.status,
                            "confidence": a.confidence,
                            "timestamp": monotonic_to_wall(a.timestamp),
                        }
                        for a in self._alert_engine.get_active_alerts()
                    ]
                    self._status.total_reps = sum(
                        pd.get("squat_count", 0) for pd in person_data.values()
                    )
                    # Count total activities from DB
                    try:
                        self._status.total_activities = self._db.count_activity_events()
                    except Exception:
                        logger.debug("DB count_activity_events failed", exc_info=True)

                # 6. Notify dashboard
                if self._on_frame:
                    try:
                        self._on_frame(annotated, self._status)
                    except Exception:
                        logger.debug("on_frame callback failed", exc_info=True)

                # Frame pacing
                elapsed = time.perf_counter() - t0
                if elapsed < frame_time_target:
                    time.sleep(frame_time_target - elapsed)

        except Exception:
            logger.exception("Pipeline loop crashed")
            with self._lock:
                self._status.error = "Pipeline crashed - check logs"
        finally:
            with self._lock:
                self._status.is_running = False
            logger.info("Pipeline loop exited")


class _FPSCounter:
    """Simple EMA-based FPS counter."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self._alpha * instant_fps + (1 - self._alpha) * self._fps
        self._last_time = now

    @property
    def fps(self) -> float:
        return self._fps
