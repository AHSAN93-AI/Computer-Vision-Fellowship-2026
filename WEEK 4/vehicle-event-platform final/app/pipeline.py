"""
app/pipeline.py — Master processing pipeline orchestrator.

Wires together all stages:
  Frame Acquisition → Detection → Tracking → Zone Analysis →
  Dwell/Occupancy → Rule Engine → Event Management → Evidence → Alerts

Runs the processing loop in a background thread so it doesn't block
the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.config import AppConfig, ZoneConfig
from app.vision.detector import VehicleDetector
from app.vision.tracker import VehicleTracker, TrackedVehicle
from app.vision.video_source import VideoSource, create_video_source
from app.analytics.zones import ZoneManager
from app.analytics.lines import LineCrossingMonitor
from app.analytics.dwell import DwellTracker
from app.analytics.occupancy import OccupancyMonitor
from app.analytics.performance import PerformanceMonitor, FrameTimings
from app.rules import build_rules
from app.rules.base_rule import BaseRule, RuleContext, RuleViolation
from app.events.event_manager import EventManager, Event
from app.events.evidence import EvidenceSaver, FrameBuffer
from app.events.alerts import AlertEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Observable pipeline state for the API/dashboard."""
    running: bool = False
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    frame_id: int = 0
    fps: float = 0.0
    active_tracks: int = 0
    active_events: int = 0
    zone_occupancies: Dict[str, Any] = field(default_factory=dict)
    line_counts: Dict[str, Any] = field(default_factory=dict)


class VideoPipeline:
    """
    Master processing pipeline.

    Orchestrates the full detection → tracking → analytics → rules → events flow.
    Runs in a background thread; communicates with FastAPI via callbacks and
    the shared PipelineState object.
    """

    def __init__(
        self,
        config: AppConfig,
        alert_engine: AlertEngine,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_frame_ready: Optional[Callable] = None,
    ):
        self._config = config
        self._alert_engine = alert_engine
        self._loop = loop
        self._on_frame_ready = on_frame_ready

        self.state = PipelineState()

        # ── Vision ──
        self._detector = VehicleDetector(
            model_path=config.model.path,
            confidence=config.model.confidence,
            classes=config.model.classes,
            input_resolution=config.model.input_resolution,
            device=config.model.device,
        )
        self._tracker = VehicleTracker(
            track_activation_threshold=config.tracker.track_activation_threshold,
            lost_track_buffer=config.tracker.lost_track_buffer,
            minimum_matching_threshold=config.tracker.minimum_matching_threshold,
        )

        # ── Analytics ──
        self._zone_manager = ZoneManager(config.zones)
        self._line_monitor = LineCrossingMonitor(config.zones)
        self._dwell_tracker = DwellTracker(config.dwell)
        self._occupancy_monitor = OccupancyMonitor(config.zones)
        self._perf_monitor = PerformanceMonitor(
            log_interval_seconds=config.performance.log_metrics_interval_seconds,
        )

        # ── Rules ──
        self._rules: List[BaseRule] = build_rules(config)

        # ── Events ──
        self._event_manager = EventManager(
            min_active_duration_seconds=config.events.min_active_duration_seconds,
            default_cooldown_seconds=config.events.default_cooldown_seconds,
            max_active_events=config.events.max_active_events,
            on_event_created=self._on_event_created,
            on_event_updated=self._on_event_updated,
            on_event_resolved=self._on_event_resolved,
        )

        # ── Evidence ──
        self._frame_buffer = FrameBuffer(max_frames=config.evidence.max_frame_buffer)
        self._evidence_saver = EvidenceSaver(
            storage_path=config.evidence.storage_path,
            save_full_frame=config.evidence.save_full_frame,
            save_crop=config.evidence.save_crop,
            save_metadata=config.evidence.save_metadata_json,
            frame_buffer=self._frame_buffer,
        )

        # ── Thread control ──
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._video_source: Optional[VideoSource] = None

        logger.info(
            f"Pipeline initialized: {len(self._rules)} rules, "
            f"{len(config.zones)} zones"
        )

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ─── Start/Stop ──────────────────────────────────────────────────────────

    def start_file(self, file_path: str) -> None:
        """Start processing from a video file."""
        self._start_source(
            create_video_source(
                "file",
                file_path=file_path,
                max_fps=self._config.video.max_fps,
                frame_skip=self._config.video.frame_skip,
            )
        )

    def start_webcam(self, device_index: int = 0) -> None:
        """Start processing from webcam."""
        self._start_source(
            create_video_source(
                "webcam",
                webcam_index=device_index,
                max_fps=self._config.video.max_fps,
                frame_skip=self._config.video.frame_skip,
            )
        )

    def start_rtsp(self, url: str) -> None:
        """Start processing from RTSP stream."""
        self._start_source(
            create_video_source(
                "rtsp",
                rtsp_url=url,
                max_fps=self._config.video.max_fps,
                frame_skip=self._config.video.frame_skip,
            )
        )

    def _start_source(self, source: VideoSource) -> None:
        """Start processing with the given video source."""
        if self.state.running:
            self.stop()
            time.sleep(0.5)  # Brief pause for thread cleanup

        self._video_source = source
        self._stop_event.clear()
        self._tracker.reset()

        self._thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="pipeline-thread",
        )
        self._thread.start()
        logger.info(f"Pipeline started with source: {source.source_id}")

    def stop(self) -> None:
        """Stop the processing loop."""
        self._stop_event.set()
        self.state.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._video_source:
            try:
                self._video_source.release()
            except Exception:
                pass
        logger.info("Pipeline stopped")

    # ─── Main processing loop ────────────────────────────────────────────────

    def _processing_loop(self) -> None:
        """Main frame processing loop. Runs in a background thread."""
        self.state.running = True
        self.state.source_type = self._video_source.source_type() if self._video_source else None
        self.state.source_id = self._video_source.source_id if self._video_source else None

        frame_count = 0
        fps_start = time.time()

        try:
            for frame, meta in self._video_source.frames():
                if self._stop_event.is_set():
                    break

                frame_count += 1
                self.state.frame_id = meta.frame_number
                t_start = time.perf_counter()

                try:
                    self._process_frame(frame, meta.frame_number, meta.timestamp)
                except Exception as e:
                    logger.error(f"Frame processing error on frame {meta.frame_number}: {e}")
                    continue

                # FPS calculation
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.state.fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

        except Exception as e:
            logger.error(f"Pipeline loop error: {e}")
        finally:
            self.state.running = False
            logger.info("Processing loop ended")

    def _process_frame(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> None:
        """Process a single frame through the full pipeline."""
        t0 = time.perf_counter()

        # ── Stage 1: Detection ──
        t_det_start = time.perf_counter()
        detections, det_stats = self._detector.detect(frame, frame_id)
        t_det = (time.perf_counter() - t_det_start) * 1000

        # ── Stage 2: Tracking ──
        t_track_start = time.perf_counter()
        tracked_vehicles = self._tracker.update(detections, frame, frame_id)
        t_track = (time.perf_counter() - t_track_start) * 1000
        self.state.active_tracks = len(tracked_vehicles)

        # ── Stage 3: Zone Analysis ──
        t_analytics_start = time.perf_counter()

        # Polygon zone containment
        zone_events = self._zone_manager.update(tracked_vehicles, frame_id)

        # Line crossing detection
        line_events = self._line_monitor.update(tracked_vehicles, frame_id)

        # Update dwell tracking
        zone_track_map: Dict[str, List[int]] = {}
        for zone_id in self._zone_manager.zone_ids():
            occ = self._zone_manager.get_occupancy(zone_id)
            if occ:
                zone_track_map[zone_id] = list(occ.occupant_track_ids)
        self._dwell_tracker.update_all(zone_track_map)

        # Confirm exits in dwell tracker
        for ze in zone_events:
            if ze.event_type.value == "ZONE_EXIT":
                self._dwell_tracker.confirm_exit(ze.track_id, ze.zone_id)

        # Update occupancy monitor
        zone_counts = {
            zid: occ.count
            for zid, occ in self._zone_manager.get_all_occupancies().items()
        }
        self._occupancy_monitor.update(zone_counts)

        # Record entries
        for ze in zone_events:
            if ze.event_type.value == "ZONE_ENTRY":
                self._occupancy_monitor.record_entry(ze.zone_id)

        t_analytics = (time.perf_counter() - t_analytics_start) * 1000

        # ── Stage 4: Rule Engine ──
        t_rules_start = time.perf_counter()

        rule_context = RuleContext(
            frame_id=frame_id,
            timestamp=timestamp,
            tracked_vehicles=tracked_vehicles,
            zone_occupancies=self._zone_manager.get_all_occupancies(),
            dwell_tracker=self._dwell_tracker,
            occupancy_monitor=self._occupancy_monitor,
            line_crossing_events=line_events,
        )

        all_violations: List[RuleViolation] = []
        for rule in self._rules:
            try:
                violations = rule.evaluate(rule_context)
                all_violations.extend(violations)
            except Exception as e:
                logger.error(f"Rule '{rule.rule_id}' evaluation error: {e}")

        t_rules = (time.perf_counter() - t_rules_start) * 1000

        # ── Stage 5: Event Management ──
        t_db_start = time.perf_counter()

        if all_violations:
            source_id = self._video_source.source_id if self._video_source else ""
            affected_events = self._event_manager.process_violations(
                all_violations, source_id=source_id
            )

        # Check for auto-resolution
        self._event_manager.check_auto_resolve()

        self.state.active_events = self._event_manager.active_count
        t_db = (time.perf_counter() - t_db_start) * 1000

        # ── Stage 6: Update state ──
        self.state.zone_occupancies = {
            zid: {
                "count": occ.count,
                "track_ids": list(occ.occupant_track_ids),
            }
            for zid, occ in self._zone_manager.get_all_occupancies().items()
        }
        self.state.line_counts = {
            zid: {"in": counts[0], "out": counts[1]}
            for zid, counts in self._line_monitor.get_all_counts().items()
        }

        # ── Stage 7: Frame buffer for evidence ──
        self._frame_buffer.add_frame(frame, timestamp)

        # ── Stage 8: Performance monitoring ──
        total_ms = (time.perf_counter() - t0) * 1000
        timings = FrameTimings(
            frame_id=frame_id,
            total_ms=total_ms,
            detection_ms=t_det,
            tracking_ms=t_track,
            analytics_ms=t_analytics,
            rules_ms=t_rules,
            db_ms=t_db,
            timestamp=time.time(),
        )
        self._perf_monitor.record_frame(timings)

        # ── Stage 9: WebSocket broadcast ──
        if self._on_frame_ready and self._loop:
            try:
                annotated = self._annotate_frame(frame, tracked_vehicles)
                _, buf = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self._config.dashboard.websocket_frame_quality],
                )
                frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                live_data = {
                    "frame_id": frame_id,
                    "fps": round(self.state.fps, 1),
                    "active_tracks": self.state.active_tracks,
                    "active_events": self.state.active_events,
                    "detection_count": len(detections),
                    "zone_occupancies": self.state.zone_occupancies,
                    "line_counts": self.state.line_counts,
                }

                asyncio.run_coroutine_threadsafe(
                    self._on_frame_ready(frame_b64, live_data),
                    self._loop,
                )
            except Exception as e:
                logger.debug(f"Frame broadcast error: {e}")

    # ─── Frame annotation ────────────────────────────────────────────────────

    def _annotate_frame(
        self, frame: np.ndarray, tracked_vehicles: List[TrackedVehicle]
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and zone overlays on the frame."""
        annotated = frame.copy()

        # Draw zone polygons
        for zc in self._config.zones:
            if zc.type == "polygon" and zc.polygon:
                pts = np.array(zc.polygon, dtype=np.int32)
                color = tuple(zc.color) if zc.color else (0, 200, 255)
                cv2.polylines(annotated, [pts], True, color, 2)
                # Zone label
                if len(pts) > 0:
                    cv2.putText(
                        annotated, zc.name,
                        (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )
            elif zc.type == "line_crossing" and zc.line:
                p1 = tuple(zc.line[0])
                p2 = tuple(zc.line[1])
                color = tuple(zc.color) if zc.color else (0, 255, 0)
                cv2.line(annotated, p1, p2, color, 2)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(
                    annotated, zc.name, (mid[0], mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        # Draw tracked vehicles
        for tv in tracked_vehicles:
            x1, y1, x2, y2 = tv.bbox
            color = (0, 255, 0)  # green for tracked vehicles

            # Red for stationary vehicles
            if tv.is_stationary():
                color = (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"#{tv.track_id} {tv.class_name} {tv.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # FPS overlay
        if self._config.performance.enable_fps_overlay:
            fps_text = f"FPS: {self.state.fps:.1f} | Tracks: {self.state.active_tracks} | Events: {self.state.active_events}"
            cv2.putText(
                annotated, fps_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        return annotated

    # ─── Event callbacks ─────────────────────────────────────────────────────

    def _on_event_created(self, event: Event) -> None:
        """Called when EventManager creates a new event."""
        # Capture evidence
        frames = self._frame_buffer.get_frames()
        if frames:
            last_frame, _ = frames[-1]
            evidence_path = self._evidence_saver.save(event, last_frame)
            if evidence_path:
                self._event_manager.set_evidence_path(event.event_id, evidence_path)

        # Send alert
        try:
            self._alert_engine.send_alert(event)
        except Exception as e:
            logger.error(f"Alert send error: {e}")

    def _on_event_updated(self, event: Event) -> None:
        """Called when an existing event is updated."""
        pass  # Could push DB update here if needed

    def _on_event_resolved(self, event: Event) -> None:
        """Called when an event is resolved."""
        logger.debug(f"Event resolved: {event.event_id}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def get_active_events(self) -> List[Event]:
        return self._event_manager.get_active_events()

    def acknowledge_event(self, event_id: str) -> Optional[Event]:
        return self._event_manager.acknowledge_event(event_id)

    def resolve_event(self, event_id: str) -> Optional[Event]:
        return self._event_manager.resolve_event(event_id)

    def get_performance_stats(self) -> dict:
        return self._perf_monitor.get_stats().to_dict()

    def get_zone_configs(self) -> List[ZoneConfig]:
        return self._zone_manager.get_all_zone_configs()

    def get_zone_config(self, zone_id: str) -> Optional[ZoneConfig]:
        return self._zone_manager.get_zone_config(zone_id)
