# Requirements Traceability Map

This document maps every requirement from the project specification to the implementing source file(s), class/function names, and a one-line explanation.

---

## Section 6: Video Input & Source Management

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 6A. File input support | `app/vision/video_source.py` | `FileVideoSource` | Opens and reads frames from local video files (.mp4, .avi, .mov, etc.) |
| 6A. Webcam input support | `app/vision/video_source.py` | `WebcamVideoSource` | Opens and reads frames from a USB/built-in webcam by device index |
| 6A. RTSP input support | `app/vision/video_source.py` | `RTSPVideoSource` | Connects to an RTSP stream URL and reads frames with reconnection |
| 6B. RTSP reconnection | `app/vision/video_source.py` | `RTSPVideoSource._handle_read_failure` | Exponential backoff reconnection on RTSP disconnection |
| 6C. Source switching via API | `app/api/video.py` | `upload_video`, `start_webcam`, `start_rtsp`, `stop_video` | REST endpoints to start/stop different video sources |
| 6D. File upload | `app/api/video.py` | `upload_video` | Saves uploaded file to `./uploads/` and starts pipeline processing |

---

## Section 7: Object Detection

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 7A. YOLO-based detection | `app/vision/detector.py` | `VehicleDetector` | Loads YOLOv8 model and runs inference on each frame |
| 7B. Configurable confidence | `app/config.py`, `config.yaml` | `ModelConfig.confidence` | Confidence threshold configurable via YAML |
| 7C. Configurable classes | `app/config.py`, `config.yaml` | `ModelConfig.classes` | Vehicle class filter (car, truck, bus, motorcycle) |
| 7D. Configurable resolution | `app/config.py`, `config.yaml` | `ModelConfig.input_resolution` | Input resolution for YOLO inference |
| 7E. No event logic in detection | `app/vision/detector.py` | N/A (architecture constraint) | Detection only returns bounding boxes; all event logic goes through rule engine |

---

## Section 8: Multi-Object Tracking

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 8A. ByteTrack tracking | `app/vision/tracker.py` | `VehicleTracker` | ByteTrack integration via `supervision` library |
| 8B. Persistent track IDs | `app/vision/tracker.py` | `TrackedVehicle.track_id` | Stable track IDs maintained across frames |
| 8C. Per-track state | `app/vision/tracker.py` | `TrackedVehicle` | Stores class, bbox, confidence, position history, first/last seen, centroid, trail, stationary check |
| 8D. Configurable tracker params | `app/config.py`, `config.yaml` | `TrackerConfig` | ByteTrack params: activation threshold, lost track buffer, matching threshold |

---

## Section 9: Zone Configuration

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 9A. ≥3 zones defined | `config.yaml` | `zones` | Three zones: entrance_lane (line), parking_area (polygon), no_parking_zone (polygon) |
| 9B. Polygon-based zones | `app/analytics/zones.py` | `ZoneManager`, `point_in_polygon` | Point-in-polygon containment test for each tracked vehicle |
| 9C. Zone config via YAML | `app/config.py` | `ZoneConfig` | Zone definitions with id, name, type, polygon/line, capacity, colors |
| 9D. Zone API | `app/api/video.py` | `zones_router` | REST endpoints to list and query zone configurations |

---

## Section 10: Zone Entry/Exit Events

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 10A. State-transition events | `app/analytics/zones.py` | `ZoneManager.update` | Emits ENTRY/EXIT events only on state change (not every frame) |
| 10B. Per-track zone state | `app/analytics/zones.py` | `ZoneManager._track_zones` | Tracks which zone each vehicle was in on the previous frame |
| 10C. ZoneEvent dataclass | `app/analytics/zones.py` | `ZoneEvent`, `ZoneEventType` | Structured event with zone_id, track_id, event_type, timestamp |

---

## Section 11: Dwell-Time Monitoring

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 11A. Per-(track, zone) timer | `app/analytics/dwell.py` | `DwellTracker` | Accumulates dwell time for each track in each zone |
| 11B. Current/avg/max stats | `app/analytics/dwell.py` | `DwellTracker.get_dwell`, `get_zone_stats` | Returns current dwell, average, and max for any zone |
| 11C. Grace period | `app/analytics/dwell.py` | `DwellTracker._grace` | Brief absence < grace_period_seconds doesn't reset the timer |
| 11D. Configurable threshold | `config.yaml` | `zones[].dwell_threshold_seconds` | Per-zone dwell threshold in config |

---

## Section 12: Loitering Detection

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 12A. Loitering rule | `app/rules/loitering.py` | `LoiteringRule` | Fires LOITERING when dwell exceeds threshold in a zone |
| 12B. State machine | `app/rules/loitering.py` | `LoiteringRule._alerted_tracks` | Prevents duplicate alerts per track; re-entry window handles brief exits |
| 12C. Re-entry handling | `app/rules/loitering.py` | `LoiteringRule` | `reentry_window_seconds` allows dwell to carry across brief exits |

---

## Section 13: Restricted Zone Intrusion

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 13A. Intrusion rule | `app/rules/intrusion.py` | `IntrusionRule` | Fires ZONE_INTRUSION immediately when any vehicle enters the restricted zone |
| 13B. Restricted zone in config | `config.yaml` | `zones[2]` (no_parking_zone) | Fire Lane / No-Parking Zone defined as restricted |
| 13C. Time-based allowed windows | N/A | Not implemented | Documented as future enhancement |

---

## Section 14: Virtual Line Crossing

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 14A. Line crossing detection | `app/analytics/lines.py` | `LineCrossingMonitor` | Detects when vehicles cross a virtual line using signed distance from line vector |
| 14B. A→B / B→A direction | `app/analytics/lines.py` | `CrossingDirection` | Enum with A_TO_B, B_TO_A values |
| 14C. Separate IN/OUT counters | `app/analytics/lines.py` | `LineCrossingMonitor.get_counts` | Returns (in_count, out_count) per line zone |
| 14D. Hysteresis | `app/analytics/lines.py` | `LineCrossingMonitor._frame_cooldown` | Per-track cooldown to prevent duplicate counts near the line |

---

## Section 15: Wrong-Direction Detection

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 15A. Wrong-direction rule | `app/rules/direction.py` | `WrongDirectionRule` | Fires WRONG_DIRECTION when crossing direction doesn't match expected |
| 15B. Expected direction config | `config.yaml` | `zones[0].expected_direction`, `rules[4].expected_direction` | Expected direction (A_to_B) configurable per zone/rule |

---

## Section 16: Occupancy Monitoring

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 16A. Real-time occupancy | `app/analytics/occupancy.py` | `OccupancyMonitor` | Tracks current vehicle count per polygon zone |
| 16B. Max capacity threshold | `config.yaml` | `zones[1].max_capacity` | Per-zone capacity limit (e.g., 20 for parking_area) |
| 16C. Over-capacity detection | `app/analytics/occupancy.py` | `OccupancyMonitor.check_over_capacity` | Returns (is_over, state_changed) tuple |
| 16D. Time series data | `app/analytics/occupancy.py` | `OccupancyStats.time_series_json` | Returns occupancy over time for charting |
| 16E. Occupancy rule | `app/rules/occupancy.py` | `OccupancyRule` | Fires OVER_CAPACITY when count exceeds threshold |

---

## Section 17: Rule Engine

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 17A. Decoupled rule engine | `app/rules/` package | `BaseRule`, all rule subclasses | Rules evaluated separately from video loop via `RuleContext` |
| 17B. BaseRule interface | `app/rules/base_rule.py` | `BaseRule` | Abstract base with `evaluate(context) → List[RuleViolation]` |
| 17C. RuleContext | `app/rules/base_rule.py` | `RuleContext` | Immutable context passed to rules with all tracked state |
| 17D. Rule factory | `app/rules/__init__.py` | `build_rules` | Creates rule instances from `RuleConfig` list |
| 17E. Enable/disable rules | `config.yaml` | `rules[].enabled` | Per-rule `enabled` flag honored by `BaseRule.evaluate` |

---

## Section 18: Event Lifecycle

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 18A. Event states | `app/events/event_manager.py` | `EventStatus` | DETECTED → ACTIVE → ACKNOWLEDGED → RESOLVED |
| 18B. Event dataclass | `app/events/event_manager.py` | `Event` | Full event with id, type, severity, zone, track, timestamps, evidence, metadata |
| 18C. State transitions | `app/events/event_manager.py` | `EventManager.process_violations` | DETECTED → ACTIVE promotion after min_active_duration_seconds |
| 18D. Auto-resolve | `app/events/event_manager.py` | `EventManager.check_auto_resolve` | Events auto-resolve after cooldown with no new violations |
| 18E. Manual acknowledge | `app/events/event_manager.py` | `EventManager.acknowledge_event` | API-callable acknowledgment |
| 18F. Manual resolve | `app/events/event_manager.py` | `EventManager.resolve_event` | API-callable resolution |

---

## Section 19: Event Debouncing

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 19A. Per-key debouncing | `app/events/event_manager.py` | `Event.debounce_key` | Key = `rule_id:track_id:zone_id`; same key merges into one event |
| 19B. 1800 violations → 1 event | `tests/test_event_manager.py` | `test_sixty_seconds_sixty_fps_one_event` | Verified: 60s × 30fps = 1 event |
| 19C. Cooldown period | `app/events/event_manager.py` | `EventManager._default_cooldown` | Configurable cooldown prevents immediate re-creation |

---

## Section 20: Evidence Capture

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 20A. Full frame saved | `app/events/evidence.py` | `EvidenceSaver._save_jpg` | Full annotated frame saved as JPEG |
| 20B. Cropped vehicle image | `app/events/evidence.py` | `EvidenceSaver._extract_crop` | Cropped bbox with 20px padding |
| 20C. Event metadata JSON | `app/events/evidence.py` | `EvidenceSaver._save_metadata` | Event fields serialized to JSON |
| 20D. Timestamp included | `app/events/evidence.py` | `EvidenceSaver._save_metadata` | `saved_at` timestamp in metadata |
| 20E. Frame buffer | `app/events/evidence.py` | `FrameBuffer` | Rolling buffer of last N frames for pre-event clip capture |

---

## Section 21: Alert Engine

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 21A. Pluggable channels | `app/events/alerts.py` | `AlertChannel` (ABC), `AlertEngine` | Base class with `.send(event)` and channel registration |
| 21B. In-app WebSocket | `app/events/alerts.py` | `InAppWebSocketChannel` | Broadcasts alerts to connected dashboard clients |
| 21C. Simulated webhook | `app/events/alerts.py` | `SimulatedWebhookChannel` | Logs what would be POSTed to webhook URL |
| 21D. Simulated email | `app/events/alerts.py` | `SimulatedEmailChannel` | Logs what would be emailed via SMTP |
| 21E. Channel config | `config.yaml` | `alerts.channels` | Three channels configured with type and enabled flag |

---

## Section 22: Event Database

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 22A. SQLite via aiosqlite | `app/database/repository.py` | `init_db`, `create_async_engine` | Async SQLAlchemy with aiosqlite backend |
| 22B. Event schema | `app/database/models.py` | `EventModel` | All event fields: id, type, severity, status, zone, track, timestamps, evidence, metadata |
| 22C. CRUD operations | `app/database/repository.py` | `create_event`, `update_event`, `get_event`, `query_events` | Full event CRUD with async error handling |
| 22D. Analytics aggregation | `app/database/repository.py` | `get_analytics_summary` | By-type, by-severity, by-zone counts; hourly histogram |
| 22E. Fallback on failure | `app/database/repository.py` | `init_db` (fallback block) | In-memory SQLite if file DB fails |

---

## Section 23: Live Dashboard

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 23A. Live video feed | `static/index.html`, `static/js/dashboard.js` | `handleFrameMessage` | WebSocket streams base64 JPEG frames |
| 23B. Metric cards | `static/index.html` | `#metric-fps`, `#metric-tracks`, `#metric-events` | FPS, active tracks, active events, 24h total |
| 23C. Zone occupancy display | `static/js/dashboard.js` | `updateZoneOccupancy` | Real-time zone counts with progress bars |
| 23D. Line crossing counts | `static/js/dashboard.js` | `updateLineCounts` | IN/OUT counters per line zone |
| 23E. Alert list | `static/js/dashboard.js` | `addAlert`, `renderAlerts` | Real-time alert feed with severity badges |
| 23F. Charts | `static/js/charts.js` | Multiple chart functions | Events over time, by type (donut), by severity (bar), occupancy trend, FPS sparkline |
| 23G. Performance metrics | `static/index.html`, `static/js/dashboard.js` | `updatePerfMeters` | Detection, tracking, rules, total ms, CPU, memory |

---

## Section 24: Event History

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 24A. Event history page | `static/history.html` | Full page | Separate page with event table |
| 24B. Filtering | `app/api/events.py` | `list_events` | Filter by type, severity, zone, status, time range |
| 24C. Pagination | `app/api/events.py` | `list_events` (page, page_size) | Configurable page size (1–200) |
| 24D. CSV export | `app/api/events.py` | `export_events_csv` | Streams CSV with all filtered events |
| 24E. Acknowledge/Resolve API | `app/api/events.py` | `acknowledge_event`, `resolve_event` | PATCH endpoints for manual event management |

---

## Section 25: Configuration

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 25A. YAML configuration | `config.yaml`, `app/config.py` | `load_config` | Loads `config.yaml` and maps to dataclass hierarchy |
| 25B. Environment overrides | `app/config.py` | `_apply_env_overrides` | Environment variables override YAML values |
| 25C. Structured dataclasses | `app/config.py` | `AppConfig`, `ModelConfig`, `TrackerConfig`, etc. | Type-safe configuration with defaults |

---

## Section 26: Performance Monitoring

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 26A. Per-stage timing | `app/analytics/performance.py` | `PerformanceMonitor` | Measures detection, tracking, analytics, rules, total per frame |
| 26B. FPS calculation | `app/analytics/performance.py` | `PerformanceMonitor.fps` | Rolling FPS over recent frames |
| 26C. CPU/memory monitoring | `app/analytics/performance.py` | `PerformanceMonitor._system_stats` | Uses `psutil` for CPU% and memory MB |
| 26D. Bottleneck identification | `app/analytics/performance.py` | `PerformanceMonitor.get_stats` | Reports which stage is the bottleneck |
| 26E. API endpoint | `app/api/analytics.py` | `get_performance` | Returns all performance metrics via REST |

---

## Section 27: Evaluation Dataset

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 27A. 20 test scenarios | `evaluation/scenarios.json` | JSON array | 20 documented scenarios across all categories |
| 27B. Ground truth format | `evaluation/ground_truth.json` | JSON array | Event-level ground truth with type, zone, time range, track_id |

---

## Section 28: Event-Level Metrics

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 28A. TP/FP/FN counts | `evaluation/evaluate_events.py` | `match_events` | Matches detected events to ground truth with time tolerance |
| 28B. Precision/Recall/F1 | `evaluation/evaluate_events.py` | `run_evaluation` | Standard P/R/F1 computation |
| 28C. Detection delay | `evaluation/evaluate_events.py` | `match_events` (delays list) | Measures time between GT start and detection |
| 28D. Duplicate event rate | `evaluation/evaluate_events.py` | `count_duplicates` | Counts events of same type+zone within a time window |
| 28E. Average FPS | `evaluation/evaluate_events.py` | `EvaluationResult.avg_fps` | Average processing FPS |
| 28F. Report output | `evaluation/evaluate_events.py` | `EvaluationResult.print_report` | Formatted console report |

---

## Section 29: Five Experiments

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 29A. Dwell-time threshold | `evaluation/experiments.py` | `experiment_dwell_threshold` | Compares [60, 120, 300, 600]s; reports delay & false alerts |
| 29B. Tracking quality | `evaluation/experiments.py` | `experiment_tracking_quality` | Compares tracker configs; reports event accuracy |
| 29C. Confidence threshold | `evaluation/experiments.py` | `experiment_confidence_threshold` | Tests [0.25, 0.40, 0.60, 0.80]; reports missed vs false |
| 29D. Event debouncing | `evaluation/experiments.py` | `experiment_debouncing` | Compares with/without; reports duplicate rate |
| 29E. Input resolution | `evaluation/experiments.py` | `experiment_resolution` | Compares [320, 480, 640, 960]; reports FPS & quality |

---

## Section 31: Architecture Documentation

| Requirement | File(s) | Purpose |
|---|---|---|
| 31A. Architecture overview | `docs/ARCHITECTURE.md` | Component diagram, data flow, design decisions |
| 31B. Technical decisions | `docs/architecture_decisions.md` | Detailed rationale for tracker, debouncing, FP analysis, etc. |

---

## Section 32: Rule Specifications

| Requirement | File(s) | Purpose |
|---|---|---|
| 32A. Rule documentation | `docs/RULE_SPECIFICATIONS.md` | Per-rule: name, purpose, trigger, threshold, severity, limitations |

---

## Section 33: System State Documentation

| Requirement | File(s) | Purpose |
|---|---|---|
| 33A. State documentation | `docs/SYSTEM_STATE.md` | Per-track state fields, lifecycle, grace periods, tracking loss handling |

---

## Section 34: Error Handling

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| 34A. Invalid video file | `app/vision/video_source.py` | `FileVideoSource.open` | Raises `VideoSourceError` with descriptive message |
| 34B. Video ending unexpectedly | `app/vision/video_source.py` | `_handle_read_failure` | Returns False to end processing loop gracefully |
| 34C. Camera disconnection | `app/vision/video_source.py` | `WebcamVideoSource._handle_read_failure` | Retry with exponential backoff |
| 34D. Missing model file | `app/vision/detector.py` | `VehicleDetector.__init__` | Raises `ModelLoadError` |
| 34E. Failed model loading | `app/vision/detector.py` | `VehicleDetector.__init__` | Raises `ModelLoadError` with error details |
| 34F. Tracker failure | `app/vision/tracker.py` | `VehicleTracker.update` | Catches exceptions, resets tracker state |
| 34G. Database failure | `app/database/repository.py` | `init_db` | Falls back to in-memory SQLite |
| 34H. Evidence storage failure | `app/events/evidence.py` | `EvidenceSaver.save` | Catches OSError, returns None |
| 34I. Invalid zone config | `app/config.py` | `ZoneConfig.validate` | Returns False with logged warning |
| 34J. Invalid rule config | `app/config.py` | `RuleConfig.validate` | Returns False with logged warning |

---

## Section 36: Automated Tests

| Requirement | File(s) | Test Function | Purpose |
|---|---|---|---|
| 36.1 Debouncing | `tests/test_event_manager.py` | `test_sixty_seconds_sixty_fps_one_event` | 1800 violations → 1 event |
| 36.2 Rule evaluation (all 5 rules) | `tests/test_rules.py` | `TestIntrusionRule`, `TestLoiteringRule`, `TestParkingViolationRule`, `TestOccupancyRule`, `TestWrongDirectionRule` | Each rule fires correctly on violation |
| 36.3 Point inside zone | `tests/test_zones_lines.py` | `TestPointInPolygon.test_point_inside_zone` | Point inside polygon returns True |
| 36.4 Point outside zone | `tests/test_zones_lines.py` | `TestPointInPolygon.test_point_outside_zone` | Point outside polygon returns False |
| 36.5 Zone entry | `tests/test_zones_lines.py` | `TestZoneEntryExit.test_zone_entry_event_fires` | Vehicle entering zone generates ENTRY event |
| 36.6 Zone exit | `tests/test_zones_lines.py` | `TestZoneEntryExit.test_zone_exit_event_fires` | Vehicle leaving zone generates EXIT event |
| 36.7 Line crossing | `tests/test_zones_lines.py` | `TestLineCrossing.test_line_crossing_detected` | Vehicle crossing line generates crossing event |
| 36.8 Direction detection | `tests/test_zones_lines.py` | `TestLineCrossing.test_wrong_direction_detected` | Wrong direction flagged correctly |
| 36.9 Dwell time tracking | `tests/test_dwell_occupancy.py` | `TestDwellTracker` | Dwell increments, grace period, multi-vehicle |
| 36.10 Occupancy monitoring | `tests/test_dwell_occupancy.py` | `TestOccupancyMonitor` | Count, capacity, utilization, time series |
| 36.11 Event lifecycle | `tests/test_event_manager.py` | `TestEventLifecycle`, `TestEventStateTransitions` | DETECTED → ACTIVE → ACKNOWLEDGED → RESOLVED |
| 36.12 Invalid config handling | `tests/test_zones_lines.py` | `TestInvalidConfiguration` | Invalid zone/rule configs fail validation gracefully |

---

## Dashboard Test Runner

| Requirement | File(s) | Class/Function | Purpose |
|---|---|---|---|
| Run Tests button | `static/index.html` | `#run-tests-btn` | Dashboard button to trigger test suite |
| Test runner API | `app/api/test_runner.py` | `run_tests` | POST /api/tests/run — runs pytest + experiments, writes CSV |
| CSV download | `app/api/test_runner.py` | `download_latest_csv` | GET /api/tests/latest-csv — downloads results CSV |
| JS handler | `static/js/dashboard.js` | `initTestRunner` | Calls API, renders results inline |
