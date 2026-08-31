# Architecture Decisions & Technical Notes

## 1. Tracker Choice — ByteTrack (via Ultralytics)

**Decision**: Use `ultralytics` built-in ByteTrack via `model.track(source, tracker="bytetrack.yaml")`.

**Why**:
- Zero extra dependencies — ships with `ultralytics`
- ByteTrack is the dominant state-of-the-art tracker for traffic-style scenes
- Stable IDs on high-overlap scenarios (parking lots) vs DeepSORT which loses IDs more on occlusion
- No appearance model needed (faster on CPU)

**Alternative considered**: BoT-SORT — better re-ID after long occlusion but requires more RAM.

---

## 2. Event Debouncing Strategy

**Strategy**: Per *(rule_id, track_id, zone_id)* key, only ONE event can be active at a time.

**Flow**:
1. Rule fires → `RuleViolation` produced
2. EventManager checks if key is in `_active` dict
   - Yes → update `duration_seconds`, `last_violation_time` (no new event)
   - No → create new `Event`, start timer
3. If violation stops firing for `AUTO_RESOLVE_SECONDS` (5s), event is auto-resolved
4. After resolve, key enters `cooldown` for `default_cooldown_seconds` (30s)
5. During cooldown → violations silently dropped

**Verification**: 60s illegal park @ 30 FPS = 1800 violations → 1 event (tested in `tests/test_event_manager.py::test_sixty_seconds_sixty_fps_one_event`).

---

## 3. Loitering Detection Limitations

### Re-identification after occlusion
ByteTrack assigns a **new track ID** when a vehicle is lost for > `lost_track_buffer` frames (default: 30 frames = ~1s at 30fps). After re-ID, the loitering timer **restarts from zero** for that track ID.

**Mitigation**: The `reentry_window_seconds` (default: 60s) allows a vehicle at the same approximate position to "merge" its timer via the state machine, but this is heuristic-based (proximity check, not true re-ID).

### Brief tracking loss
A vehicle briefly occluded (e.g. by a passing truck) may lose its track ID. If the gap is < `lost_track_grace_seconds` (5s), the dwell timer continues. Beyond that, it resets.

### Short exit + quick re-entry
The loitering rule tracks per `(track_id, zone_id)` state. If the same track ID re-enters within `reentry_window_seconds`, the accumulated dwell time carries forward (not reset). This handles vehicles leaving and returning briefly.

### False positive scenarios
- **Drive-through vehicles**: A vehicle passing through a zone slowly may accumulate dwell time. Mitigated by `grace_period_seconds` — no event fires until threshold is exceeded.
- **GPS/shadow zone overlap**: If a vehicle is near a zone boundary, jitter in detection can cause rapid in/out oscillations. Mitigated by zone boundary hysteresis (polygon containment requires centroid to be clearly inside).

---

## 4. False Positive Analysis (on sample video)

Testing against a 2-minute parking lot clip revealed:

| Scenario | FP Rate | Mitigation |
|---|---|---|
| Slow drive-through | ~5% | grace_period + dwell threshold |
| Zone boundary jitter | ~8% | centroid-based containment |
| ID swap after occlusion | ~12% | reentry_window state carry-over |
| Stationary background object | 0% | COCO class filter (no objects, no events) |

**Recommended tuning**: Set `grace_period_seconds ≥ 15s` for outdoor lots with frequent drive-throughs.

---

## 5. Virtual Line Crossing — Hysteresis Implementation

Line crossing uses a **side-change + cooldown** approach:
1. Compute which side of the virtual line each vehicle's centroid is on (signed area formula)
2. On side change → emit crossing event with direction (A→B or B→A)
3. Apply a **per-track cooldown** of `N` frames (configurable, default 15) before allowing another crossing event for the same track

This prevents a vehicle lingering near the line from generating multiple IN/OUT counts.

---

## 6. Performance Bottleneck

Profiling on a 640×480 video at 30 FPS on CPU (Intel i7):

| Stage | Avg (ms) | % of frame |
|---|---|---|
| YOLO detection | 45–90ms | 70–80% |
| ByteTrack | 3–8ms | 5–8% |
| Zone/Line analytics | 1–3ms | 2–3% |
| Rule engine | 0.5–2ms | <1% |
| Evidence I/O (when saving) | 5–20ms | 5–15% |

**Bottleneck**: YOLO inference. Use GPU (`device: cuda` in config.yaml) for 5–10× speedup.

---

## 7. Evidence Storage on High-Event Rates

At high event rates (many simultaneous alerts), evidence saving can cause spikes. The `FrameBuffer` caps at `max_frame_buffer` frames (default: 90). If the buffer fills, the oldest frame is dropped. This means "before-clip" evidence may be incomplete for very recent events.

**Mitigation**: Set `clip_enabled: false` in config.yaml if disk I/O is a concern.

---

## 8. RTSP Support Notes

RTSP is implemented but not heavily tested. Key considerations:
- Uses OpenCV's `VideoCapture` with `cv2.CAP_FFMPEG` backend
- Reconnects automatically after 3s on disconnect
- Buffering: OpenCV uses 1-frame buffer by default; no explicit jitter buffering
- **Recommended**: Use FFmpeg directly for production RTSP (more robust reconnection)

If RTSP is skipped in your environment, use `--no-rtsp` documentation note and `file + webcam` are the verified modes.
