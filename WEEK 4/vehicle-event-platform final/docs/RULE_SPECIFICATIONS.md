# Rule Specifications

This document details every rule in the platform. Each entry covers the rule's name, purpose, implementing class, target zone, trigger conditions, thresholds, severity, start/end conditions, debouncing behavior, evidence output, and known limitations.

---

## 1. Zone Intrusion (`IntrusionRule`)

| Field | Value |
|---|---|
| **Rule ID** | `intrusion` |
| **Name** | Restricted Zone Intrusion |
| **Purpose** | Detect any vehicle entering a restricted/no-parking zone |
| **File** | `app/rules/intrusion.py` |
| **Class** | `IntrusionRule` |
| **Event Type** | `ZONE_INTRUSION` |
| **Severity** | `HIGH` |
| **Target Zone** | `no_parking_zone` (Fire Lane / No-Parking Zone) |

### Trigger Conditions
- Vehicle's centroid is inside the zone polygon (point-in-polygon test)
- Vehicle was NOT in this zone on the previous frame (state-transition trigger)

### Thresholds
- None — fires immediately on entry
- `cooldown_seconds: 60` — after resolution, same key suppressed for 60s

### Start/End Conditions
- **Start**: First frame where vehicle centroid enters the zone polygon
- **End**: Auto-resolves after `default_cooldown_seconds` (30s) with no new violations, or manually resolved via API

### Debouncing
- Key: `intrusion:{track_id}:{zone_id}`
- Continuous presence in zone merges into single event
- Different track IDs → separate events

### Evidence
- Full frame JPEG with vehicle highlighted
- Cropped vehicle image with 20px padding
- Metadata JSON with event details

### Known Limitations
- No time-based allowed windows (e.g., "allowed during business hours") — all times treated equally
- Zone boundary jitter can cause rapid in/out on detections near polygon edges; mitigated by centroid-based containment

---

## 2. Loitering Detection (`LoiteringRule`)

| Field | Value |
|---|---|
| **Rule ID** | `loitering` |
| **Name** | Loitering Detection |
| **Purpose** | Detect vehicles that stay in a zone beyond the allowed dwell time |
| **File** | `app/rules/loitering.py` |
| **Class** | `LoiteringRule` |
| **Event Type** | `LOITERING` |
| **Severity** | `WARNING` |
| **Target Zone** | `parking_area` (Main Parking Area) |

### Trigger Conditions
- Vehicle is inside the target zone
- Accumulated dwell time exceeds `threshold_seconds` (300s)

### Thresholds
- `threshold_seconds: 300` (5 minutes)
- `reentry_window_seconds: 60` — if vehicle re-enters within 60s, dwell carries forward
- `cooldown_seconds: 300` — post-resolve suppression

### Start/End Conditions
- **Start**: Frame where dwell time first exceeds threshold_seconds
- **End**: Auto-resolves when violations stop for cooldown period, or manually resolved

### Debouncing
- Key: `loitering:{track_id}:{zone_id}`
- Internal `_alerted_tracks` set prevents repeated alerts for same track
- Dwell timer is persistent — not reset by brief absences within grace period

### Evidence
- Full frame, cropped vehicle, metadata JSON
- Metadata includes: dwell_seconds, threshold_seconds

### Known Limitations
- Track ID reassignment after occlusion resets the dwell timer (new track = new timer)
- Mitigated by `reentry_window_seconds` which merges timers for vehicles at similar positions
- Drive-through vehicles may accumulate dwell if moving slowly (mitigated by threshold being high)

---

## 3. Parking Violation (`ParkingViolationRule`)

| Field | Value |
|---|---|
| **Rule ID** | `parking_violation` |
| **Name** | Parking Violation (No-Parking Zone) |
| **Purpose** | Detect vehicles parked illegally in a restricted zone after a grace period |
| **File** | `app/rules/parking_violation.py` |
| **Class** | `ParkingViolationRule` |
| **Event Type** | `PARKING_VIOLATION` |
| **Severity** | `CRITICAL` |
| **Target Zone** | `no_parking_zone` (Fire Lane / No-Parking Zone) |

### Trigger Conditions (Two-Stage)
1. Vehicle is inside the restricted zone (centroid in polygon)
2. Vehicle is **stationary** (centroid variance < `stationary_px_threshold` over last `stationary_frames` frames)
3. Dwell time exceeds `threshold_seconds` (grace period)

### Thresholds
- `threshold_seconds: 30` (grace period — 30s)
- `stationary_px_threshold: 10` (pixels — centroid must move < 10px)
- `stationary_frames: 15` (window — checks last 15 position samples)

### Start/End Conditions
- **Start**: First frame where all three conditions are met (in zone + stationary + dwell > grace)
- **End**: Auto-resolves when violations stop, or manually resolved

### Debouncing
- Key: `parking_violation:{track_id}:{zone_id}`
- Stationary check distinguishes from a mere pass-through

### Evidence
- Full frame, cropped vehicle, metadata JSON
- Metadata includes: dwell_seconds, grace_period_seconds, stationary flag

### Known Limitations
- A vehicle that stops momentarily (e.g., at a stop sign) and then moves may briefly trigger if dwell accumulates
- GPS/shadow overlap at zone boundaries can cause detection flicker

---

## 4. Occupancy / Over-Capacity (`OccupancyRule`)

| Field | Value |
|---|---|
| **Rule ID** | `over_capacity` |
| **Name** | Parking Over Capacity |
| **Purpose** | Detect when the number of vehicles in a zone exceeds maximum capacity |
| **File** | `app/rules/occupancy.py` |
| **Class** | `OccupancyRule` |
| **Event Type** | `OVER_CAPACITY` |
| **Severity** | `CRITICAL` |
| **Target Zone** | `parking_area` (Main Parking Area) |

### Trigger Conditions
- `OccupancyMonitor.check_over_capacity()` returns `(is_over=True, state_changed=True)`
- Fires only on the *transition* to over-capacity (not every frame while over)

### Thresholds
- `threshold: 20` (max vehicle count)
- `hysteresis: 2` (must drop to threshold - hysteresis before re-triggering)

### Start/End Conditions
- **Start**: Frame where count transitions from ≤ threshold to > threshold
- **End**: Auto-resolves when capacity drops below threshold - hysteresis

### Debouncing
- Key: `over_capacity:{track_id}:{zone_id}` (track_id may be 0 for zone-level events)
- State-change detection prevents repeated firing while over capacity

### Evidence
- Full frame showing all vehicles in zone
- Metadata includes: current_count, max_capacity

### Known Limitations
- Count depends on tracking accuracy — missed detections can undercount
- Vehicles partially in zone (centroid on boundary) may flicker between counted/not-counted

---

## 5. Wrong Direction (`WrongDirectionRule`)

| Field | Value |
|---|---|
| **Rule ID** | `wrong_direction` |
| **Name** | Wrong-Way Detection (Exit Only Lane) |
| **Purpose** | Detect vehicles crossing a virtual line in the wrong direction |
| **File** | `app/rules/direction.py` |
| **Class** | `WrongDirectionRule` |
| **Event Type** | `WRONG_DIRECTION` |
| **Severity** | `CRITICAL` |
| **Target Zone** | `entrance_lane` (Entrance Gate Lane) |

### Trigger Conditions
- A `LineCrossingEvent` is generated for the target zone
- The event's `direction` does NOT match `expected_direction`
- The event's `is_wrong_direction` flag is True

### Thresholds
- `expected_direction: A_to_B` — correct direction for the entrance lane
- `cooldown_seconds: 30` — post-resolve suppression

### Start/End Conditions
- **Start**: Frame where the wrong-direction crossing is detected
- **End**: Auto-resolves after cooldown period

### Debouncing
- Key: `wrong_direction:{track_id}:{zone_id}`
- Per-track cooldown in `LineCrossingMonitor` prevents duplicate crossings from hysteresis

### Evidence
- Full frame, cropped vehicle, metadata JSON
- Metadata includes: detected_direction, expected_direction

### Known Limitations
- Vehicles lingering near the line and oscillating may trigger if they cross during oscillation
- Mitigated by frame-cooldown hysteresis in `LineCrossingMonitor`
- Track ID swap near the line can cause a missed detection (vehicle crosses but ID changes mid-cross)

---

## Rule Configuration (config.yaml)

All rules are configured in the `rules` section of `config.yaml`:

```yaml
rules:
  - id: parking_violation
    name: Parking Violation (No-Parking Zone)
    event_type: PARKING_VIOLATION
    severity: CRITICAL
    enabled: true
    zone: no_parking_zone
    condition: stationary_in_zone
    threshold_seconds: 30
    stationary_px_threshold: 10
    stationary_frames: 15

  - id: intrusion
    name: Restricted Zone Intrusion
    event_type: ZONE_INTRUSION
    severity: HIGH
    enabled: true
    zone: no_parking_zone
    condition: zone_entry
    cooldown_seconds: 60

  - id: loitering
    name: Loitering Detection
    event_type: LOITERING
    severity: WARNING
    enabled: true
    zone: parking_area
    condition: dwell_exceeded
    threshold_seconds: 300
    reentry_window_seconds: 60
    cooldown_seconds: 300

  - id: over_capacity
    name: Parking Over Capacity
    event_type: OVER_CAPACITY
    severity: CRITICAL
    enabled: true
    zone: parking_area
    condition: occupancy_exceeded
    threshold: 20
    hysteresis: 2

  - id: wrong_direction
    name: Wrong-Way Detection
    event_type: WRONG_DIRECTION
    severity: CRITICAL
    enabled: true
    zone: entrance_lane
    condition: direction_violation
    expected_direction: A_to_B
    cooldown_seconds: 30
```

Each rule can be independently enabled/disabled via the `enabled` field.
