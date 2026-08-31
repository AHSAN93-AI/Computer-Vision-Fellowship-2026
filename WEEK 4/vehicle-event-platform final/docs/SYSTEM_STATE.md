# System State Documentation

This document describes the per-track state maintained by the platform, the lifecycle of tracked objects, grace periods, and how temporary tracking loss is handled.

---

## Per-Track State Fields

Each tracked vehicle is represented by a `TrackedVehicle` instance in `app/vision/tracker.py`:

| Field | Type | Description | Updated |
|---|---|---|---|
| `track_id` | `int` | Persistent identifier assigned by ByteTrack | On first detection; stable across frames |
| `class_id` | `int` | COCO class index (2=car, 5=bus, 7=truck, etc.) | Every frame |
| `class_name` | `str` | Human-readable class ("car", "truck", "bus", "motorcycle") | Every frame |
| `bbox` | `Tuple[int,int,int,int]` | Bounding box (x1, y1, x2, y2) in pixel coordinates | Every frame |
| `confidence` | `float` | Detection confidence score (0.0–1.0) | Every frame |
| `first_seen` | `float` | Timestamp of first detection | Once (on creation) |
| `last_seen` | `float` | Timestamp of most recent detection | Every frame |
| `frame_id` | `int` | Frame number of most recent detection | Every frame |
| `position_history` | `deque[(float,float)]` | Last N centroid positions (default: 30) | Every frame |
| `centroid` | `property → (float,float)` | Current (cx, cy) = center of bbox | Computed from bbox |
| `trail` | `property → List[(float,float)]` | Full position history as list | Computed from position_history |
| `age_seconds` | `property → float` | Time since first_seen | Computed |

### Derived State

| Method/Property | Returns | Description |
|---|---|---|
| `is_stationary(px_threshold, window)` | `bool` | True if centroid variance over last `window` positions is below `px_threshold` |
| `centroid` | `(float, float)` | Center of current bounding box |
| `trail` | `List[(float, float)]` | Full history of centroid positions |

---

## Zone State (Per Track)

The `ZoneManager` in `app/analytics/zones.py` maintains:

| State | Type | Description |
|---|---|---|
| `_track_zones` | `Dict[int, Set[str]]` | Maps track_id → set of zone_ids the vehicle is currently inside |
| `_previous_track_zones` | `Dict[int, Set[str]]` | Same mapping from the previous frame |

### Zone Transitions

On each `ZoneManager.update()` call:

1. For each tracked vehicle, compute which zones contain its centroid (point-in-polygon)
2. Compare to previous frame's zones
3. **New zone** → emit `ZoneEvent(ENTRY)` 
4. **Left zone** → emit `ZoneEvent(EXIT)`
5. **Still in zone** → no event (state-based, not per-frame)

```
Frame N:   Vehicle #5 in zones {A}
Frame N+1: Vehicle #5 in zones {A, B}
  → ENTRY event for zone B

Frame N+2: Vehicle #5 in zones {B}
  → EXIT event for zone A
```

---

## Dwell State (Per Track × Zone)

The `DwellTracker` in `app/analytics/dwell.py` maintains:

| State | Type | Description |
|---|---|---|
| `_entries` | `Dict[(zone_id, track_id), DwellEntry]` | Active dwell timers |
| `_completed` | `List[CompletedDwell]` | Finished dwell records for statistics |

### DwellEntry Fields

| Field | Type | Description |
|---|---|---|
| `zone_id` | `str` | Zone being monitored |
| `track_id` | `int` | Vehicle being tracked |
| `enter_time` | `float` | Timestamp when vehicle entered zone |
| `last_update` | `float` | Timestamp of most recent update |
| `accumulated` | `float` | Total accumulated dwell in seconds |

### Dwell Lifecycle

```
1. Vehicle enters zone → DwellEntry created, enter_time = now
2. Vehicle still in zone → last_update refreshed, accumulated = now - enter_time
3. Vehicle exits zone → grace period starts
4. During grace period:
   a. Vehicle re-enters → DwellEntry preserved, timer continues
   b. Grace expires → DwellEntry completed, moved to _completed
5. Dwell statistics updated with completed entry
```

---

## Grace Periods

### Dwell Grace Period (`lost_track_grace_seconds`)

- **Default**: 5.0 seconds
- **Purpose**: When a vehicle's track ID temporarily disappears (occlusion, detection miss), the dwell timer is NOT immediately reset
- **Behavior**: If the same track_id reappears in the same zone within `grace_period` seconds, the existing dwell entry continues
- **If exceeded**: Dwell entry is finalized and completed

### Loitering Re-Entry Window (`reentry_window_seconds`)

- **Default**: 60 seconds
- **Purpose**: If a vehicle exits a zone and re-enters within this window, the accumulated dwell time carries forward
- **Behavior**: The `LoiteringRule` checks if the vehicle was recently in the zone; if so, dwell is not reset
- **Use case**: Vehicle that briefly exits to let another pass, then returns

### Parking Grace Period (`threshold_seconds`)

- **Default**: 30 seconds (for no_parking_zone)
- **Purpose**: Vehicles are allowed to transit through a restricted zone briefly (e.g., loading/unloading)
- **Behavior**: PARKING_VIOLATION only fires if dwell > grace AND vehicle is stationary

---

## Tracking Loss Handling

### Scenario 1: Brief Occlusion (< `lost_track_buffer` frames)

```
Frame 100: Vehicle #5 detected, tracked
Frame 101: Vehicle #5 occluded by truck — NOT detected
Frame 102: Vehicle #5 still occluded
...
Frame 125: Vehicle #5 visible again — ByteTrack re-identifies as #5

Result: Same track_id maintained. Dwell timer continues.
```

ByteTrack maintains lost tracks for `lost_track_buffer` frames (default: 30 = ~1 second at 30fps).

### Scenario 2: Extended Occlusion (> `lost_track_buffer` frames)

```
Frame 100: Vehicle #5 detected
Frame 101–140: Occluded (40 frames > 30 frame buffer)
Frame 141: Vehicle re-appears → assigned NEW track_id #12

Result: Track #5 finalized. New track #12 starts fresh.
         Dwell timer for #5 enters grace period, then completes.
         New dwell timer starts for #12 from zero.
```

### Scenario 3: Dwell Grace Period Recovery

```
Frame 100: Track #5 in zone_a, dwell = 45s
Frame 101: Track #5 not detected (brief miss)
Frame 102: Track #5 not detected (still missing)
  → Grace timer starts (5s default)
Frame 103: Track #5 re-detected in zone_a
  → Grace timer cancelled, dwell continues at 45s + elapsed

Frame 200: Track #5 in zone_a, dwell = 45s  
Frame 201–260: Track #5 missing (60 frames = 2s at 30fps)
  → Still within grace (2s < 5s grace)
Frame 261: Track #5 back → dwell continues

Frame 300: Track #5 in zone_a, dwell = 100s
Frame 301–500: Track #5 missing (200 frames ≈ 6.7s)
  → Grace expired (6.7s > 5s)
  → Dwell entry finalized at 100s
```

---

## Event State Machine

Events go through a well-defined lifecycle managed by `EventManager`:

```
                    ┌─────────────┐
                    │  DETECTED   │ ← First violation
                    └──────┬──────┘
                           │ (after min_active_duration)
                    ┌──────▼──────┐
                    │   ACTIVE    │ ← Ongoing violation
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       (manual ack)   (manual resolve)  (auto-resolve)
              │            │            │
       ┌──────▼──────┐     │     ┌──────▼──────┐
       │ ACKNOWLEDGED │     │     │  RESOLVED   │
       └──────┬──────┘     │     └─────────────┘
              │            │
        (manual resolve)   │
              │            │
       ┌──────▼────────────▼┐
       │      RESOLVED      │
       └────────────────────┘
```

### State Descriptions

| State | Description | Can Transition To |
|---|---|---|
| `DETECTED` | New event, first violation received | ACTIVE, ACKNOWLEDGED, RESOLVED |
| `ACTIVE` | Event has been ongoing for > `min_active_duration_seconds` (5s) | ACKNOWLEDGED, RESOLVED |
| `ACKNOWLEDGED` | Operator has seen and acknowledged the event | RESOLVED |
| `RESOLVED` | Event is complete (manually or auto-resolved) | Terminal state |

### Auto-Resolution

An event auto-resolves when:
1. No new violation with the same debounce key arrives for `default_cooldown_seconds` (30s)
2. The `EventManager.check_auto_resolve()` method is called (runs every pipeline tick)

After resolution, the event enters a cooldown period where new violations with the same key are suppressed.

---

## Occupancy State

The `OccupancyMonitor` maintains per-zone statistics:

| Field | Type | Description |
|---|---|---|
| `current_count` | `int` | Number of vehicles currently in zone |
| `max_capacity` | `int` | Configured maximum capacity |
| `max_observed` | `int` | Peak count ever observed |
| `average_occupancy` | `float` | Rolling average count |
| `utilization_pct` | `float` | current_count / max_capacity × 100 |
| `is_over_capacity` | `bool` | True if current_count > max_capacity |
| `time_series` | `deque[(timestamp, count)]` | Historical count values for charting |

---

## Line Crossing State

The `LineCrossingMonitor` maintains per-track state for each line:

| Field | Type | Description |
|---|---|---|
| `_track_sides` | `Dict[int, int]` | Which side of the line each track is on (-1 or +1) |
| `_track_cooldown` | `Dict[int, int]` | Frame-based cooldown counter per track (prevents re-fire) |
| `_in_count` | `int` | Total A→B crossings (IN) |
| `_out_count` | `int` | Total B→A crossings (OUT) |

A crossing is detected when `_track_sides[track_id]` changes sign between frames, and the track is not in cooldown.
