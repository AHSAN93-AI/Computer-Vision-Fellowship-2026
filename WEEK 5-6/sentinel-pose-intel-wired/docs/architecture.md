# Sentinel Pose Intel — Architecture

## Overview

Sentinel Pose Intel is a real-time human pose and activity intelligence platform.
It processes video feeds through a multi-stage pipeline: detection, tracking,
pose estimation, activity recognition, and alerting — then streams results to
a React dashboard via WebSocket.

## Data Flow

```mermaid
graph TD
    VS["VideoSource<br/>(video_source.py)"] -->|BGR frame| PT["PersonTracker<br/>(person_tracker.py)"]
    PT -->|PersonKeypoints[]| AM["ActivityManager<br/>(activity_manager.py)"]
    AM -->|per-person| SM["8 x ActivityStateMachine<br/>(state_machine.py)"]
    AM -->|per-person| SC["SquatCounter<br/>(squats.py)"]
    AM -->|per-person| EM["ErgonomicMonitor<br/>(ergonomic.py)"]
    AM -->|per-person| BUF["PoseSequenceBuffer<br/>(sequence.py)"]
    BUF -->|angles, velocity| ANG["Angles + Normalization<br/>(angles.py, normalization.py)"]
    AM -->|PersonActivityState| PO["Pipeline Orchestrator<br/>(pipeline.py)"]
    PO -->|fall/ergo alerts| AE["AlertEngine<br/>(alerts.py)"]
    PO -->|activity events| DB[("SQLite<br/>(db.py)")]
    PO -->|evidence frames| EV["EvidenceCapture<br/>(evidence.py)"]
    PO -->|status + frame| WS["WebSocket + REST<br/>(routes.py)"]
    WS -->|JSON updates| FE["React Dashboard<br/>(App.tsx)"]
```

## Layer Architecture

### Layer 1: Vision (`app/vision/`)

| Module | Responsibility |
|--------|---------------|
| `video_source.py` | Abstracts video input (file, webcam, RTSP). Auto-detects source type, handles reconnection. |
| `person_tracker.py` | Wraps YOLO-Pose `.track()` with ByteTrack for persistent person IDs across frames. Maintains `TrackedPerson` state per ID. |
| `pose_estimator.py` | Defines `PoseEstimator` protocol + `YoloPoseEstimator` implementation. Used by the demo script; the pipeline uses `PersonTracker` directly. |
| `skeleton_renderer.py` | `draw_skeletons()` overlay utility. Separated from `pose_estimator.py` to avoid coupling the drawing function to the YOLO model class. |

### Layer 2: Pose Analysis (`app/pose/`)

| Module | Responsibility |
|--------|---------------|
| `keypoints.py` | COCO-17 keypoint constants, `Keypoint` / `PersonKeypoints` dataclasses, confidence filtering, colour palette. |
| `angles.py` | Joint angle calculations: elbow, knee, hip, torso angle, torso length, shoulder width, bbox aspect ratio, head-to-hip distance. All return `None` for missing keypoints. |
| `normalization.py` | Hip-centred, torso-length-scaled coordinate normalisation. Makes activity thresholds invariant to camera distance. Also provides `compute_velocity_from_raw()`. |
| `sequence.py` | `PoseSequenceBuffer` — rolling deque of `PoseSnapshot` objects. Computes per-frame features (angles, velocity, normalised coords) and provides temporal analysis (averages, peaks, oscillation detection). |

### Layer 3: Activity Recognition (`app/activities/`)

| Module | Responsibility |
|--------|---------------|
| `base_activity.py` | `ActivityRecogniser` abstract base class and `ActivityCandidate` dataclass. |
| `state_machine.py` | `ActivityStateMachine` — temporal smoothing via Idle -> Candidate -> Active -> Ended lifecycle with configurable confirm/end frame thresholds. |
| `standing.py` | Standing recogniser: upright torso, low velocity, straight knees. |
| `sitting.py` | Sitting recogniser: bent knees < 120 deg, upright torso, low velocity. |
| `walking.py` | Walking recogniser: hip velocity > threshold, upright torso, knee variation. |
| `hand_raise.py` | Hand raise: wrist above shoulder, extended elbow angle. |
| `bending.py` | Bending: torso > 45 deg, low velocity (not falling), straight legs. |
| `waving.py` | Waving: raised hand + lateral wrist oscillation in the buffer. |
| `fall.py` | Multi-factor fall detection: 5 indicators, need >= 3 to fire. |
| `squats.py` | Squat counter with 4-phase state machine (standing/descending/down/ascending). |
| `ergonomic.py` | Prolonged bending/crouching timer with configurable warning thresholds. |

### Layer 4: Events & Alerts (`app/events/`)

| Module | Responsibility |
|--------|---------------|
| `activity_manager.py` | Per-person orchestrator: runs all recognisers, manages state machines, determines winning activity by priority, records timeline events. |
| `alerts.py` | `AlertEngine` — generates fall (CRITICAL), ergonomic (WARNING), and inactivity (INFO) alerts with cooldown and deduplication. |
| `evidence.py` | `EvidenceCapture` — saves JPEG evidence frames with metadata filenames. |

### Layer 5: Dashboard (`app/dashboard/`)

| Module | Responsibility |
|--------|---------------|
| `routes.py` | FastAPI router with REST endpoints (status, events, alerts, config, pipeline control) and WebSocket (`/ws/live`) for real-time frame + status streaming. |
| `schemas.py` | Pydantic response models for dashboard-compatible JSON output. |

### Layer 6: Persistence (`app/database/`)

| Module | Responsibility |
|--------|---------------|
| `db.py` | SQLite with WAL mode. Tables: `activity_events`, `alert_events`. Parameterised queries, CSV export, indexed lookups. |

### Infrastructure

| Module | Responsibility |
|--------|---------------|
| `config.py` | Pydantic Settings with `.env` loading + JSON sidecar hot-reload. All thresholds configurable. |
| `__init__.py` | Package bootstrap: `setup_logging()` with file + console handlers. |
| `main.py` | FastAPI app creation, CORS, route registration, static file mounts. |
| `pipeline.py` | Background-thread pipeline orchestrator tying all layers together. |

## Frontend Architecture

```
src/
├── App.tsx              # Top-level state, layout, tab routing
├── types.ts             # Shared TypeScript interfaces
├── data/initialData.ts  # Mock data (fallback when backend offline)
├── hooks/
│   └── usePipelineSocket.ts  # WebSocket hook with auto-reconnect
├── api/
│   └── client.ts        # REST + WebSocket client (typed fetch wrappers)
└── components/
    ├── Header.tsx        # Top bar with alert count, sound toggle
    ├── Sidebar.tsx       # Left navigation
    ├── CameraDetailModal.tsx  # Fullscreen camera inspector
    ├── LiveGrid/         # Camera tiles, timeline, entities, alerts, config
    └── views/            # Alert history, analytics, config pages
```

**Stack**: React 19 + TypeScript + TailwindCSS 4 + Vite + Recharts + Lucide icons.

## WebSocket Protocol

The `/ws/live` endpoint sends JSON messages at ~10 Hz:

```json
{
  "type": "frame_update",
  "status": { "isRunning": true, "fps": 28.5, "inferenceTimeMs": 34.2, ... },
  "cameras": [{ "id": "CAM_01", "skeletons": [...], ... }],
  "trackedEntities": [{ "id": "#001", "posture": "Walking", ... }],
  "alerts": [{ "id": "...", "severity": "CRITICAL", ... }],
  "frame": "<base64 JPEG>"
}
```

## Configuration System

All thresholds are defined in `app/config.py` as Pydantic `Field` values with defaults.

**Load order** (later wins):
1. Field defaults in code
2. `.env` file values
3. Environment variables
4. JSON sidecar (`data/settings_override.json`) written by dashboard `PUT /api/config`
