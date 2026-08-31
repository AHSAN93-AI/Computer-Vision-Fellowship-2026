# Sentinel Pose Intel — Spec Traceability

Cross-reference of specification sections to implementing code. Based on the sections referenced in the codebase docstrings and config comments.

## Coverage Matrix

| Spec Section | Description | Status | Implementing File(s) |
|:-------------|:------------|:------:|:---------------------|
| §4.1 | System Overview / Architecture | DONE | `app/pipeline.py`, `app/main.py` |
| §4.2 | Video Input & Display | DONE | `app/vision/video_source.py`, `app/vision/skeleton_renderer.py` |
| §4.3 | Keypoint Confidence Filtering | DONE | `app/pose/keypoints.py` (filter_keypoints_by_confidence) |
| §4.4 | Pose Normalisation | DONE | `app/pose/normalization.py` |
| §4.5 | Sequence Buffer | DONE | `app/pose/sequence.py` |
| §4.6 | Joint Angle Calculations | DONE | `app/pose/angles.py` |
| §4.7 | Person Tracking | DONE | `app/vision/person_tracker.py` |
| §4.8 | Activity Classification (Rule-Based) | DONE | `app/activities/standing.py`, `sitting.py`, `walking.py`, `hand_raise.py`, `bending.py`, `waving.py` |
| §4.9 | Activity State Machine | DONE | `app/activities/state_machine.py` |
| §4.10 | Fall Detection (Multi-Factor) | DONE | `app/activities/fall.py` |
| §4.11 | Fall Alert Lifecycle | DONE | `app/events/alerts.py` (check_fall, acknowledge, resolve) |
| §4.12 | Repetition Counting (Squats) | DONE | `app/activities/squats.py` |
| §4.13 | Ergonomic Monitoring | DONE | `app/activities/ergonomic.py` |
| §4.14 | Activity Manager (Orchestration) | DONE | `app/events/activity_manager.py` |
| §4.15 | Database Persistence | DONE | `app/database/db.py` |
| §4.16 | Evidence Capture | DONE | `app/events/evidence.py` |
| §4.17 | Alert Engine | DONE | `app/events/alerts.py` |
| §4.18 | Evaluation Harness | DONE | `evaluation/evaluate.py`, `evaluation/synthetic.py` |
| §4.19 | Unit Testing | DONE | `tests/test_state_machine.py`, `test_angles.py`, `test_normalization.py`, `test_sequence.py`, `test_activities.py` |
| §4.20 | Documentation | DONE | `docs/architecture.md`, `docs/activity_rules.md`, `docs/traceability.md` |
| §4.21 | Dashboard UI | DONE | `src/App.tsx`, `src/hooks/usePipelineSocket.ts`, `src/api/client.ts`, `app/dashboard/routes.py` |

## Gap Analysis

### §4.18 — Evaluation Harness

**Status**: DONE (synthetic mode implemented)

- `evaluation/synthetic.py`: Generates synthetic `PersonKeypoints` for all 8 activity types
- `evaluation/evaluate.py`: Runs activity recognisers against synthetic sequences, reports per-activity precision/recall/F1 and latency metrics (avg, p95, p99)
- **Gap**: Video-mode evaluation (processing real recorded clips with ground-truth annotation JSON sidecars) is scaffolded but not yet implemented. The synthetic mode provides full coverage of the activity rule logic.

### §4.19 — Unit Testing

**Status**: DONE

| Test File | Module Under Test | Coverage Areas |
|-----------|------------------|----------------|
| `test_state_machine.py` | `state_machine.py` | Full lifecycle, hysteresis, confirm/end thresholds, properties, reset |
| `test_angles.py` | `angles.py` | calculate_angle (90/180/0/45/60 deg), all named angle functions, torso_angle sign convention, auxiliary functions |
| `test_normalization.py` | `normalization.py` | Hip-centred origin, torso scaling, missing keypoints, velocity computation |
| `test_sequence.py` | `sequence.py` | Buffer lifecycle, rolling window, temporal analysis, wrist oscillations |
| `test_activities.py` | All recognisers | Detection/non-detection per activity, SquatCounter phases, ErgonomicMonitor timing |

### §4.20 — Documentation

**Status**: DONE

- `docs/architecture.md`: Module responsibilities, Mermaid data-flow diagram, layer descriptions, frontend architecture, WebSocket protocol, configuration system
- `docs/activity_rules.md`: Per-activity rule tables, state machine diagrams, threshold reference, squat hysteresis, ergonomic monitoring, alert system, priority table, complete config reference
- `docs/traceability.md`: This document

### §4.21 — Dashboard UI

**Status**: DONE

The React dashboard is fully implemented with:
- Real-time WebSocket updates via `usePipelineSocket.ts`
- Live camera grid with skeleton overlays
- Alert management (acknowledge/resolve)
- Pipeline control (start/stop/configure)
- Activity timeline and event history
- Analytics charts (Recharts)
- Responsive layout with TailwindCSS

**Gap**: The dashboard currently uses base64 JPEG frames over WebSocket, which works for single-camera prototype use but would need optimisation (e.g., MJPEG stream or HLS) for multi-camera production deployment.

## Summary

All 21 specification sections (§4.1-§4.21) now have implementing code. The evaluation harness (§4.18) has full synthetic coverage; video-mode evaluation is scaffolded for future annotated test clips.
