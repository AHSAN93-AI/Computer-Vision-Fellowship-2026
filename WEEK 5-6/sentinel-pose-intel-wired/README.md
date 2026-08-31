# Sentinel Pose Intel

Human pose and activity intelligence platform — real-time camera feed, per-person
activity tracking (walking, sitting, standing, bending, hand-raise, waving,
squat reps), fall/posture alerts, alert history, and analytics.

**Stack:** Python / Flask backend + a plain HTML / CSS / vanilla JS dashboard
(no React, no Tailwind, no build step).

The project has two parts:

- **Backend** (`app/`): a Flask service that runs a YOLO-pose + ByteTrack
  pipeline over a video source, classifies per-person activity with
  rule-based state machines, raises fall/ergonomic alerts, persists events to
  SQLite, and serves the dashboard (`app/dashboard/routes.py`). Fall
  detection is an **ensemble of two independent signals**:
    1. `app/activities/fall.py` — rule-based, 5-factor pose-geometry analysis
       over time (torso angle, descent speed, aspect ratio, head/hip
       position, post-fall stillness).
    2. `app/vision/fall_classifier.py` — a trained YOLOv8-cls CNN
       (`models/fall_classifier.pt`, classes `Fall`/`NotFall`) run on each
       tracked person's cropped bounding box, with a rolling per-track vote
       window so a single noisy frame can't trigger a false alarm.
  A fall alert fires if *either* signal confirms; the reported confidence is
  the max of the two.
- **Frontend** (`static/`, `templates/index.html`): a single-page dashboard
  in plain HTML/CSS/JS. It polls `GET /api/live` every ~1s for status,
  tracked entities, and alerts, and displays the live annotated feed via an
  MJPEG stream at `/video_feed` (a plain `<img>` tag — no WebSocket, no
  client framework).

See `docs/architecture.md` for the full data-flow diagram and layer
breakdown, and `docs/activity_rules.md` for the per-activity detection
thresholds.

## Run locally

**Prerequisites:** Python 3.10+

1. Install dependencies:
   `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and adjust `VIDEO_SOURCE`, thresholds, etc.
   as needed.
3. Run the server:
   `python -m app.main`
   This starts Flask on `0.0.0.0:8000` and serves the dashboard at `/`.
4. Open `http://localhost:8000` in a browser, enter a video source (`0` for
   a webcam, a file path, or an `rtsp://` URL) in the Live Grid tab, and
   click **Start**.
5. Run the test suite:
   `pytest`

## Project structure

```
app/
├── main.py                   # Flask entry point
├── pipeline.py                # Orchestrates the end-to-end video → alert pipeline
├── config.py                  # Env-driven settings (see .env.example)
├── vision/
│   ├── video_source.py        # Webcam / file / RTSP video source
│   ├── person_tracker.py      # YOLO-pose + ByteTrack tracking
│   ├── pose_estimator.py
│   ├── skeleton_renderer.py
│   └── fall_classifier.py     # CNN Fall/NotFall classifier + per-track voting
├── pose/                      # Keypoints, joint angles, normalisation, sequence buffer
├── activities/                # Rule-based recognisers (walking, fall, squats, ...) + state machine
├── events/                    # Activity manager, alert engine, evidence capture
├── dashboard/routes.py        # Flask blueprint: REST routes + MJPEG video stream
└── database/                  # SQLite persistence

models/
└── fall_classifier.pt         # Trained YOLOv8-cls Fall/NotFall checkpoint

evaluation/                   # Synthetic pose generators + evaluation harness
tests/                        # Pytest unit tests for pose/activity logic

templates/index.html          # Dashboard page (Live Grid / Analytics / Alert History / Config tabs)
static/
├── css/style.css
└── js/app.js                  # Polling, controls, alert sound, tab switching
```

## API summary

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard page |
| `/video_feed` | GET | MJPEG live annotated stream |
| `/api/health` | GET | Liveness probe |
| `/api/status` | GET | Pipeline status snapshot |
| `/api/live` | GET | Combined status + cameras + tracked entities + alerts (polled) |
| `/api/events` | GET | Paginated activity events |
| `/api/events/export` | GET | CSV export |
| `/api/events/stats` | GET | Activity type counts / avg durations |
| `/api/alerts` | GET | Alert list |
| `/api/alerts/<id>/acknowledge` | POST | Acknowledge an alert |
| `/api/config` | GET / PUT | Read / update runtime configuration |
| `/api/pipeline/start` | POST | Start the pipeline (optional `?source=`) |
| `/api/pipeline/stop` | POST | Stop the pipeline |
