# Vehicle Event Detection & Alerting Platform

> Intelligent vehicle monitoring: detection, tracking, zone analysis, rule engine, and event management.

---

## Features

- 🚗 **Real-time vehicle detection** (YOLOv8, COCO classes: car, truck, bus, motorcycle)
- 🔄 **Multi-object tracking** with persistent IDs (ByteTrack)
- 🗺️ **Configurable zones** (polygon-based, line-crossing)
- ⏱️ **Dwell-time monitoring** and **loitering detection**
- 🚫 **Parking violation** and **restricted zone intrusion** alerts
- ↕️ **Line crossing** IN/OUT counting with wrong-direction detection
- 👥 **Occupancy monitoring** with over-capacity alerts
- 🧠 **Configurable rule engine** (enable/disable rules per zone)
- 🗄️ **SQLite event database** with full lifecycle management
- 📸 **Evidence capture** (full frame + crop + metadata JSON)
- 📊 **Live dashboard** + **event history** (plain HTML/CSS/JS)
- 🔔 **Alert channels**: in-app, log, webhook, email, Telegram (simulated if unconfigured)

---

## Requirements

| Requirement | Version |
|---|---|
| **Python** | 3.11+ |
| **pip** | pip3.11 |
| **OS** | macOS / Linux / Windows |

---

## Quick Start

### 1. Install dependencies

```bash
pip3.11 install -r requirements.txt
```

> The YOLO model (`yolov8n.pt`) auto-downloads on first run (~6 MB).

### 2. Set up environment

```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env

# Edit .env if needed — defaults work out of the box
```

### 3. Run the server

```bash
# Option A — Python module (recommended)
python3.11 -m app.main

# Option B — Uvicorn with auto-reload (development)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the dashboard

| Page | URL |
|---|---|
| 🖥️ Live Dashboard | http://localhost:8000 |
| 📋 Event History | http://localhost:8000/history.html |
| 📖 API Docs | http://localhost:8000/docs |

---

## Video Sources

### Upload a video file
1. Dashboard → **📁 File** tab → Select video → **Upload & Start**

### Use webcam
1. Dashboard → **📷 Webcam** tab → Set index → **Start Webcam**

### RTSP stream *(bonus)*
1. Dashboard → **📡 RTSP** tab → Enter `rtsp://user:pass@ip:port/stream` → **Connect**

---

## Configuration

All settings live in `config.yaml`. Key sections:

### Model
```yaml
model:
  path: yolov8n.pt      # or yolov8s.pt for better accuracy
  confidence: 0.40
  classes: [car, truck, bus, motorcycle]
  device: cpu           # use "cuda" for GPU
```

### Zones
```yaml
zones:
  - id: parking_area
    name: Parking Area
    type: polygon
    polygon: [[100,150],[500,150],[500,450],[100,450]]
    max_capacity: 20
    dwell_threshold_seconds: 300
```

### Rules
```yaml
rules:
  - id: parking_violation
    name: No-Parking Violation
    event_type: PARKING_VIOLATION
    zone: no_parking_zone
    severity: CRITICAL
    threshold_seconds: 30
    enabled: true
```

### Zone coordinate tuning
Polygon coordinates are `[x, y]` pixel positions relative to your **video resolution** (default zones assume 1280×720).

To get coordinates:
1. Open your video in VLC → **Tools → Media Information → Video tab** for resolution
2. Pause at a frame and note pixel positions using the mouse cursor
3. Or use the [Roboflow Polygon Zone Tool](https://roboflow.github.io/polygonzone/)

---

## Running Tests

### Command Line
```bash
python3.11 -m pytest tests/ -v
```

### Dashboard (Run Tests Button)
Click **"▶ Run Tests"** in the dashboard to execute all 66 unit tests + 5 comparison experiments.
Results are displayed inline and saved to `evaluation/results/` as CSV.

### Test Coverage (66 tests)
- `test_rules.py` — Rule engine (intrusion, loitering, parking violation, occupancy, wrong direction)
- `test_event_manager.py` — Debouncing (1800 violations → 1 event), lifecycle, state transitions, cooldown, capacity
- `test_dwell_occupancy.py` — Dwell timing, grace periods, occupancy calculations, time series
- `test_zones_lines.py` — Point-in-polygon, zone entry/exit, line crossing, direction detection, invalid config

### 5 Experiments
```bash
python3.11 evaluation/experiments.py          # quick mode
python3.11 evaluation/experiments.py --full   # full mode
```
1. Dwell-time threshold comparison
2. Tracking quality comparison
3. Confidence threshold comparison
4. Event debouncing comparison
5. Input resolution comparison

---

## Evidence Files

Saved to `./evidence/` by default:
- `<event_id>_full.jpg` — annotated full frame
- `<event_id>_crop.jpg` — cropped vehicle
- `<event_id>_meta.json` — event metadata

Change path in `.env`: `EVIDENCE_DIR=./evidence`

---

## Optional Alert Channels

Set in `.env` (leave blank to simulate/log):

| Channel | Variables |
|---|---|
| Webhook | `WEBHOOK_URL`, `WEBHOOK_SECRET` |
| Email | `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO` |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |

---

## Known Limitations

1. **Zone coordinates must be tuned per video resolution** — defaults are for 1280×720
2. **ByteTrack re-ID after long occlusion is approximate** — IDs may change after >1s loss
3. **RTSP support** is basic — no jitter buffer, reconnect has 3s delay
4. **GPU inference** requires CUDA drivers — CPU is default (slower)
5. **Evidence clips** need 3s of frame buffer before an event — first-second events lack "before" clip

See `docs/architecture_decisions.md` for detailed technical notes and false-positive analysis.

---

## Project Structure

```
vehicle-event-platform/
├── app/
│   ├── main.py              # FastAPI app + entry point
│   ├── config.py            # Config loading (YAML + .env)
│   ├── pipeline.py          # Main processing pipeline
│   ├── vision/              # Detection + tracking + video source
│   ├── analytics/           # Zones, lines, dwell, occupancy, performance
│   ├── rules/               # Rule engine (base + all rule types)
│   ├── events/              # Event lifecycle, evidence, alerts
│   ├── database/            # SQLite models + async repository
│   └── api/                 # FastAPI routers (video, events, analytics, websocket)
├── static/
│   ├── index.html           # Live dashboard
│   ├── history.html         # Event history page
│   ├── css/styles.css       # Shared styles
│   └── js/                  # api.js, charts.js, dashboard.js
├── tests/
│   ├── test_rules.py        # Rule engine tests
│   ├── test_event_manager.py # Event debouncing + lifecycle tests
│   ├── test_dwell_occupancy.py # Dwell + occupancy tests
│   ├── test_zones_lines.py  # Zone/line/validation tests
│   └── conftest.py          # Shared fixtures
├── evaluation/
│   ├── evaluate_events.py   # Event-level evaluation metrics
│   ├── experiments.py       # 5 comparison experiments
│   ├── scenarios.json       # 20 test scenarios
│   ├── ground_truth.json    # Ground-truth events
│   └── results/             # CSV experiment results
├── docs/
│   ├── REQUIREMENTS_MAP.md  # Full requirement → code traceability
│   ├── ARCHITECTURE.md      # System architecture + diagrams
│   ├── RULE_SPECIFICATIONS.md # Per-rule detailed specs
│   ├── SYSTEM_STATE.md      # State management + tracking loss
│   └── architecture_decisions.md # Technical decision rationale
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies (pip3.11)
└── .env.example             # Environment variable template
```
