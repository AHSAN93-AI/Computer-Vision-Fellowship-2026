# VEHTRACK — Intelligent Vehicle Video Analytics Platform

A real-time vehicle detection, tracking, and analytics platform built for the
Visibility Bots Innovation Lab — AI Summer Fellowship 2026, Week 3 Challenge.

Detects vehicles with **YOLOv8**, tracks them across frames with **ByteTrack**
(via the Supervision library), and turns that into live analytics: counting,
a virtual counting line, ROI zone occupancy, movement trajectories, and a
live dashboard — all through a browser front end talking to a Flask backend.

---

## 1. Architecture

```
Webcam / Video File
        │
        ▼
  Frame Capture (OpenCV)
        │
        ▼
  Object Detection (YOLOv8n — car / motorcycle / bus / truck)
        │
        ▼
  Object Tracking (ByteTrack — persistent IDs across frames)
        │
        ▼
  Event Processing (line crossing, ROI occupancy, trajectories)
        │
        ▼
  Analytics Engine (counts, avg confidence, FPS, processing time)
        │
        ├──► Dashboard (live stats, polled every 1s)
        ├──► Annotated MJPEG stream (browser video panel)
        └──► Recorded MP4 report (annotated video saved to disk)
```

**Backend:** Flask, OpenCV, Ultralytics YOLOv8, Supervision
**Frontend:** Vanilla HTML / CSS / JS (canvas overlay for drawing line & ROI)

## 2. Project Structure

```
vehicle-analytics-platform/
├── backend/
│   ├── app.py            # Flask routes (REST API + MJPEG stream)
│   ├── engine.py          # Detection + tracking + analytics engine
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── uploads/               # uploaded video files land here
├── recordings/            # saved annotated recordings land here
└── README.md
```

## 3. Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run auto-downloads `yolov8n.pt` (~6 MB) — no manual dataset or
training required. The pretrained model already recognizes the four vehicle
classes used here (car, motorcycle, bus, truck) from COCO.

## 4. Run

```bash
cd backend
python app.py
```

Open **http://localhost:5000** in your browser.

## 5. Using the Dashboard

| Action | How |
|---|---|
| Start webcam | Click **▶ Start Webcam** |
| Analyze a video file | Click **▶ Load Video File** and pick a traffic video |
| Stop the feed | Click **■ Stop** |
| Draw counting line | Click **＋ Draw Counting Line**, then click two points on the video |
| Draw ROI zone | Click **＋ Draw ROI Zone**, click 3+ points, press **Enter** to finish |
| Clear overlays | Click **✕ Clear Overlays** |
| Adjust sensitivity | Drag the **Confidence Threshold** slider |
| Record annotated output | Click **● Record** / **■ Stop Recording** — file appears under Recordings |

The right-hand dashboard updates every second with: active objects, total
unique objects, entered (IN) / exited (OUT) counts, objects inside the ROI,
average detection confidence, FPS, and per-frame processing time.

## 6. Feature Checklist

- [x] Feature 1 — Real-time detection (YOLOv8, pretrained)
- [x] Feature 2 — Object tracking with persistent IDs (ByteTrack)
- [x] Feature 3 — Object counter (total / active / entered / exited)
- [x] Feature 4 — Virtual counting line (user-drawn, line-crossing IN/OUT)
- [x] Feature 5 — ROI zone monitoring (user-drawn polygon, occupancy count)
- [x] Feature 6 — Movement analytics (trajectory trails + direction)
- [x] Feature 7 — Analytics dashboard (current objects, total, avg confidence, FPS, processing time)
- [x] Feature 8 — Video recording (annotated MP4 saved to `recordings/`)
- [x] Bonus — Heatmap (toggle on/off, accumulates vehicle density over time)
- [x] Extra — Still-image detection (upload a photo and get annotated results + per-class counts)
- [x] Extra — Per-class ID numbering series (see below)
- [x] Extra — Remove Video button (clear the current source to load another one)

## 7. Crash Fix — "Assertion fctx->async_lock failed"

If you hit this error while switching sources quickly (remove → upload →
start in fast succession), it was caused by a race condition: multiple
Flask request threads were touching the same OpenCV video capture object
at once — one thread reading a frame while another released/reassigned it,
which crashes FFmpeg's decoder.

This is fixed with a single dedicated background thread that is now the
**only** thing allowed to open, read, or release the camera/video file
(`_capture_loop` in `engine.py`). Every Flask route (`/api/start`,
`/api/remove_source`, `/video_feed`, etc.) now goes through a lock instead
of touching the capture object directly, so this class of crash can no
longer happen no matter how fast you switch sources. Verified with an
automated stress test simulating rapid remove/upload/start cycles under
concurrent viewers with zero errors.

## 8. Object Lock System (fixes flickering boxes / inflated counts)

Previously, a vehicle's bounding box could briefly flicker frame-to-frame
(a normal side-effect of per-frame confidence fluctuation), which caused the
tracker to drop the ID and assign a brand-new one on the very next frame —
inflating "Total Objects" far beyond the real vehicle count.

This is now fixed with a proper lock system, built on ByteTrack's own
confirmation + occlusion-handling mechanics (`engine.py`):

- **Detection always runs at a low base confidence (0.10)** internally, so
  the tracker has enough signal to keep matching an object even when its
  per-frame confidence briefly dips — instead of losing it and starting over.
- **A track only "locks" (gets a stable ID, gets counted, gets drawn) after
  being matched for 5 consecutive frames** (`minimum_consecutive_frames`).
  One-off false-positive blips never lock, so they never inflate the count.
- **Once locked, a track survives up to 50 frames of being fully
  undetected** (`lost_track_buffer`) before it's dropped — e.g. a car
  briefly hidden behind another vehicle keeps its original ID instead of
  getting a new one when it re-emerges.
- The **Confidence Threshold** slider now controls how confident a *new*
  object must be to lock on (`track_activation_threshold`) — raise it if
  you're seeing false locks, lower it if real vehicles aren't locking fast
  enough.
- Locked vehicles are labeled with a 🔒 icon in the video feed.

## 9. Object Counter — Entered / Exited Logic

`Entered` and `Exited` are appearance-based, not line-crossing based:

- **Entered** increments the moment a vehicle is first detected anywhere in
  the camera view (a new tracker ID appears).
- **Exited** increments once a previously-tracked vehicle hasn't been seen
  for a short grace period (`MISSING_FRAMES_THRESHOLD = 20` frames in
  `engine.py`) — meaning it left the frame or tracking was lost. This grace
  period avoids false "exits" from a single missed detection frame.
- If a vehicle reappears before being marked exited, it isn't double-counted.

This is separate from **Line Crossed IN / Line Crossed OUT**, which only
count vehicles that physically cross your drawn counting line (Feature 4) —
useful if you want directional counting across a specific road/lane rather
than whole-frame entry/exit.

## 10. Per-Class ID Numbering Series

Each vehicle class gets its own independent ID series instead of one shared
counter, so IDs are unambiguous at a glance. Format: `[class digit][2-digit sequence]`,
sequence incrementing by 2:

| Class | Digit | ID sequence |
|---|---|---|
| car | `0` | `002`, `004`, `006`, ... |
| motorcycle | `1` | `102`, `104`, `106`, ... |
| truck | `2` | `202`, `204`, `206`, ... |
| bus | `3` | `302`, `304`, `306`, ... |

The first car detected is always `002`, the second car `004`, and so on —
independent of how many motorcycles, trucks, or buses are also on screen.
This is implemented in `_get_custom_id()` in `engine.py` and applied both to
live video tracking and to still-image detection (image mode restarts each
class's sequence fresh per uploaded image).

## 11. Heatmap (Bonus)

Toggle with the **▦ Heatmap** button. Accumulates vehicle positions into a
decaying density map, rendered as a JET colormap overlay directly on the
video feed — hot zones show where vehicles spend the most time in frame.

## 12. Still-Image Detection

Click **🖼 Upload Image** to run detection on a single photo instead of
video. Returns an annotated image plus per-class counts and average
confidence in a result panel below the video feed. No tracking IDs persist
across images — each image gets its own fresh ID sequence.

## 13. Switching Sources

Click **🗑 Remove Video** to fully stop and release the current source
(webcam or file) before loading a new one — this clears all counters,
IDs, and overlays so the next video starts clean.

## 14. Notes on Tracking Algorithm Choice

ByteTrack is used by default because it associates *all* detection boxes
(including low-confidence ones) rather than discarding them, which reduces
ID switches under partial occlusion — common in traffic footage where
vehicles pass behind each other or signposts. Supervision also ships
BoT-SORT; swapping `sv.ByteTrack()` for `sv.BoTSORT()` in `engine.py` is a
one-line change if you want to run the comparison experiment described in
the fellowship brief (Assignment 2).

## 15. Known Limitations / Next Steps

- Runs on CPU by default; GPU (CUDA) will be used automatically by
  Ultralytics if available, improving FPS.
- The "exited" grace period (20 frames) means very brief occlusions won't
  falsely trigger an exit, but a vehicle that stays fully hidden longer than
  that will be re-assigned a new ID (and a new entered/exited pair) if it
  reappears — this is a known tradeoff of ID-based tracking under occlusion,
  worth mentioning in your Builder Journal.
- The webcam source assumes the camera is on the same machine running the
  Flask server (local deployment), consistent with how this project is
  meant to be demoed.
