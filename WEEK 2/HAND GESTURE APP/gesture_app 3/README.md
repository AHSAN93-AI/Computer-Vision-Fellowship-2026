# GestureVision — Detection Application

A dark-themed Flask + YOLOv8 web app for hand gesture / object detection,
with three input modes, two switchable trained models, live statistics, and
exportable results.

![status](https://img.shields.io/badge/status-ready-7fd6c0)

---

## ✨ Features (Assignment 5)

- **Image Upload** — upload a photo, run detection, see boxes + labels drawn
  on the image, and download the annotated result.
- **Video Upload** — upload a video, the server runs detection frame-by-frame
  and returns a fully annotated `.mp4` you can preview and download.
- **Webcam Detection** — real-time detection on your live camera feed,
  drawn on a canvas overlay (same as the original GestureVision app).
- **Confidence Threshold Slider** — one shared slider (0.10–0.95) that
  applies to whichever mode you're using.
- **Detection Statistics panel** — always shows, for the last run:
  - Number of objects detected
  - Processing time (inference ms for image/webcam, total ms for video)
  - Average / min / max confidence, plus a per-detection confidence chip list
- **Export Results** — "Download Annotated Image" / "Download Annotated
  Video" buttons save the server-drawn, boxed-and-labeled output file.
- **Two switchable models** — toggle between:
  - **Augmented** (`model/best.pt`) — trained on the **815-image augmented**
    dataset
  - **No-Augmentation** (`model/bestnoaug.pt`) — trained on the **350-image
    raw** dataset (no augmentation)

  The active model applies to all three modes (webcam / image / video) and
  switching is instant — both checkpoints are loaded into memory at startup.

## 🧠 About the models

Both checkpoints are YOLOv8 detectors trained on the same hand-gesture class
set; only the training data differs:

| Model | File | Training images | Notes |
|---|---|---|---|
| Augmented | `model/best.pt` | 815 | Trained with data augmentation |
| No-Augmentation | `model/bestnoaug.pt` | 350 | Trained on raw images only |

Class names are read directly from whichever checkpoint is active
(`model.names`) — never hardcoded — so the UI's gesture library and labels
always match the loaded model.

## 📁 Project structure

```
gesture_app/
├── app.py                  # Flask backend: webcam / image / video endpoints
├── requirements.txt
├── README.md
├── model/
│   ├── best.pt               # Augmented — 815 images
│   └── bestnoaug.pt          # No-Augmentation — 350 images
├── templates/
│   └── index.html            # Tabbed UI: Webcam / Image / Video
└── static/
    ├── css/
    │   └── style.css         # Black & grey HUD theme
    ├── js/
    │   └── app.js             # Tabs, model switch, webcam loop, uploads
    └── exports/               # Annotated images/videos land here for download
```

## 🚀 Getting started

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First-time note: `ultralytics` will pull in `torch`/`torchvision` if not
> already installed. CPU-only machines work fine — inference is just a bit
> slower, especially for video.

### 3. Run the app

```bash
python app.py
```

The server starts at **http://127.0.0.1:5000**.

### ⚠️ Camera access requires a "secure context"

Browsers only allow webcam access on `localhost`/`127.0.0.1` or over
**HTTPS**. If you're opening the app from another device on your network,
run `USE_HTTPS=1 python app.py` (requires `pip install pyopenssl`) and visit
`https://<your-computer's-LAN-IP>:5000`, accepting the self-signed cert
warning. This only affects the **Webcam** tab — Image and Video upload work
over plain HTTP from any device.

### 4. Use it

1. Pick a **model** at the top of the panel (Augmented vs No-Augmentation).
2. Pick a **mode** tab: Webcam, Image Upload, or Video Upload.
3. Adjust the **confidence threshold** slider as needed.
4. Run detection:
   - **Webcam:** click Start Camera.
   - **Image:** choose a file, click Run Detection, then Download Annotated
     Image if you want to save it.
   - **Video:** choose a file, click Run Detection (this can take a little
     while depending on video length), then Download Annotated Video.
5. Check the **Detection Statistics** panel on the right for object count,
   processing time, and confidence scores from the last run.

## ⚙️ Configuration

In `app.py`:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATHS` | `best.pt` / `bestnoaug.pt` | Paths for the two switchable checkpoints |
| `DEFAULT_MODEL_KEY` | `augmented` | Which model is active on page load |
| `DEFAULT_CONF_THRESHOLD` | `0.5` | Initial slider position |
| `IMG_SIZE` | `640` | Inference resolution |
| `VIDEO_FRAME_SKIP` | `2` | Run detection every Nth video frame (speed vs. smoothness) |

In `static/js/app.js`:

| Variable | Default | Purpose |
|---|---|---|
| `CAPTURE_INTERVAL_MS` | `250` | How often a webcam frame is sent to the server |

## 🔌 API

### `POST /predict` — webcam single-frame JSON detection
Body: `{ "image": "data:image/jpeg;base64,...", "threshold": 0.5, "model": "augmented" }`
Returns detections (normalized box coords), `inference_ms`, and a `stats` block.

### `POST /predict_image` — multipart image upload
Form fields: `image` (file), `threshold`, `model`.
Returns detections, `stats`, a base64 `annotated_image`, and an `export_url`
to download the same annotated JPEG from disk.

### `POST /predict_video` — multipart video upload
Form fields: `video` (file), `threshold`, `model`.
Returns `stats` aggregated across the whole clip, `frame_count`,
`frames_analyzed`, `processing_ms`, and an `export_url` to the annotated
`.mp4`.

### `GET /exports/<filename>`
Serves annotated images/videos written to `static/exports/`.

### `GET /health`
Liveness check — `{"status": "ok", "models_loaded": [...]}`.

## 🛠 Tech stack

- **Backend:** Flask, Ultralytics YOLOv8, OpenCV
- **Frontend:** HTML, CSS (no framework), vanilla JavaScript
- **Models:** two YOLOv8 checkpoints (augmented vs. non-augmented training data)

## 📌 Notes & tips

- Inference runs on whatever device PyTorch picks (GPU if available, CPU
  otherwise). Video processing is the most CPU-intensive mode — raise
  `VIDEO_FRAME_SKIP` if it feels slow.
- Exported videos are written with the `mp4v` codec via OpenCV. Most
  browsers can download and play them fine; if in-browser preview looks
  odd on your system, the downloaded file will still open correctly in a
  standard media player (VLC, etc.).
- To compare the two models on the same input, run detection once, switch
  the model pill, and run again — the stats panel updates each time.

---

Built with Flask + Ultralytics YOLOv8.
