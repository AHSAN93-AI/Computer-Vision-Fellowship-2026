"""
GestureVision — Flask backend  (Assignment 5: Detection Application)
----------------------------------------------------------------------
Three detection modes, all sharing the same YOLOv8 inference pipeline:

  1. Webcam        — client captures frames (getUserMedia), posts base64
                      JPEGs to /predict, gets JSON detections back.
  2. Image upload   — client posts a file to /predict_image, server
                      draws boxes and returns an annotated image + stats.
  3. Video upload   — client posts a file to /predict_video, server
                      processes every frame, writes an annotated .mp4,
                      and returns aggregate stats.

Two trained checkpoints are available and swappable per-request via a
"model" field ("augmented" or "baseline"):

  model/best.pt       — trained on the AUGMENTED dataset (815 images)
  model/bestnoaug.pt  — trained on the RAW dataset, no augmentation (350 images)

Both are loaded once at startup and cached in memory so switching
between them in the UI is instant.
"""

import base64
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "augmented": "model/best.pt",       # trained on augmented dataset (815 images)
    "baseline": "model/bestnoaug.pt",   # trained on raw dataset, no augmentation (350 images)
}
MODEL_LABELS = {
    "augmented": "Augmented · 815 imgs",
    "baseline": "No-Augmentation · 350 imgs",
}
DEFAULT_MODEL_KEY = "augmented"

DEFAULT_CONF_THRESHOLD = 0.5
IMG_SIZE = 640

OUTPUT_DIR = os.path.join("static", "exports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# For video we run inference every Nth frame and re-draw the last known
# boxes on the frames in between. This keeps processing time reasonable
# on CPU while still writing a smooth annotated video.
VIDEO_FRAME_SKIP = 2

ACCENT_BGR = (192, 214, 127)  # OpenCV-space BGR version of the UI's teal accent
INK_BGR = (13, 11, 10)

# ---------------------------------------------------------------------------
# App + models
# ---------------------------------------------------------------------------
app = Flask(__name__)

_model_cache = {}


def get_model(key):
    """Return (resolved_key, model) for a requested model key, loading + caching on first use."""
    key = key if key in MODEL_PATHS else DEFAULT_MODEL_KEY
    if key not in _model_cache:
        print(f"Loading model '{key}' from {MODEL_PATHS[key]} ...")
        _model_cache[key] = YOLO(MODEL_PATHS[key])
        print(f"Model '{key}' loaded. Classes:", get_class_names(_model_cache[key]))
    return key, _model_cache[key]


def get_class_names(model):
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


# Warm both models at startup so switching in the UI never has to wait on a
# cold load.
for _k in MODEL_PATHS:
    get_model(_k)

CLASS_NAMES = get_class_names(_model_cache[DEFAULT_MODEL_KEY])


# ---------------------------------------------------------------------------
# Shared inference helpers
# ---------------------------------------------------------------------------
def decode_base64_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def parse_threshold(value):
    try:
        t = float(value)
    except (TypeError, ValueError):
        t = DEFAULT_CONF_THRESHOLD
    return max(0.01, min(t, 0.99))


def run_inference(model, frame, conf_threshold):
    """Run YOLO on a single BGR frame. Returns (detections, inference_ms).

    Each detection has both pixel-space ("box") and normalized 0-1 ("box_norm")
    coordinates, since the webcam/image/video consumers each want a different one.
    """
    t0 = time.time()
    results = model.predict(source=frame, imgsz=IMG_SIZE, conf=conf_threshold, verbose=False)
    inference_ms = round((time.time() - t0) * 1000, 1)

    detections = []
    h, w = frame.shape[:2]
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            names = model.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                label = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else names[cls_id]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "box_norm": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
                })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, inference_ms


def draw_detections(frame, detections):
    """Draw bounding boxes + label chips onto a BGR frame in place. Returns the frame."""
    for d in detections:
        b = d["box"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BGR, 2)

        label = f"{d['label']} {d['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, ty), (x1 + tw + 8, ty + th + 8), ACCENT_BGR, -1)
        cv2.putText(frame, label, (x1 + 4, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK_BGR, 1, cv2.LINE_AA)
    return frame


def summarize(detections):
    confs = [d["confidence"] for d in detections]
    class_counts = {}
    for d in detections:
        class_counts[d["label"]] = class_counts.get(d["label"], 0) + 1
    return {
        "num_objects": len(detections),
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0,
        "min_confidence": round(min(confs), 4) if confs else 0,
        "max_confidence": round(max(confs), 4) if confs else 0,
        "class_counts": class_counts,
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        classes=CLASS_NAMES,
        default_conf=DEFAULT_CONF_THRESHOLD,
        model_options=MODEL_LABELS,
        default_model=DEFAULT_MODEL_KEY,
    )


# ---------------------------------------------------------------------------
# 1) Webcam — single frame in, JSON detections out
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "No image supplied"}), 400

    conf_threshold = parse_threshold(payload.get("threshold", DEFAULT_CONF_THRESHOLD))
    model_key, model = get_model(payload.get("model", DEFAULT_MODEL_KEY))

    frame = decode_base64_image(image_data)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    detections, inference_ms = run_inference(model, frame, conf_threshold)
    stats = summarize(detections)
    h, w = frame.shape[:2]

    return jsonify({
        "detections": [
            {"label": d["label"], "confidence": d["confidence"], "box": d["box_norm"]}
            for d in detections
        ],
        "inference_ms": inference_ms,
        "threshold": conf_threshold,
        "model": model_key,
        "frame_size": {"width": w, "height": h},
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# 2) Image upload — file in, annotated image + JSON out
# ---------------------------------------------------------------------------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file supplied"}), 400

    file = request.files["image"]
    conf_threshold = parse_threshold(request.form.get("threshold", DEFAULT_CONF_THRESHOLD))
    model_key, model = get_model(request.form.get("model", DEFAULT_MODEL_KEY))

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    detections, inference_ms = run_inference(model, frame, conf_threshold)
    annotated = draw_detections(frame.copy(), detections)
    stats = summarize(detections)

    export_name = f"image_{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, export_name), annotated)

    ok, buf = cv2.imencode(".jpg", annotated)
    annotated_b64 = ("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")) if ok else None

    h, w = frame.shape[:2]
    return jsonify({
        "detections": [
            {"label": d["label"], "confidence": d["confidence"], "box": d["box_norm"]}
            for d in detections
        ],
        "inference_ms": inference_ms,
        "threshold": conf_threshold,
        "model": model_key,
        "frame_size": {"width": w, "height": h},
        "stats": stats,
        "annotated_image": annotated_b64,
        "export_url": f"/exports/{export_name}",
    })


# ---------------------------------------------------------------------------
# 3) Video upload — file in, annotated .mp4 + JSON stats out
# ---------------------------------------------------------------------------
@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file supplied"}), 400

    file = request.files["video"]
    conf_threshold = parse_threshold(request.form.get("threshold", DEFAULT_CONF_THRESHOLD))
    model_key, model = get_model(request.form.get("model", DEFAULT_MODEL_KEY))

    tmp_in = os.path.join(OUTPUT_DIR, f"in_{uuid.uuid4().hex[:10]}_{file.filename}")
    file.save(tmp_in)

    cap = cv2.VideoCapture(tmp_in)
    if not cap.isOpened():
        os.remove(tmp_in)
        return jsonify({"error": "Could not open video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    export_name = f"video_{uuid.uuid4().hex[:10]}.mp4"
    export_path = os.path.join(OUTPUT_DIR, export_name)
    writer = cv2.VideoWriter(export_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_confidences = []
    class_counts = {}
    frames_processed = 0
    frame_idx = 0
    last_detections = []
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % VIDEO_FRAME_SKIP == 0:
            detections, _ = run_inference(model, frame, conf_threshold)
            last_detections = detections
            frames_processed += 1
            for d in detections:
                all_confidences.append(d["confidence"])
                class_counts[d["label"]] = class_counts.get(d["label"], 0) + 1

        writer.write(draw_detections(frame, last_detections))
        frame_idx += 1

    cap.release()
    writer.release()
    os.remove(tmp_in)

    processing_ms = round((time.time() - t_start) * 1000, 1)

    return jsonify({
        "model": model_key,
        "threshold": conf_threshold,
        "processing_ms": processing_ms,
        "frame_count": frame_idx,
        "frames_analyzed": frames_processed,
        "fps": round(fps, 2),
        "stats": {
            "num_objects": len(all_confidences),
            "avg_confidence": round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0,
            "min_confidence": round(min(all_confidences), 4) if all_confidences else 0,
            "max_confidence": round(max(all_confidences), 4) if all_confidences else 0,
            "class_counts": class_counts,
        },
        "export_url": f"/exports/{export_name}",
    })


@app.route("/exports/<path:filename>")
def exports(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": list(_model_cache.keys())})


if __name__ == "__main__":
    use_https = os.environ.get("USE_HTTPS", "0") == "1"
    ssl_context = "adhoc" if use_https else None

    if use_https:
        print("Starting with HTTPS (self-signed cert) -> https://<your-ip>:5000")
    else:
        print("Starting with HTTP. Camera access only works on http://localhost:5000")
        print("or http://127.0.0.1:5000 -- for LAN/HTTPS access run with USE_HTTPS=1")

    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=ssl_context)
