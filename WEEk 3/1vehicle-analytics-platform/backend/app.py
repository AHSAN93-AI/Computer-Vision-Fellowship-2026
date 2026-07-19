"""
app.py
Flask backend for the Intelligent Vehicle Video Analytics Platform.

Run with:
    python app.py
Then open:
    http://localhost:5000
"""

import os
import atexit
import base64

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory, send_file

from engine import VehicleAnalyticsEngine

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
RECORDING_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
engine = VehicleAnalyticsEngine()
atexit.register(engine.shutdown)


# ------------------------------------------------------------- frontend --
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    if os.path.exists(os.path.join(FRONTEND_DIR, filename)):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": "not found"}), 404


# --------------------------------------------------------------- stream --
@app.route("/video_feed")
def video_feed():
    return Response(
        engine.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# -------------------------------------------------------------- control --
@app.route("/api/start", methods=["POST"])
def start():
    data = request.get_json(force=True, silent=True) or {}
    source_type = data.get("source", "webcam")
    path = data.get("path")
    engine.set_source(source_type, path)
    engine.start()
    return jsonify({"status": "started", "source": source_type})


@app.route("/api/stop", methods=["POST"])
def stop():
    engine.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/remove_source", methods=["POST"])
def remove_source():
    engine.remove_source()
    return jsonify({"status": "source removed"})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    f = request.files["video"]
    save_path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(save_path)
    return jsonify({"status": "uploaded", "path": save_path, "filename": f.filename})


# ------------------------------------------------------------ still image
@app.route("/api/detect_image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    f = request.files["image"]
    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "could not decode image"}), 400

    annotated, stats = engine.detect_image(image)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({"error": "encoding failed"}), 500

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return jsonify({"image": f"data:image/jpeg;base64,{b64}", "stats": stats})


# ---------------------------------------------------------------- line ---
@app.route("/api/set_line", methods=["POST"])
def set_line():
    data = request.get_json(force=True)
    engine.set_line(int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))
    return jsonify({"status": "line set"})


@app.route("/api/clear_line", methods=["POST"])
def clear_line():
    engine.clear_line()
    return jsonify({"status": "line cleared"})


# ----------------------------------------------------------------- roi ---
@app.route("/api/set_roi", methods=["POST"])
def set_roi():
    data = request.get_json(force=True)
    points = data.get("points", [])
    if len(points) < 3:
        return jsonify({"error": "roi needs at least 3 points"}), 400
    engine.set_roi(points)
    return jsonify({"status": "roi set"})


@app.route("/api/clear_roi", methods=["POST"])
def clear_roi():
    engine.clear_roi()
    return jsonify({"status": "roi cleared"})


# ---------------------------------------------------------- threshold ---
@app.route("/api/set_threshold", methods=["POST"])
def set_threshold():
    data = request.get_json(force=True)
    engine.set_confidence_threshold(data.get("value", 0.3))
    return jsonify({"status": "threshold set"})


# -------------------------------------------------------------- heatmap -
@app.route("/api/heatmap", methods=["POST"])
def toggle_heatmap():
    data = request.get_json(force=True)
    engine.set_heatmap(bool(data.get("enabled", False)))
    return jsonify({"status": "heatmap updated"})


# ------------------------------------------------------------ recording -
@app.route("/api/record/start", methods=["POST"])
def record_start():
    engine.toggle_recording(True, frame_size=(640, 480))
    return jsonify({"status": "recording started"})


@app.route("/api/record/stop", methods=["POST"])
def record_stop():
    engine.toggle_recording(False)
    return jsonify({"status": "recording stopped"})


@app.route("/api/recordings")
def list_recordings():
    files = sorted(os.listdir(RECORDING_DIR), reverse=True)
    return jsonify({"recordings": files})


@app.route("/api/recordings/<path:filename>")
def download_recording(filename):
    return send_file(os.path.join(RECORDING_DIR, filename), as_attachment=True)


# ------------------------------------------------------------- analytics-
@app.route("/api/stats")
def stats():
    return jsonify(engine.get_stats())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
