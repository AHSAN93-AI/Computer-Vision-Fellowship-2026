"""
app.dashboard.routes — Flask Routes (REST + MJPEG video stream)

Provides:
  • ``GET  /api/health``        — liveness probe
  • ``GET  /api/status``        — pipeline status snapshot
  • ``GET  /api/live``          — combined status/cameras/entities/alerts (polled by the frontend)
  • ``GET  /api/events``        — paginated activity events
  • ``GET  /api/events/export`` — CSV download
  • ``GET  /api/events/stats``  — analytics stats
  • ``GET  /api/alerts``        — alert list
  • ``POST /api/alerts/<id>/acknowledge``
  • ``GET  /api/config``        — read config
  • ``PUT  /api/config``        — update config
  • ``POST /api/pipeline/start``
  • ``POST /api/pipeline/stop``
  • ``POST /api/upload``        — upload a video file to use as a source
  • ``GET  /api/uploads``       — list previously uploaded videos
  • ``GET  /video_feed``        — MJPEG live stream (multipart/x-mixed-replace)
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from flask import Blueprint, Response, jsonify, request
from werkzeug.utils import secure_filename

from app.config import get_settings, update_settings
from app.pipeline import Pipeline, PipelineStatus

logger = logging.getLogger(__name__)

# ── Video upload ─────────────────────────────────────────
_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

bp = Blueprint("dashboard", __name__)

# ── Singleton pipeline instance ─────────────────────────
_pipeline = Pipeline()

_latest_frame = None


def _on_frame(frame, status: PipelineStatus) -> None:
    """Pipeline callback — stores latest annotated frame for MJPEG streaming."""
    global _latest_frame
    _latest_frame = frame


def _on_alert(alert) -> None:
    logger.info("New alert via callback: %s", alert.alert_id)


_pipeline.set_callbacks(on_frame=_on_frame, on_alert=_on_alert)


# ── Helper: build dashboard-compatible data from pipeline status ──

def _build_camera_feed(status: PipelineStatus) -> List[dict]:
    if not status.is_running:
        return []

    skeletons = []
    for pid, pd in status.persons.items():
        bbox = pd.get("bbox")
        if bbox:
            skeletons.append({
                "id": f"#{pid:03d}",
                "posture": pd.get("activity_display", "Unknown"),
                "confidence": pd.get("confidence", 0.0),
                "box": {
                    "x": float(bbox[0]) / 10,
                    "y": float(bbox[1]) / 10,
                    "width": float(bbox[2] - bbox[0]) / 10,
                    "height": float(bbox[3] - bbox[1]) / 10,
                },
            })

    is_alert = any(a.get("severity") == "CRITICAL" for a in status.active_alerts)
    cam_status = "CRITICAL" if is_alert else ("NOMINAL" if status.active_people > 0 else "IDLE")
    status_text = "FALL_DETECTED: CRITICAL" if is_alert else f"TRACKING: {cam_status}"

    return [{
        "id": "CAM_01",
        "name": f"CAM_01_{status.source_info[:20]}",
        "location": status.source_info,
        "fps": round(status.fps, 1),
        "mbps": round(status.fps * 0.4, 1),
        "status": cam_status,
        "statusText": status_text,
        "personCount": status.active_people,
        "skeletons": skeletons,
    }]


def _build_tracked_entities(status: PipelineStatus) -> List[dict]:
    entities = []
    for pid, pd in status.persons.items():
        dur = pd.get("activity_duration", 0)
        mins = int(dur // 60)
        secs = int(dur % 60)
        entities.append({
            "id": f"#{pid:03d}",
            "posture": pd.get("activity_display", "Unknown"),
            "duration": f"{mins:02d}:{secs:02d}s",
            "confidence": pd.get("confidence", 0.0),
            "cameraId": "CAM_01",
            "cameraName": f"CAM_01_{status.source_info[:20]}",
            "isAlarm": pd.get("activity") == "fall",
            "fallCnnLabel": pd.get("fall_cnn_label", "Unknown"),
            "fallCnnConfidence": pd.get("fall_cnn_confidence", 0.0),
            "squatCount": pd.get("squat_count", 0),
            "personIdRaw": pid,
        })
    return entities


def _build_alerts(status: PipelineStatus) -> List[dict]:
    alerts = []
    for a in status.active_alerts:
        ts = a.get("timestamp")
        alerts.append({
            "id": a.get("alert_id", ""),
            "severity": a.get("severity", "WARNING"),
            "alertType": a.get("alert_type", "unknown"),
            "title": f"{a.get('severity', 'WARNING')}: {a.get('alert_type', 'unknown')}",
            "location": "CAM_01",
            "personId": a.get("person_id", 0),
            "targetId": f"#{a.get('person_id', 0):03d}",
            "confidence": a.get("confidence", 0.0),
            "status": a.get("status", "active"),
            "timestamp": ts,
            "timestampDisplay": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "",
            "acknowledged": a.get("status") == "acknowledged",
        })
    return alerts


def _status_dict(status: PipelineStatus) -> dict:
    return {
        "isRunning": status.is_running,
        "fps": round(status.fps, 1),
        "inferenceTimeMs": round(status.inference_time_ms, 1),
        "frameNumber": status.frame_number,
        "activePeople": status.active_people,
        "totalActivities": status.total_activities,
        "totalFalls": status.total_falls,
        "totalPostureRisks": status.total_posture_risks,
        "totalReps": status.total_reps,
        "sourceInfo": status.source_info,
        "error": status.error,
    }


# ── REST routes ─────────────────────────────────────────

@bp.get("/api/health")
def health_check():
    return jsonify({"status": "ok", "version": "0.1.0"})


@bp.get("/api/status")
def get_status():
    return jsonify(_status_dict(_pipeline.status))


@bp.get("/api/live")
def get_live():
    """Combined snapshot polled by the frontend every ~1s in place of a websocket."""
    status = _pipeline.status
    return jsonify({
        "type": "frame_update",
        "status": _status_dict(status),
        "cameras": _build_camera_feed(status),
        "trackedEntities": _build_tracked_entities(status),
        "alerts": _build_alerts(status),
    })


@bp.get("/api/events")
def get_events():
    if _pipeline.db is None:
        return jsonify([])
    person_id = request.args.get("person_id", type=int)
    activity_type = request.args.get("activity_type", type=str)
    limit = min(request.args.get("limit", default=100, type=int), 1000)
    offset = request.args.get("offset", default=0, type=int)
    return jsonify(_pipeline.db.query_activity_events(
        person_id=person_id, activity_type=activity_type, limit=limit, offset=offset,
    ))


@bp.get("/api/events/export")
def export_events_csv():
    if _pipeline.db is None:
        return Response("", mimetype="text/plain")
    return Response(_pipeline.db.export_events_csv(), mimetype="text/csv")


@bp.get("/api/events/stats")
def get_event_stats():
    if _pipeline.db is None:
        return jsonify({"typeCounts": {}, "avgDurations": {}})
    return jsonify({
        "typeCounts": _pipeline.db.get_activity_type_counts(),
        "avgDurations": _pipeline.db.get_average_duration_by_type(),
    })


@bp.get("/api/alerts")
def get_alerts():
    if _pipeline.db is None:
        return jsonify([])
    status = request.args.get("status", type=str)
    limit = min(request.args.get("limit", default=100, type=int), 1000)
    offset = request.args.get("offset", default=0, type=int)
    return jsonify(_pipeline.db.query_alerts(status=status, limit=limit, offset=offset))


@bp.post("/api/alerts/<alert_id>/acknowledge")
def acknowledge_alert(alert_id: str):
    if _pipeline.alert_engine is None:
        return jsonify({"success": False, "error": "Pipeline not running"})
    success = _pipeline.alert_engine.acknowledge_alert(alert_id)
    return jsonify({"success": success})


@bp.post("/api/alerts/<alert_id>/resolve")
def resolve_alert(alert_id: str):
    if _pipeline.alert_engine is None:
        return jsonify({"success": False, "error": "Pipeline not running"})
    success = _pipeline.alert_engine.resolve_alert(alert_id)
    return jsonify({"success": success})


@bp.get("/api/config")
def get_config():
    s = get_settings()
    return jsonify({
        "poseModel": s.pose_model,
        "detectionThreshold": s.detection_confidence_threshold,
        "keypointThreshold": s.keypoint_confidence_threshold,
        "sequenceLength": s.sequence_buffer_length,
        "fpsTarget": 30,
        "fallConfirmFrames": s.fall_confirm_frames,
        "alertCooldownSeconds": s.alert_cooldown_seconds,
        "inactivityWarnSeconds": s.inactivity_warn_seconds,
        "videoSource": s.video_source,
        "useFallClassifier": s.use_fall_classifier,
        "fallClassifierModel": s.fall_classifier_model,
        "fallClassifierConfidenceThreshold": s.fall_classifier_confidence_threshold,
    })


@bp.put("/api/config")
def update_config():
    config = request.get_json(force=True) or {}
    overrides = {
        "pose_model": config.get("poseModel"),
        "detection_confidence_threshold": config.get("detectionThreshold"),
        "keypoint_confidence_threshold": config.get("keypointThreshold"),
        "sequence_buffer_length": config.get("sequenceLength"),
        "fall_confirm_frames": config.get("fallConfirmFrames"),
        "alert_cooldown_seconds": config.get("alertCooldownSeconds"),
        "inactivity_warn_seconds": config.get("inactivityWarnSeconds"),
        "video_source": config.get("videoSource"),
        "use_fall_classifier": config.get("useFallClassifier"),
        "fall_classifier_confidence_threshold": config.get("fallClassifierConfidenceThreshold"),
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    update_settings(overrides)
    return jsonify({"success": True})


@bp.post("/api/pipeline/start")
def start_pipeline():
    source = request.args.get("source", type=str) or (request.get_json(silent=True) or {}).get("source")
    success = _pipeline.start(source)
    return jsonify({"success": success, "error": _pipeline.status.error})


@bp.post("/api/pipeline/stop")
def stop_pipeline():
    _pipeline.stop()
    return jsonify({"success": True})


# ── Video upload ─────────────────────────────────────────

@bp.post("/api/upload")
def upload_video():
    """Accept a video file (multipart/form-data, field name ``video``),
    save it under the configured upload directory, and return its path
    so the frontend can use it as a pipeline source.
    """
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file provided (expected field 'video')"}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_VIDEO_EXTENSIONS))}",
        }), 400

    settings = get_settings()
    upload_dir = settings.upload_abs_path
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Prefix with a short uuid so concurrent/duplicate filenames never collide.
    stored_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    dest_path = upload_dir / stored_name

    # Enforce a max size while streaming to disk, rather than buffering
    # the whole upload in memory first.
    max_bytes = settings.upload_max_bytes
    bytes_written = 0
    chunk_size = 1024 * 1024
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    out.close()
                    dest_path.unlink(missing_ok=True)
                    return jsonify({
                        "success": False,
                        "error": f"File exceeds max upload size of {max_bytes // (1024 * 1024)} MB",
                    }), 413
                out.write(chunk)
    except Exception as e:
        logger.exception("Failed to save uploaded video")
        dest_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": f"Upload failed: {e}"}), 500

    if bytes_written == 0:
        dest_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": "Uploaded file is empty"}), 400

    logger.info("Saved uploaded video: %s (%d bytes)", dest_path, bytes_written)

    # Sanity-check that OpenCV can actually open it before handing it back.
    probe = cv2.VideoCapture(str(dest_path))
    playable = probe.isOpened()
    probe.release()
    if not playable:
        dest_path.unlink(missing_ok=True)
        return jsonify({
            "success": False,
            "error": "File was saved but could not be opened as a video (unsupported/corrupt codec).",
        }), 400

    return jsonify({
        "success": True,
        "filename": original_name,
        "path": str(dest_path),
        "sizeBytes": bytes_written,
    })


@bp.get("/api/uploads")
def list_uploads():
    """List previously uploaded video files, most recent first."""
    settings = get_settings()
    upload_dir = settings.upload_abs_path
    if not upload_dir.exists():
        return jsonify([])

    items = []
    for p in upload_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _ALLOWED_VIDEO_EXTENSIONS:
            stat = p.stat()
            items.append({
                "filename": p.name,
                "path": str(p),
                "sizeBytes": stat.st_size,
                "modified": stat.st_mtime,
            })
    items.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify(items)


# ── MJPEG video stream ───────────────────────────────────

def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = _latest_frame
        if frame is None:
            time.sleep(0.3)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        chunk = (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n" +
            buf.tobytes() + b"\r\n"
        )
        yield chunk
        time.sleep(1 / 30)


@bp.get("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
