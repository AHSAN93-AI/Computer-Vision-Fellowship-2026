"""
Flask entrypoint. Wires the pipeline together:

    upload -> preprocessing.quality -> preprocessing.transforms
           -> vision.classifier + vision.anomaly
           -> inspection.severity -> inspection.decision
           -> inspection.evidence -> inspection.result -> database.db
"""

import csv
import io
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request, render_template, send_file, Response

from app.config import (
    MODEL_VERSION,
    EVIDENCE_DIR,
    runtime_settings,
    DEFAULT_DECISION_THRESHOLDS,
    DEFAULT_SEVERITY_THRESHOLDS,
)
from app.preprocessing.quality import check_image_quality
from app.preprocessing.transforms import decode_image, to_model_tensor
from app.vision import classifier, anomaly
from app.inspection.severity import classify_severity, max_severity
from app.inspection.decision import decide
from app.inspection.result import InspectionResult
from app.inspection.evidence import save_evidence
from app.database import db

app = Flask(__name__, template_folder="../templates", static_folder="../static")


def _load_persisted_settings():
    saved = db.load_settings()
    if saved.get("decision"):
        runtime_settings["decision"] = saved["decision"]
    if saved.get("severity"):
        runtime_settings["severity"] = saved["severity"]


db.init_db()
_load_persisted_settings()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def page_inspect():
    return render_template("inspect.html")


@app.route("/history")
def page_history():
    return render_template("history.html")


@app.route("/analytics")
def page_analytics():
    return render_template("analytics.html")


@app.route("/settings")
def page_settings():
    return render_template("settings.html")


# ---------------------------------------------------------------------------
# API: run an inspection
# ---------------------------------------------------------------------------
@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    pipeline_start = time.perf_counter()

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name must be 'image')"}), 400

    file = request.files["image"]
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    product_type = request.form.get("product_type", "fabric_patch")
    inspection_id = f"insp_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        gray_full = decode_image(file_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # --- Requirement 18: quality validation ---
    quality = check_image_quality(gray_full)
    if not quality["valid"]:
        elapsed_ms = (time.perf_counter() - pipeline_start) * 1000.0
        result = InspectionResult(
            inspection_id=inspection_id,
            timestamp=timestamp,
            product_type=product_type,
            model_version=MODEL_VERSION,
            status="INVALID",
            processing_time_ms=elapsed_ms,
            invalid_reason=quality["reason"],
        )
        db.insert_inspection(result.to_dict())
        return jsonify(result.to_dict())

    tensor = to_model_tensor(gray_full)

    # --- classification ---
    cls_result = classifier.predict(tensor)

    # --- anomaly analysis ---
    anomaly_threshold = runtime_settings["decision"]["anomaly_score_threshold"]
    an_result = anomaly.anomaly_map(tensor, anomaly_threshold)

    # --- defect size estimation ---
    defect_area_ratio = float(an_result["thresholded_mask"].mean())

    # --- severity ---
    predicted_class = cls_result["predicted_class"]
    severity_for_class = (
        classify_severity(predicted_class, defect_area_ratio)
        if predicted_class != "good"
        else "None"
    )

    # --- decision engine ---
    decision = decide(
        predicted_class=predicted_class,
        classifier_confidence=cls_result["confidence"],
        is_anomalous=an_result["is_anomalous"],
        anomaly_score=an_result["anomaly_score"],
        defect_area_ratio=defect_area_ratio,
    )

    severities = [severity_for_class] if decision["status"] == "FAIL" and predicted_class != "good" else []
    if decision["status"] == "FAIL" and "anomaly" in decision["defects_detected"]:
        severities.append(classify_severity("anomaly", defect_area_ratio))
    overall_severity = max_severity(severities) if severities else "None"

    # --- evidence capture ---
    evidence_path = save_evidence(
        inspection_id=inspection_id,
        original_gray=gray_full,
        error_map=an_result["error_map"],
        status=decision["status"],
        max_severity=overall_severity,
        predicted_class=predicted_class,
    )

    elapsed_ms = (time.perf_counter() - pipeline_start) * 1000.0

    result = InspectionResult(
        inspection_id=inspection_id,
        timestamp=timestamp,
        product_type=product_type,
        model_version=MODEL_VERSION,
        status=decision["status"],
        defects_detected=decision["defects_detected"],
        defect_count=decision["defect_count"],
        max_severity=overall_severity,
        classifier_confidence=cls_result["confidence"],
        anomaly_score=an_result["anomaly_score"],
        defect_area_ratio=defect_area_ratio,
        processing_time_ms=elapsed_ms,
        evidence_path=evidence_path,
    )

    defect_confidences = {}
    if predicted_class in decision["defects_detected"]:
        defect_confidences[predicted_class] = cls_result["confidence"]
    if "anomaly" in decision["defects_detected"]:
        defect_confidences["anomaly"] = an_result["anomaly_score"]

    db.insert_inspection(result.to_dict(), defect_confidences)

    response = result.to_dict()
    response["reasons"] = decision["reasons"]
    response["class_probabilities"] = cls_result["class_probabilities"]
    return jsonify(response)


# ---------------------------------------------------------------------------
# API: history
# ---------------------------------------------------------------------------
@app.route("/api/inspections")
def api_inspections():
    status = request.args.get("status")
    defect = request.args.get("defect")
    date = request.args.get("date")
    rows = db.query_inspections(status=status, defect=defect, date=date)
    return jsonify(rows)


@app.route("/api/inspections/<inspection_id>/evidence/<evidence_type>")
def api_evidence(inspection_id, evidence_type):
    if evidence_type not in ("original", "heatmap", "annotated"):
        return jsonify({"error": "invalid evidence type"}), 400
    import os

    path = os.path.join(EVIDENCE_DIR, inspection_id, f"{evidence_type}.png")
    if not os.path.exists(path):
        return jsonify({"error": "evidence not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/inspections/export")
def api_export_csv():
    status = request.args.get("status")
    defect = request.args.get("defect")
    date = request.args.get("date")
    rows = db.query_inspections(status=status, defect=defect, date=date, limit=100000)

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=inspections.csv"},
    )


# ---------------------------------------------------------------------------
# API: analytics
# ---------------------------------------------------------------------------
@app.route("/api/analytics")
def api_analytics():
    return jsonify(db.get_analytics())


# ---------------------------------------------------------------------------
# API: settings
# ---------------------------------------------------------------------------
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify(
            {
                "decision": runtime_settings["decision"],
                "severity": runtime_settings["severity"],
                "defaults": {
                    "decision": DEFAULT_DECISION_THRESHOLDS,
                    "severity": DEFAULT_SEVERITY_THRESHOLDS,
                },
            }
        )

    payload = request.get_json(force=True, silent=True) or {}
    if "decision" in payload:
        runtime_settings["decision"].update(payload["decision"])
    if "severity" in payload:
        runtime_settings["severity"].update(payload["severity"])
    db.save_settings(runtime_settings["decision"], runtime_settings["severity"])
    return jsonify({"decision": runtime_settings["decision"], "severity": runtime_settings["severity"]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
