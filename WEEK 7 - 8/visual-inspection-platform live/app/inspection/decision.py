"""
Requirement 10: Quality Decision Engine.

Combines classifier confidence + anomaly score + defect area + defect count
into PASS / FAIL. Kept as its own module, separate from vision/*, so the
business rule can change without touching model code.

    IF critical_defect_detected:      FAIL
    ELSE IF defect_count > allowed:   FAIL
    ELSE IF defect_area > maximum:    FAIL
    ELSE:                             PASS

INVALID is decided earlier, by preprocessing/quality.py, before this
module ever runs.
"""

from app.config import runtime_settings


def decide(
    predicted_class: str,
    classifier_confidence: float,
    is_anomalous: bool,
    anomaly_score: float,
    defect_area_ratio: float,
) -> dict:
    """
    Returns:
        {
          "status": "PASS" | "FAIL",
          "defects_detected": [str, ...],
          "defect_count": int,
          "reasons": [str, ...],
        }
    """
    settings = runtime_settings["decision"]
    reasons = []
    defects_detected = []

    classifier_flags_defect = (
        predicted_class != "good"
        and classifier_confidence >= settings["classifier_confidence_threshold"]
    )

    if classifier_flags_defect:
        defects_detected.append(predicted_class)

    if is_anomalous and "anomaly" not in defects_detected:
        # Anomaly model doesn't name a class -- it flags "abnormal" generally,
        # which matters most for defect types the classifier never saw trained.
        defects_detected.append("anomaly")

    defect_count = len(defects_detected)

    critical_hit = predicted_class in settings["critical_classes"] and classifier_flags_defect
    if critical_hit:
        reasons.append(f"Critical defect class detected: {predicted_class}")

    count_exceeded = defect_count > settings["max_allowed_defect_count"]
    if count_exceeded:
        reasons.append(
            f"Defect count {defect_count} exceeds allowed {settings['max_allowed_defect_count']}"
        )

    area_exceeded = defect_area_ratio > settings["max_allowed_area_ratio"]
    if area_exceeded:
        reasons.append(
            f"Defect area ratio {defect_area_ratio:.4f} exceeds allowed "
            f"{settings['max_allowed_area_ratio']}"
        )

    if critical_hit or count_exceeded or area_exceeded:
        status = "FAIL"
    else:
        status = "PASS"
        defects_detected = []  # PASS never lists defects, even minor classifier noise

    return {
        "status": status,
        "defects_detected": defects_detected,
        "defect_count": len(defects_detected),
        "reasons": reasons,
    }
