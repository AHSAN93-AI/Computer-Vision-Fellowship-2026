"""
Requirement 22 / 53: 15+ automated tests, business logic only.
No model inference is needed for any of these -- they exercise
preprocessing.quality, inspection.severity, inspection.decision,
inspection.result, inspection.evidence path generation, and database.db.
"""

import os
import sys
import tempfile
import uuid

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.quality import check_image_quality
from app.inspection.severity import classify_severity, max_severity
from app.inspection.decision import decide
from app.inspection.result import InspectionResult
from app.inspection.evidence import evidence_dir_for
from app.config import runtime_settings, DEFAULT_DECISION_THRESHOLDS, DEFAULT_SEVERITY_THRESHOLDS


@pytest.fixture(autouse=True)
def reset_settings():
    """Each test gets fresh default thresholds so tests can't leak state."""
    runtime_settings["decision"] = dict(DEFAULT_DECISION_THRESHOLDS)
    runtime_settings["severity"] = dict(DEFAULT_SEVERITY_THRESHOLDS)
    yield


# ---------------------------------------------------------------------------
# Quality checks (Requirement 18)
# ---------------------------------------------------------------------------
def test_quality_rejects_too_dark():
    dark = np.full((200, 200), 5, dtype=np.uint8)
    result = check_image_quality(dark)
    assert result["valid"] is False
    assert result["reason"] == "Image too dark"


def test_quality_rejects_too_bright():
    bright = np.full((200, 200), 250, dtype=np.uint8)
    result = check_image_quality(bright)
    assert result["valid"] is False
    assert result["reason"] == "Image too bright"


def test_quality_rejects_too_blurred():
    flat = np.full((200, 200), 128, dtype=np.uint8)  # zero variance -> "blurred"
    result = check_image_quality(flat)
    assert result["valid"] is False
    assert result["reason"] == "Image too blurred"


def test_quality_passes_good_image():
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
    result = check_image_quality(noisy)
    assert result["valid"] is True
    assert result["reason"] is None


# ---------------------------------------------------------------------------
# Severity rules (Requirement 9)
# ---------------------------------------------------------------------------
def test_severity_minor_band():
    assert classify_severity("thread", 0.005) == "Minor"


def test_severity_major_band():
    assert classify_severity("color", 0.03) == "Major"


def test_severity_critical_band():
    assert classify_severity("color", 0.10) == "Critical"


def test_severity_class_floor_overrides_small_area():
    # "hole" has a Major floor even at a tiny area ratio that would
    # otherwise be classified Minor.
    assert classify_severity("hole", 0.001) == "Major"


def test_severity_none_when_no_area():
    assert classify_severity("good", 0.0) == "None"


def test_max_severity_picks_highest():
    assert max_severity(["Minor", "Critical", "Major"]) == "Critical"


def test_max_severity_empty_list():
    assert max_severity([]) == "None"


# ---------------------------------------------------------------------------
# Decision engine (Requirement 10)
# ---------------------------------------------------------------------------
def test_decision_pass_on_good_low_area():
    decision = decide(
        predicted_class="good",
        classifier_confidence=0.95,
        is_anomalous=False,
        anomaly_score=0.01,
        defect_area_ratio=0.0,
    )
    assert decision["status"] == "PASS"
    assert decision["defect_count"] == 0


def test_decision_fail_on_critical_class():
    decision = decide(
        predicted_class="hole",
        classifier_confidence=0.9,
        is_anomalous=False,
        anomaly_score=0.01,
        defect_area_ratio=0.001,
    )
    assert decision["status"] == "FAIL"
    assert "hole" in decision["defects_detected"]


def test_decision_fail_on_defect_count_exceeded():
    runtime_settings["decision"]["max_allowed_defect_count"] = 0
    decision = decide(
        predicted_class="thread",
        classifier_confidence=0.9,
        is_anomalous=False,
        anomaly_score=0.01,
        defect_area_ratio=0.001,
    )
    assert decision["status"] == "FAIL"


def test_decision_fail_on_area_exceeded():
    decision = decide(
        predicted_class="good",
        classifier_confidence=0.95,
        is_anomalous=True,
        anomaly_score=0.9,
        defect_area_ratio=0.5,
    )
    assert decision["status"] == "FAIL"
    assert "anomaly" in decision["defects_detected"]


def test_decision_low_confidence_defect_class_is_ignored():
    # classifier says "thread" but with confidence under threshold -> shouldn't count
    runtime_settings["decision"]["classifier_confidence_threshold"] = 0.5
    decision = decide(
        predicted_class="thread",
        classifier_confidence=0.2,
        is_anomalous=False,
        anomaly_score=0.01,
        defect_area_ratio=0.0,
    )
    assert decision["status"] == "PASS"


# ---------------------------------------------------------------------------
# Result schema (Requirement 19)
# ---------------------------------------------------------------------------
def test_result_schema_has_required_fields():
    result = InspectionResult(
        inspection_id="insp_test01",
        timestamp="2026-08-29T00:00:00Z",
        product_type="fabric_patch",
        model_version="classifier_v1 + anomaly_v1",
        status="PASS",
    )
    d = result.to_dict()
    required = {
        "inspection_id", "timestamp", "product_type", "model_version", "status",
        "defects_detected", "defect_count", "max_severity", "classifier_confidence",
        "anomaly_score", "defect_area_ratio", "processing_time_ms", "evidence_path",
    }
    assert required.issubset(d.keys())
    assert isinstance(d["defects_detected"], list)
    assert isinstance(d["defect_count"], int)


def test_result_defaults_are_sane():
    result = InspectionResult(
        inspection_id="insp_test02",
        timestamp="2026-08-29T00:00:00Z",
        product_type="fabric_patch",
        model_version="v1",
        status="INVALID",
        invalid_reason="Image too blurred",
    )
    assert result.defect_count == 0
    assert result.max_severity == "None"
    assert result.invalid_reason == "Image too blurred"


# ---------------------------------------------------------------------------
# Evidence path generation (Requirement 20)
# ---------------------------------------------------------------------------
def test_evidence_path_is_deterministic():
    path_a = evidence_dir_for("insp_abc123")
    path_b = evidence_dir_for("insp_abc123")
    assert path_a == path_b


def test_evidence_path_is_unique_per_id():
    id1 = f"insp_{uuid.uuid4().hex[:8]}"
    id2 = f"insp_{uuid.uuid4().hex[:8]}"
    assert evidence_dir_for(id1) != evidence_dir_for(id2)


# ---------------------------------------------------------------------------
# Database round-trip (Requirement 21) -- uses a temp DB file, not the real one
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(monkeypatch):
    from app import config
    from app.database import db as db_module

    tmp_path = os.path.join(tempfile.gettempdir(), f"test_inspections_{uuid.uuid4().hex}.db")
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path)
    monkeypatch.setattr(db_module, "DATABASE_PATH", tmp_path)
    db_module.init_db()
    yield db_module
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_db_insert_and_retrieve_round_trip(temp_db):
    result = InspectionResult(
        inspection_id="insp_dbtest01",
        timestamp="2026-08-29T00:00:00Z",
        product_type="fabric_patch",
        model_version="v1",
        status="FAIL",
        defects_detected=["hole"],
        defect_count=1,
        max_severity="Major",
    ).to_dict()
    temp_db.insert_inspection(result, defect_confidences={"hole": 0.9})

    fetched = temp_db.get_inspection("insp_dbtest01")
    assert fetched is not None
    assert fetched["status"] == "FAIL"
    assert fetched["max_severity"] == "Major"


def test_db_filter_by_status(temp_db):
    for status in ("PASS", "FAIL", "PASS"):
        result = InspectionResult(
            inspection_id=f"insp_{uuid.uuid4().hex[:8]}",
            timestamp="2026-08-29T00:00:00Z",
            product_type="fabric_patch",
            model_version="v1",
            status=status,
        ).to_dict()
        temp_db.insert_inspection(result)

    passed = temp_db.query_inspections(status="PASS")
    failed = temp_db.query_inspections(status="FAIL")
    assert len(passed) == 2
    assert len(failed) == 1
