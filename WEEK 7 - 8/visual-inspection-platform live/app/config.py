"""
Central configuration for the AI Visual Quality Inspection Platform.

Everything that a human might want to tune without touching model code
lives here: paths, the class list the classifier was trained with, and
the two threshold groups (`DecisionThresholds`, `SeverityThresholds`)
that the /api/settings endpoint reads and writes at runtime.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Model / image constants -- these MUST match how the .pt files were trained.
# ---------------------------------------------------------------------------
CLASS_NAMES = ["color", "cut", "good", "hole", "metal_contamination", "thread"]
NORMAL_CLASS = "good"
IMG_SIZE = 64
IN_CHANNELS = 1

CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "models", "defect_cnn.pt")
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_autoencoder.pt")

MODEL_VERSION = "classifier_v1 + anomaly_v1"

# ---------------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------------
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
DATABASE_PATH = os.path.join(BASE_DIR, "database", "inspections.db")

# ---------------------------------------------------------------------------
# Image quality validation (Requirement 18)
# ---------------------------------------------------------------------------
QUALITY_THRESHOLDS = {
    "min_brightness": 25,      # mean pixel value (0-255) below this -> too dark
    "max_brightness": 230,     # mean pixel value above this -> too bright/blown out
    "min_laplacian_var": 30.0,  # variance of Laplacian below this -> too blurred
}

# ---------------------------------------------------------------------------
# Decision engine thresholds (Requirement 10 / 14)
# Runtime-editable via GET/POST /api/settings.
# ---------------------------------------------------------------------------
DEFAULT_DECISION_THRESHOLDS = {
    "classifier_confidence_threshold": 0.50,   # below this, classifier vote is ignored
    "anomaly_score_threshold": 0.62,           # anomaly score >= this -> anomalous
    "max_allowed_defect_count": 0,             # any confirmed defect fails the part
    "max_allowed_area_ratio": 0.02,            # defect_area_ratio above this -> FAIL
    "critical_classes": ["hole", "cut"],       # these always fail regardless of area
}

# ---------------------------------------------------------------------------
# Severity thresholds (Requirement 9)
# `defect_area_ratio` bands, plus a floor severity for certain classes.
# ---------------------------------------------------------------------------
DEFAULT_SEVERITY_THRESHOLDS = {
    "minor_max_area_ratio": 0.01,     # <= this -> Minor
    "major_max_area_ratio": 0.05,     # <= this (and > minor) -> Major, above -> Critical
    "minimum_severity_by_class": {
        "hole": "Major",
        "cut": "Major",
        "metal_contamination": "Major",
    },
}

# Mutable, process-local "current" settings. `app/inspection/*` modules
# read from here so that a POST to /api/settings takes effect immediately
# without restarting the server. Persisted to the DB `settings` table too.
runtime_settings = {
    "decision": dict(DEFAULT_DECISION_THRESHOLDS),
    "severity": dict(DEFAULT_SEVERITY_THRESHOLDS),
}
