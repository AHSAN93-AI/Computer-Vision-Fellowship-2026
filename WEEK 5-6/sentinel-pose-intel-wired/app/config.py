"""
Sentinel Pose Intel — Configuration Module

Centralises every tuneable parameter. Values are loaded from environment
variables (and / or an .env file) via pydantic-settings, so they can be
overridden at deploy time without touching code.

At runtime the singleton ``get_settings()`` returns a frozen Settings
instance.  For hot-reload from the dashboard, ``update_settings()``
writes changes to a JSON sidecar and recreates the singleton.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_OVERRIDE_FILE = _PROJECT_ROOT / "data" / "settings_override.json"


class Settings(BaseSettings):
    """All configurable parameters for the platform.

    Load order (later wins):
        1. Field defaults (below)
        2. .env file in project root
        3. Real environment variables
        4. JSON sidecar overrides written by the dashboard
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Video source ────────────────────────────────────
    video_source: str = Field(
        default="0",
        description="Video file path, webcam index (0, 1, …), or RTSP URL.",
    )

    # ── Pose model ──────────────────────────────────────
    pose_model: str = Field(
        default="yolov8n-pose.pt",
        description="Ultralytics model name or local path.",
    )

    # ── Detection thresholds ────────────────────────────
    detection_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0
    )
    keypoint_confidence_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0
    )
    min_visible_keypoints: int = Field(
        default=8, ge=1, le=17,
        description="Minimum keypoints above confidence threshold to accept a pose.",
    )

    # ── Tracking ────────────────────────────────────────
    tracker_type: str = Field(
        default="bytetrack",
        description="Tracker backend: bytetrack or botsort.",
    )
    track_loss_timeout_frames: int = Field(
        default=30, ge=1,
        description="Frames to keep a lost track before cleanup.",
    )

    # ── Sequence buffer ─────────────────────────────────
    sequence_buffer_length: int = Field(
        default=30, ge=5, le=120,
        description=(
            "Rolling buffer size per person (frames). "
            "30 frames ≈ 1 s at 30 FPS — captures a full gait cycle and "
            "is long enough for fall detection while keeping memory bounded."
        ),
    )

    # ── Activity state machine ──────────────────────────
    activity_confirm_frames: int = Field(
        default=5, ge=1,
        description="Frames an activity must persist to be confirmed.",
    )
    activity_end_frames: int = Field(
        default=8, ge=1,
        description="Frames an activity must be absent to be ended.",
    )

    # ── Standing ────────────────────────────────────────
    standing_max_velocity: float = Field(default=0.008)
    standing_max_torso_angle: float = Field(default=15.0)

    # ── Walking ─────────────────────────────────────────
    walking_velocity_threshold: float = Field(default=0.015)

    # ── Sitting ─────────────────────────────────────────
    sitting_max_knee_angle: float = Field(default=120.0)

    # ── Hand raise ──────────────────────────────────────
    hand_raise_min_elbow_angle: float = Field(default=120.0)

    # ── Bending ─────────────────────────────────────────
    bending_min_torso_angle: float = Field(default=45.0)

    # ── Waving ──────────────────────────────────────────
    waving_min_oscillations: int = Field(default=2)

    # ── Fall detection ──────────────────────────────────
    fall_torso_angle_threshold: float = Field(default=60.0)
    fall_speed_threshold: float = Field(default=0.05)
    fall_confirm_frames: int = Field(default=15)
    fall_alert_cooldown_seconds: float = Field(default=30.0)
    fall_min_factors: int = Field(
        default=3, ge=1, le=5,
        description="Minimum number of fall factors (out of 5) to trigger.",
    )

    # ── CNN fall classifier (YOLOv8-cls, ensemble with rule-based) ──
    use_fall_classifier: bool = Field(
        default=True,
        description="If true, run the trained Fall/NotFall CNN classifier "
                    "alongside the rule-based FallRecogniser as a second signal.",
    )
    fall_classifier_model: str = Field(
        default="models/fall_classifier.pt",
        description="Path to the YOLOv8-cls Fall/NotFall checkpoint.",
    )
    fall_classifier_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum per-frame classifier confidence to count as a Fall vote.",
    )
    fall_classifier_vote_window: int = Field(
        default=10, ge=1,
        description="Number of recent frames considered for the classifier vote.",
    )
    fall_classifier_vote_required: int = Field(
        default=6, ge=1,
        description="Number of Fall votes (out of the window) required to confirm.",
    )

    # ── Repetition counting (squats) ────────────────────
    squat_down_knee_angle: float = Field(default=90.0)
    squat_up_knee_angle: float = Field(default=160.0)
    squat_hysteresis_angle: float = Field(default=110.0)

    # ── Ergonomic monitoring ────────────────────────────
    ergo_bend_warn_seconds: float = Field(default=15.0)
    ergo_crouch_warn_seconds: float = Field(default=15.0)
    ergo_crouch_max_knee_angle: float = Field(default=100.0)

    # ── Alerts ──────────────────────────────────────────
    alert_cooldown_seconds: float = Field(default=60.0)
    inactivity_warn_seconds: float = Field(default=300.0)

    # ── Database ────────────────────────────────────────
    database_path: str = Field(default="data/sentinel_pose.db")

    # ── Evidence ────────────────────────────────────────
    evidence_dir: str = Field(default="evidence")

    # ── Uploaded video files ─────────────────────────────
    upload_dir: str = Field(
        default="uploads",
        description="Directory where videos uploaded via the dashboard are stored.",
    )
    upload_max_bytes: int = Field(
        default=500 * 1024 * 1024,
        description="Maximum accepted size (bytes) for an uploaded video file.",
    )

    # ── Server ──────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Logging ─────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/sentinel_pose.log")

    # ── Derived helpers ─────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def database_abs_path(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def evidence_abs_path(self) -> Path:
        p = Path(self.evidence_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def upload_abs_path(self) -> Path:
        p = Path(self.upload_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def log_file_abs_path(self) -> Path:
        p = Path(self.log_file)
        return p if p.is_absolute() else _PROJECT_ROOT / p


# ── Singleton access ────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide Settings singleton.

    On first call the object is created from env / .env, then
    patched with any overrides from the JSON sidecar written by
    ``update_settings()``.
    """
    settings = Settings()  # type: ignore[call-arg]

    # Apply persisted dashboard overrides if present
    if _SETTINGS_OVERRIDE_FILE.exists():
        try:
            overrides = json.loads(_SETTINGS_OVERRIDE_FILE.read_text(encoding="utf-8"))
            # Reconstruct with overrides applied
            merged = settings.model_dump()
            merged.update(overrides)
            settings = Settings(**merged)  # type: ignore[call-arg]
            logger.info("Applied %d setting overrides from %s", len(overrides), _SETTINGS_OVERRIDE_FILE)
        except Exception:
            logger.warning("Failed to load settings overrides from %s", _SETTINGS_OVERRIDE_FILE, exc_info=True)

    return settings


def update_settings(overrides: dict) -> Settings:
    """Persist dashboard-driven config changes and refresh the singleton.

    Only keys that exist in Settings are accepted; unknown keys are
    silently dropped.
    """
    valid_keys = set(Settings.model_fields.keys())
    clean = {k: v for k, v in overrides.items() if k in valid_keys}

    # Merge with existing overrides
    existing: dict = {}
    if _SETTINGS_OVERRIDE_FILE.exists():
        try:
            existing = json.loads(_SETTINGS_OVERRIDE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read existing settings overrides", exc_info=True)

    existing.update(clean)

    _SETTINGS_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_OVERRIDE_FILE.write_text(
        json.dumps(existing, indent=2), encoding="utf-8"
    )
    logger.info("Persisted %d setting overrides", len(clean))

    # Bust the lru_cache so the next call rebuilds
    get_settings.cache_clear()
    return get_settings()
