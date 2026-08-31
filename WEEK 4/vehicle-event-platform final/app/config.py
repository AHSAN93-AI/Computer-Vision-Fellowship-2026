"""
app/config.py — Configuration loading, validation, and hot-reload.

Loads from config.yaml + .env overrides. Validates zone polygons and rule
integrity at startup. Logs warnings for invalid entries (graceful degradation).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ─── Load .env early ─────────────────────────────────────────────────────────
load_dotenv()

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    path: str = "yolov8n.pt"
    confidence: float = 0.40
    classes: List[str] = field(default_factory=lambda: ["car", "truck", "bus", "motorcycle"])
    input_resolution: Tuple[int, int] = (640, 640)
    device: str = "cpu"


@dataclass
class TrackerConfig:
    type: str = "bytetrack"
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8


@dataclass
class VideoConfig:
    default_source: str = "webcam"
    webcam_index: int = 0
    rtsp_url: str = ""
    frame_skip: int = 0
    max_fps: int = 30


@dataclass
class ZoneConfig:
    id: str = ""
    name: str = ""
    type: str = "polygon"                # "polygon" or "line_crossing"
    polygon: Optional[List[List[int]]] = None
    line: Optional[List[List[int]]] = None
    expected_direction: str = "A_to_B"
    in_label: str = "IN"
    out_label: str = "OUT"
    max_capacity: int = 0
    dwell_threshold_seconds: float = 300.0
    grace_period_seconds: float = 30.0
    monitored_classes: List[str] = field(
        default_factory=lambda: ["car", "truck", "bus", "motorcycle"]
    )
    rules: List[str] = field(default_factory=list)
    color: List[int] = field(default_factory=lambda: [0, 200, 255])

    def validate(self) -> bool:
        """Validate zone configuration. Returns False if invalid."""
        if not self.id or not self.name:
            logger.warning("Zone missing id or name — skipping")
            return False
        if self.type == "polygon":
            if not self.polygon or len(self.polygon) < 3:
                logger.warning(f"Zone '{self.id}' has invalid polygon (< 3 points) — skipping")
                return False
            for pt in self.polygon:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    logger.warning(f"Zone '{self.id}' has malformed polygon point {pt} — skipping")
                    return False
        elif self.type == "line_crossing":
            if not self.line or len(self.line) != 2:
                logger.warning(f"Zone '{self.id}' has invalid line (need exactly 2 points) — skipping")
                return False
        else:
            logger.warning(f"Zone '{self.id}' has unknown type '{self.type}' — skipping")
            return False
        return True


@dataclass
class RuleConfig:
    id: str = ""
    name: str = ""
    event_type: str = ""
    severity: str = "WARNING"
    enabled: bool = True
    zone: str = ""
    condition: str = ""
    threshold_seconds: float = 0.0
    threshold: int = 0
    cooldown_seconds: float = 30.0
    stationary_px_threshold: int = 10
    stationary_frames: int = 15
    reentry_window_seconds: float = 60.0
    expected_direction: str = "A_to_B"
    hysteresis: int = 2

    def validate(self) -> bool:
        if not self.id or not self.event_type:
            logger.warning(f"Rule '{self.id}' missing id or event_type — skipping")
            return False
        return True


@dataclass
class DwellConfig:
    lost_track_grace_seconds: float = 5.0
    max_history_minutes: int = 60


@dataclass
class EvidenceConfig:
    enabled: bool = True
    save_full_frame: bool = True
    save_crop: bool = True
    save_metadata_json: bool = True
    clip_enabled: bool = True
    clip_before_seconds: float = 3.0
    clip_after_seconds: float = 3.0
    clip_fps: int = 10
    max_frame_buffer: int = 90
    storage_path: str = "./evidence"


@dataclass
class AlertChannelConfig:
    type: str = ""
    enabled: bool = True
    endpoint: str = ""
    method: str = "POST"
    recipient: str = ""


@dataclass
class AlertConfig:
    channels: List[AlertChannelConfig] = field(default_factory=list)


@dataclass
class EventConfig:
    min_active_duration_seconds: float = 5.0
    default_cooldown_seconds: float = 30.0
    max_active_events: int = 500


@dataclass
class DashboardConfig:
    live_refresh_interval_ms: int = 1000
    websocket_frame_quality: int = 70
    max_websocket_fps: int = 15
    occupancy_chart_minutes: int = 10


@dataclass
class PerformanceConfig:
    log_metrics_interval_seconds: int = 10
    enable_fps_overlay: bool = True
    enable_cpu_monitoring: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./logs/app.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    zones: List[ZoneConfig] = field(default_factory=list)
    rules: List[RuleConfig] = field(default_factory=list)
    dwell: DwellConfig = field(default_factory=DwellConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    events: EventConfig = field(default_factory=EventConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def zone_by_id(self, zone_id: str) -> Optional[ZoneConfig]:
        return next((z for z in self.zones if z.id == zone_id), None)

    def rule_by_id(self, rule_id: str) -> Optional[RuleConfig]:
        return next((r for r in self.rules if r.id == rule_id), None)


# ─── Global singleton ─────────────────────────────────────────────────────────
_config: Optional[AppConfig] = None


def _from_dict(cls, data: Dict[str, Any]) -> Any:
    """Shallow dataclass instantiation from dict, ignoring unknown keys."""
    if data is None:
        return cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in fields}
    return cls(**filtered)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Load and validate config from YAML file. Returns AppConfig."""
    global _config
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path} — using defaults")
        raw = {}
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {path}: {e} — using defaults")
        raw = {}

    # Model
    model_data = raw.get("model", {})
    # Apply .env overrides
    if os.getenv("MODEL_PATH"):
        model_data["path"] = os.getenv("MODEL_PATH")
    if os.getenv("MODEL_CONFIDENCE"):
        model_data["confidence"] = float(os.getenv("MODEL_CONFIDENCE"))
    model = _from_dict(ModelConfig, model_data)
    if isinstance(model.input_resolution, list):
        model.input_resolution = tuple(model.input_resolution)

    tracker = _from_dict(TrackerConfig, raw.get("tracker", {}))
    video = _from_dict(VideoConfig, raw.get("video", {}))

    # Zones — validate each
    zones: List[ZoneConfig] = []
    for zdata in raw.get("zones", []):
        z = ZoneConfig(**{k: v for k, v in zdata.items() if k in ZoneConfig.__dataclass_fields__})
        if z.validate():
            zones.append(z)

    # Rules — validate each
    rules: List[RuleConfig] = []
    for rdata in raw.get("rules", []):
        r = RuleConfig(**{k: v for k, v in rdata.items() if k in RuleConfig.__dataclass_fields__})
        if r.validate():
            rules.append(r)

    dwell = _from_dict(DwellConfig, raw.get("dwell", {}))
    evidence = _from_dict(EvidenceConfig, raw.get("evidence", {}))

    # Alert channels
    alert_channels = [
        AlertChannelConfig(**{k: v for k, v in ch.items() if k in AlertChannelConfig.__dataclass_fields__})
        for ch in raw.get("alerts", {}).get("channels", [])
    ]
    alerts = AlertConfig(channels=alert_channels)

    events = _from_dict(EventConfig, raw.get("events", {}))
    dashboard = _from_dict(DashboardConfig, raw.get("dashboard", {}))
    performance = _from_dict(PerformanceConfig, raw.get("performance", {}))
    logging_cfg = _from_dict(LoggingConfig, raw.get("logging", {}))

    _config = AppConfig(
        model=model,
        tracker=tracker,
        video=video,
        zones=zones,
        rules=rules,
        dwell=dwell,
        evidence=evidence,
        alerts=alerts,
        events=events,
        dashboard=dashboard,
        performance=performance,
        logging=logging_cfg,
    )

    logger.info(
        f"Config loaded: {len(zones)} zones, {len(rules)} rules, model={model.path}"
    )
    return _config


def get_config() -> AppConfig:
    """Get the global config singleton, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """Hot-reload config from disk."""
    logger.info("Reloading configuration...")
    return load_config()


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure root logger from LoggingConfig."""
    import logging.handlers

    log_path = Path(cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg.level.upper(), logging.INFO)

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        handlers.append(file_handler)
    except OSError as e:
        print(f"Warning: Could not open log file {log_path}: {e}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
