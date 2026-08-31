"""
app.events.alerts — Alert Engine (§4.17)

Generates alerts for:
  • Fall detected (CRITICAL)
  • Unsafe posture / ergonomic risk (WARNING)
  • Long inactivity (INFO)

Each alert carries event info + evidence and is:
  1. Pushed to connected dashboard clients via WebSocket (in-app).
  2. Persisted to the ``alert_events`` table in SQLite.

**Cooldown**: No repeated alerts for the same person + alert type
within ``alert_cooldown_seconds`` (default 60 s).

**Fall alert lifecycle** (§4.11):
  Possible Fall → Fall Confirmed → Alert Active → Acknowledged → Resolved.
  Managed here in coordination with the fall state machine.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from app.config import get_settings
from app.database.db import Database
from app.timeutil import monotonic_to_wall

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """In-memory alert representation."""

    alert_id: str
    person_id: int
    alert_type: str
    severity: str               # CRITICAL, WARNING, INFO
    timestamp: float
    confidence: float
    evidence_path: Optional[str]
    camera_source: str
    status: str = "active"      # active, acknowledged, resolved
    event_id: Optional[str] = None


class AlertEngine:
    """Manages alert generation, cooldowns, and lifecycle.

    Parameters
    ----------
    db:
        Database instance for persistence.
    on_alert:
        Optional callback invoked when a new alert is created.
        Signature: ``(alert: Alert) -> None``.  Used by the dashboard
        WebSocket to push alerts to clients.
    """

    def __init__(
        self,
        db: Database,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self._db = db
        self._on_alert = on_alert
        self._settings = get_settings()

        # Cooldown tracker: {(person_id, alert_type): last_alert_time}
        self._cooldowns: Dict[Tuple[int, str], float] = {}
        # Active alerts by person+type for lifecycle management
        self._active: Dict[Tuple[int, str], Alert] = {}

    def check_fall(
        self,
        person_id: int,
        is_fall_active: bool,
        confidence: float,
        evidence_path: Optional[str] = None,
        camera_source: str = "default",
        event_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """Check and generate a fall alert if appropriate.

        Called each frame for each person whose fall state machine is active.
        Handles cooldown and deduplication.
        """
        if not is_fall_active:
            return None

        return self._maybe_create_alert(
            person_id=person_id,
            alert_type="fall_detected",
            severity="CRITICAL",
            confidence=confidence,
            evidence_path=evidence_path,
            camera_source=camera_source,
            event_id=event_id,
            cooldown=self._settings.fall_alert_cooldown_seconds,
        )

    def check_ergonomic(
        self,
        person_id: int,
        is_bend_risk: bool,
        is_crouch_risk: bool,
        bend_duration: float = 0.0,
        crouch_duration: float = 0.0,
        evidence_path: Optional[str] = None,
        camera_source: str = "default",
    ) -> Optional[Alert]:
        """Check and generate an ergonomic risk alert."""
        if is_bend_risk:
            return self._maybe_create_alert(
                person_id=person_id,
                alert_type="posture_risk_bending",
                severity="WARNING",
                confidence=0.8,
                evidence_path=evidence_path,
                camera_source=camera_source,
                cooldown=self._settings.alert_cooldown_seconds,
            )
        if is_crouch_risk:
            return self._maybe_create_alert(
                person_id=person_id,
                alert_type="posture_risk_crouching",
                severity="WARNING",
                confidence=0.8,
                evidence_path=evidence_path,
                camera_source=camera_source,
                cooldown=self._settings.alert_cooldown_seconds,
            )
        return None

    def check_inactivity(
        self,
        person_id: int,
        current_activity: str,
        activity_duration: float,
        camera_source: str = "default",
    ) -> Optional[Alert]:
        """Check if a person has been inactive (standing/unknown) too long."""
        inactive_activities = {"standing", "Unknown"}
        if current_activity not in inactive_activities:
            return None
        if activity_duration < self._settings.inactivity_warn_seconds:
            return None

        return self._maybe_create_alert(
            person_id=person_id,
            alert_type="long_inactivity",
            severity="INFO",
            confidence=0.7,
            camera_source=camera_source,
            cooldown=self._settings.alert_cooldown_seconds,
        )

    def acknowledge_alert(self, alert_id: str, by: str = "operator") -> bool:
        """Acknowledge an alert (from dashboard)."""
        success = self._db.acknowledge_alert(alert_id, by)
        if success:
            # Update in-memory state
            for key, alert in self._active.items():
                if alert.alert_id == alert_id:
                    alert.status = "acknowledged"
                    break
            logger.info("Alert acknowledged: %s (by %s)", alert_id, by)
        return success

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        success = self._db.resolve_alert(alert_id)
        if success:
            for key, alert in list(self._active.items()):
                if alert.alert_id == alert_id:
                    alert.status = "resolved"
                    del self._active[key]
                    break
            logger.info("Alert resolved: %s", alert_id)
        return success

    def get_active_alerts(self) -> List[Alert]:
        """Return all currently active (unresolved) alerts."""
        return [a for a in self._active.values() if a.status in ("active", "acknowledged")]

    # ── Internal ────────────────────────────────────────

    def _maybe_create_alert(
        self,
        person_id: int,
        alert_type: str,
        severity: str,
        confidence: float,
        cooldown: float,
        evidence_path: Optional[str] = None,
        camera_source: str = "default",
        event_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """Create an alert if not in cooldown."""
        key = (person_id, alert_type)
        now = time.monotonic()

        # Check cooldown
        last_time = self._cooldowns.get(key, 0.0)
        if now - last_time < cooldown:
            return None

        # Check if there's already an active alert for this person+type
        if key in self._active and self._active[key].status == "active":
            return None

        # Create the alert
        alert_id = self._db.insert_alert(
            person_id=person_id,
            alert_type=alert_type,
            severity=severity,
            timestamp=monotonic_to_wall(now),
            confidence=confidence,
            camera_source=camera_source,
            evidence_path=evidence_path,
            event_id=event_id,
        )

        alert = Alert(
            alert_id=alert_id,
            person_id=person_id,
            alert_type=alert_type,
            severity=severity,
            timestamp=monotonic_to_wall(now),
            confidence=confidence,
            evidence_path=evidence_path,
            camera_source=camera_source,
            event_id=event_id,
        )

        self._active[key] = alert
        self._cooldowns[key] = now

        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                logger.warning("Alert callback failed", exc_info=True)

        return alert
