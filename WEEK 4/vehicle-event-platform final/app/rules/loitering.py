"""
app/rules/loitering.py — Loitering / prolonged dwell detection.

State machine per (track_id, zone_id):
  IDLE → ENTERED → LOITERING → ALERTED

Key behaviors:
- Brief re-entry within reentry_window_seconds does NOT reset the clock.
- Once ALERTED, no new event until track leaves AND cooldown expires.
- Brief tracking loss (< grace_seconds in DwellTracker) doesn't reset timer.

Limitations (documented):
- Re-ID after full occlusion relies on ByteTrack IoU association.
  If a vehicle is completely occluded and re-appears after the lost_track_buffer,
  it gets a new track ID and the loitering timer resets.
- In very crowded scenes, track fragmentation can cause false resets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from app.rules.base_rule import BaseRule, EventType, RuleContext, RuleViolation, Severity

logger = logging.getLogger(__name__)


class LoiterState(str, Enum):
    IDLE = "IDLE"
    ENTERED = "ENTERED"
    LOITERING = "LOITERING"
    ALERTED = "ALERTED"


@dataclass
class LoiterTrackState:
    """State machine for one (track_id, zone_id) pair."""
    track_id: int
    zone_id: str
    state: LoiterState = LoiterState.IDLE
    entry_time: Optional[float] = None         # When they first entered
    last_in_zone: Optional[float] = None       # Last time seen in zone
    last_exit_time: Optional[float] = None     # When they last exited
    alerted_at: Optional[float] = None         # When we last sent an alert


class LoiteringRule(BaseRule):
    """
    Detects vehicles lingering beyond a dwell threshold.

    Uses dwell_tracker for accurate elapsed time (handles brief tracking loss).
    State machine prevents duplicate alerts.
    """

    def __init__(
        self,
        rule_id_or_config=None,
        name: str = "",
        zone_id: str = "",
        zone_name: str = "",
        threshold_seconds: float = 300.0,
        reentry_window_seconds: float = 60.0,
        cooldown_seconds: float = 300.0,
        enabled: bool = True,
        rule_id: str = "",
    ):
        # Support passing a RuleConfig as the first positional arg (from tests)
        from app.config import RuleConfig as RC
        if isinstance(rule_id_or_config, RC):
            cfg = rule_id_or_config
            _rule_id = cfg.id
            _name = cfg.name or "Loitering"
            _zone_id = cfg.zone
            _zone_name = zone_name
            _threshold = cfg.threshold_seconds if cfg.threshold_seconds else threshold_seconds
            _reentry = cfg.reentry_window_seconds if cfg.reentry_window_seconds else reentry_window_seconds
            _cooldown = cfg.cooldown_seconds if cfg.cooldown_seconds else cooldown_seconds
            _enabled = cfg.enabled
        else:
            _rule_id = rule_id_or_config or rule_id
            _name = name
            _zone_id = zone_id
            _zone_name = zone_name
            _threshold = threshold_seconds
            _reentry = reentry_window_seconds
            _cooldown = cooldown_seconds
            _enabled = enabled

        super().__init__(
            rule_id=_rule_id,
            name=_name,
            event_type=EventType.LOITERING,
            severity=Severity.WARNING,
            zone_id=_zone_id,
            enabled=_enabled,
        )
        self._zone_name = _zone_name
        self._threshold = _threshold
        self._reentry_window = _reentry
        self._cooldown = _cooldown
        # (track_id) → LoiterTrackState
        self._track_states: Dict[int, LoiterTrackState] = {}

    def _get_state(self, track_id: int) -> LoiterTrackState:
        if track_id not in self._track_states:
            self._track_states[track_id] = LoiterTrackState(
                track_id=track_id, zone_id=self.zone_id
            )
        return self._track_states[track_id]

    def evaluate(self, context: RuleContext) -> List[RuleViolation]:
        if not self.enabled:
            return []

        violations: List[RuleViolation] = []
        now = context.timestamp

        try:
            vehicles_in_zone = context.vehicles_in_zone(self.zone_id)
            in_zone_ids = {tv.track_id for tv in vehicles_in_zone}
            all_track_ids = {tv.track_id for tv in context.tracked_vehicles}

            # Process vehicles currently in zone
            for tv in vehicles_in_zone:
                ts = self._get_state(tv.track_id)
                dwell = context.dwell_tracker.current_dwell(tv.track_id, self.zone_id)

                if ts.state == LoiterState.IDLE:
                    ts.state = LoiterState.ENTERED
                    ts.entry_time = now
                    ts.last_in_zone = now
                    # Dwell may already exceed threshold on the very first
                    # evaluation of this track (e.g. dwell_tracker already had
                    # accumulated time, or re-entry restored an in-progress
                    # dwell). Fall through immediately instead of waiting a tick.
                    if dwell >= self._threshold:
                        ts.state = LoiterState.LOITERING

                if ts.state == LoiterState.ENTERED:
                    ts.last_in_zone = now
                    if dwell >= self._threshold:
                        ts.state = LoiterState.LOITERING

                if ts.state == LoiterState.LOITERING:
                    ts.last_in_zone = now
                    # Fire violation
                    ts.state = LoiterState.ALERTED
                    ts.alerted_at = now
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        event_type=EventType.LOITERING,
                        severity=Severity.WARNING,
                        track_id=tv.track_id,
                        zone_id=self.zone_id,
                        zone_name=self._zone_name,
                        class_name=tv.class_name,
                        description=(
                            f"{tv.class_name.capitalize()} #{tv.track_id} loitering in "
                            f"'{self._zone_name}' for {dwell:.0f}s (threshold={self._threshold:.0f}s)"
                        ),
                        timestamp=now,
                        confidence=tv.confidence,
                        metadata={
                            "dwell_seconds": round(dwell, 1),
                            "threshold_seconds": self._threshold,
                        },
                    ))
                    logger.info(f"Loitering detected: track={tv.track_id} zone={self.zone_id} dwell={dwell:.0f}s")

                elif ts.state == LoiterState.ALERTED:
                    ts.last_in_zone = now
                    # Check if enough time has passed to re-alert
                    if ts.alerted_at and (now - ts.alerted_at) >= self._cooldown:
                        ts.alerted_at = now
                        violations.append(RuleViolation(
                            rule_id=self.rule_id,
                            event_type=EventType.LOITERING,
                            severity=Severity.WARNING,
                            track_id=tv.track_id,
                            zone_id=self.zone_id,
                            zone_name=self._zone_name,
                            class_name=tv.class_name,
                            description=(
                                f"{tv.class_name.capitalize()} #{tv.track_id} still loitering in "
                                f"'{self._zone_name}' — total dwell {dwell:.0f}s"
                            ),
                            timestamp=now,
                            confidence=tv.confidence,
                            metadata={"dwell_seconds": round(dwell, 1), "re_alert": True},
                        ))

            # Handle vehicles that left the zone
            for tid, ts in list(self._track_states.items()):
                if tid not in in_zone_ids:
                    if ts.state in (LoiterState.ENTERED, LoiterState.LOITERING, LoiterState.ALERTED):
                        ts.last_exit_time = now

                    # If track is gone completely, reset after cooldown
                    if tid not in all_track_ids:
                        if ts.last_exit_time and (now - ts.last_exit_time) > self._cooldown:
                            del self._track_states[tid]

            # Handle re-entry: vehicle was in zone, left, comes back within reentry_window
            for tv in vehicles_in_zone:
                ts = self._track_states.get(tv.track_id)
                if ts and ts.state == LoiterState.IDLE and ts.last_exit_time:
                    if (now - ts.last_exit_time) <= self._reentry_window:
                        # Re-entry within window: restore state without resetting entry_time
                        ts.state = LoiterState.ENTERED
                        ts.last_in_zone = now
                        logger.debug(
                            f"Loitering re-entry within window: track={tv.track_id} "
                            f"exit_ago={(now - ts.last_exit_time):.0f}s"
                        )

        except Exception as e:
            logger.error(f"LoiteringRule.evaluate error: {e}")

        return violations
