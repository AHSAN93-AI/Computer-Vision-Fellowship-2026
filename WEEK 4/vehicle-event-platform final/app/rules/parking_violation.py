"""
app/rules/parking_violation.py — Two-stage parking violation rule.

Stage 1 (ZONE_INTRUSION): fires immediately when a vehicle enters the restricted zone.
Stage 2 (PARKING_VIOLATION): escalates if the vehicle is stationary AND
dwell > grace_period_seconds. Distinct event type from mere intrusion.
"""

from __future__ import annotations

import logging
import time
from typing import List

from app.rules.base_rule import BaseRule, EventType, RuleContext, RuleViolation, Severity

logger = logging.getLogger(__name__)


class ParkingViolationRule(BaseRule):
    """
    Detects vehicles parked illegally in a no-parking/restricted zone.

    Two-stage:
    1. ZONE_INTRUSION: any vehicle entering the zone (immediate).
    2. PARKING_VIOLATION: vehicle is stationary AND dwell > grace_period.

    Stationary check: centroid variance < px_threshold over last N frames.
    """

    def __init__(
        self,
        rule_id_or_config=None,
        name: str = "",
        zone_id: str = "",
        zone_name: str = "",
        grace_period_seconds: float = 30.0,
        stationary_px_threshold: float = 10.0,
        stationary_frames: int = 15,
        enabled: bool = True,
        rule_id: str = "",
    ):
        # Support passing a RuleConfig as the first positional arg (from tests)
        from app.config import RuleConfig as RC
        if isinstance(rule_id_or_config, RC):
            cfg = rule_id_or_config
            _rule_id = cfg.id
            _name = cfg.name or "Parking Violation"
            _zone_id = cfg.zone
            _zone_name = zone_name
            _grace = cfg.threshold_seconds if cfg.threshold_seconds else grace_period_seconds
            _px = cfg.stationary_px_threshold if cfg.stationary_px_threshold else stationary_px_threshold
            _frames = cfg.stationary_frames if cfg.stationary_frames else stationary_frames
            _enabled = cfg.enabled
        else:
            _rule_id = rule_id_or_config or rule_id
            _name = name
            _zone_id = zone_id
            _zone_name = zone_name
            _grace = grace_period_seconds
            _px = stationary_px_threshold
            _frames = stationary_frames
            _enabled = enabled

        super().__init__(
            rule_id=_rule_id,
            name=_name,
            event_type=EventType.PARKING_VIOLATION,
            severity=Severity.CRITICAL,
            zone_id=_zone_id,
            enabled=_enabled,
        )
        self._zone_name = _zone_name
        self._grace_period = _grace
        self._px_threshold = _px
        self._stationary_frames = _frames

    def evaluate(self, context: RuleContext) -> List[RuleViolation]:
        if not self.enabled:
            return []

        violations: List[RuleViolation] = []

        try:
            vehicles_in_zone = context.vehicles_in_zone(self.zone_id)

            for tv in vehicles_in_zone:
                dwell = context.dwell_tracker.current_dwell(tv.track_id, self.zone_id)

                # Check if stationary
                stationary = tv.is_stationary(
                    px_threshold=self._px_threshold,
                    window=self._stationary_frames,
                )

                if dwell >= self._grace_period and stationary:
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        event_type=EventType.PARKING_VIOLATION,
                        severity=Severity.CRITICAL,
                        track_id=tv.track_id,
                        zone_id=self.zone_id,
                        zone_name=self._zone_name,
                        class_name=tv.class_name,
                        description=(
                            f"{tv.class_name.capitalize()} #{tv.track_id} parked illegally in "
                            f"'{self._zone_name}' for {dwell:.0f}s (grace={self._grace_period:.0f}s)"
                        ),
                        timestamp=context.timestamp,
                        confidence=tv.confidence,
                        metadata={
                            "dwell_seconds": round(dwell, 1),
                            "grace_period_seconds": self._grace_period,
                            "stationary": stationary,
                        },
                    ))
        except Exception as e:
            logger.error(f"ParkingViolationRule.evaluate error: {e}")

        return violations
