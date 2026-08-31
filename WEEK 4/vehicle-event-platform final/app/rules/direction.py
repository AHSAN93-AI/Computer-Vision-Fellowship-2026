"""
app/rules/direction.py — Wrong-direction detection rule.

Monitors line-crossing events and fires WRONG_DIRECTION violations
when a vehicle crosses in the unexpected direction.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Set

from app.analytics.lines import CrossingDirection, LineCrossingEvent
from app.rules.base_rule import BaseRule, EventType, RuleContext, RuleViolation, Severity

logger = logging.getLogger(__name__)


class WrongDirectionRule(BaseRule):
    """
    Detects vehicles crossing a line in the wrong direction.

    Uses line-crossing events from LineCrossingMonitor to determine
    if a vehicle has crossed against the expected direction.
    """

    def __init__(
        self,
        rule_id_or_config=None,
        name: str = "",
        zone_id: str = "",
        zone_name: str = "",
        expected_direction: str = "A_to_B",
        enabled: bool = True,
        rule_id: str = "",
    ):
        # Support passing a RuleConfig as the first positional arg (from tests)
        from app.config import RuleConfig as RC
        if isinstance(rule_id_or_config, RC):
            cfg = rule_id_or_config
            _rule_id = cfg.id
            _name = cfg.name or "Wrong Direction"
            _zone_id = cfg.zone
            _zone_name = zone_name
            _expected = cfg.expected_direction
            _enabled = cfg.enabled
        else:
            _rule_id = rule_id_or_config or rule_id
            _name = name
            _zone_id = zone_id
            _zone_name = zone_name
            _expected = expected_direction
            _enabled = enabled

        super().__init__(
            rule_id=_rule_id,
            name=_name,
            event_type=EventType.WRONG_DIRECTION,
            severity=Severity.CRITICAL,
            zone_id=_zone_id,
            enabled=_enabled,
        )
        self._zone_name = _zone_name
        self._expected_direction = CrossingDirection(_expected)

    def evaluate(self, context: RuleContext) -> List[RuleViolation]:
        if not self.enabled:
            return []

        violations: List[RuleViolation] = []

        try:
            # Check line-crossing events from this frame
            for lce in context.line_crossing_events:
                if lce.zone_id != self.zone_id:
                    continue

                if lce.is_wrong_direction:
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        event_type=EventType.WRONG_DIRECTION,
                        severity=Severity.CRITICAL,
                        track_id=lce.track_id,
                        zone_id=self.zone_id,
                        zone_name=self._zone_name,
                        class_name=lce.class_name,
                        description=(
                            f"{lce.class_name.capitalize()} #{lce.track_id} crossed "
                            f"'{self._zone_name}' in wrong direction "
                            f"({lce.direction.value}, expected {self._expected_direction.value})"
                        ),
                        timestamp=context.timestamp,
                        confidence=lce.confidence,
                        metadata={
                            "actual_direction": lce.direction.value,
                            "expected_direction": self._expected_direction.value,
                            "centroid": list(lce.centroid),
                        },
                    ))
                    logger.warning(
                        f"Wrong direction: track={lce.track_id} zone={self.zone_id} "
                        f"direction={lce.direction.value}"
                    )

        except Exception as e:
            logger.error(f"WrongDirectionRule.evaluate error: {e}")

        return violations
