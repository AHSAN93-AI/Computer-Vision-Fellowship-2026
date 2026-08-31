"""
app/rules/occupancy.py — Over-capacity detection rule.

Monitors zone occupancy vs configured max_capacity.
Hysteresis: fires when over capacity, resolves when drops below max_capacity - hysteresis.
Prevents event flapping at the threshold boundary.
"""

from __future__ import annotations

import logging
from typing import List

from app.rules.base_rule import BaseRule, EventType, RuleContext, RuleViolation, Severity

logger = logging.getLogger(__name__)


class OccupancyRule(BaseRule):
    """
    Fires OVER_CAPACITY when zone occupancy exceeds configured threshold.
    Uses hysteresis (default 2) to prevent flapping.
    """

    def __init__(
        self,
        rule_id_or_config=None,
        name: str = "",
        zone_id: str = "",
        zone_name: str = "",
        max_capacity: int = 20,
        hysteresis: int = 2,
        enabled: bool = True,
        rule_id: str = "",
    ):
        # Support passing a RuleConfig as the first positional arg (from tests)
        from app.config import RuleConfig as RC
        if isinstance(rule_id_or_config, RC):
            cfg = rule_id_or_config
            _rule_id = cfg.id
            _name = cfg.name or "Over Capacity"
            _zone_id = cfg.zone
            _zone_name = zone_name
            _cap = cfg.threshold if cfg.threshold else max_capacity
            _hyst = hysteresis
            _enabled = cfg.enabled
        else:
            _rule_id = rule_id_or_config or rule_id
            _name = name
            _zone_id = zone_id
            _zone_name = zone_name
            _cap = max_capacity
            _hyst = hysteresis
            _enabled = enabled

        super().__init__(
            rule_id=_rule_id,
            name=_name,
            event_type=EventType.OVER_CAPACITY,
            severity=Severity.CRITICAL,
            zone_id=_zone_id,
            enabled=_enabled,
        )
        self._zone_name = _zone_name
        self._max_capacity = _cap
        self._hysteresis = _hyst

    def evaluate(self, context: RuleContext) -> List[RuleViolation]:
        if not self.enabled:
            return []

        violations: List[RuleViolation] = []

        try:
            is_over, changed = context.occupancy_monitor.check_over_capacity(
                self.zone_id, self._hysteresis
            )

            if is_over and changed:
                stats = context.occupancy_monitor.get_stats(self.zone_id)
                count = stats.current_count if stats else 0

                violations.append(RuleViolation(
                    rule_id=self.rule_id,
                    event_type=EventType.OVER_CAPACITY,
                    severity=Severity.CRITICAL,
                    track_id=None,
                    zone_id=self.zone_id,
                    zone_name=self._zone_name,
                    class_name=None,
                    description=(
                        f"Zone '{self._zone_name}' over capacity: "
                        f"{count} vehicles (max={self._max_capacity})"
                    ),
                    timestamp=context.timestamp,
                    confidence=1.0,
                    metadata={
                        "current_count": count,
                        "max_capacity": self._max_capacity,
                    },
                ))
                logger.warning(
                    f"Over capacity: zone={self.zone_id} count={count} max={self._max_capacity}"
                )

        except Exception as e:
            logger.error(f"OccupancyRule.evaluate error: {e}")

        return violations
