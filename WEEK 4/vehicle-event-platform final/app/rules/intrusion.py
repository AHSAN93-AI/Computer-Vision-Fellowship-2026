"""
app/rules/intrusion.py — Restricted zone intrusion detection rule.

Fires ZONE_INTRUSION immediately when a vehicle enters a restricted zone.
Uses per-track state to avoid re-firing for the same entry.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Set

from app.rules.base_rule import BaseRule, EventType, RuleContext, RuleViolation, Severity

logger = logging.getLogger(__name__)


class IntrusionRule(BaseRule):
    """
    Detects vehicles entering a restricted zone.

    Fires immediately on zone entry. Tracks which vehicles have already
    triggered to avoid duplicate alerts for the same entry.
    Resets when the vehicle exits the zone.
    """

    def __init__(
        self,
        rule_id_or_config=None,
        name: str = "",
        zone_id: str = "",
        zone_name: str = "",
        enabled: bool = True,
        rule_id: str = "",
    ):
        # Support passing a RuleConfig as the first positional arg (from tests)
        from app.config import RuleConfig as RC
        if isinstance(rule_id_or_config, RC):
            cfg = rule_id_or_config
            _rule_id = cfg.id
            _name = cfg.name or "Zone Intrusion"
            _zone_id = cfg.zone
            _zone_name = zone_name
            _enabled = cfg.enabled
        else:
            _rule_id = rule_id_or_config or rule_id
            _name = name
            _zone_id = zone_id
            _zone_name = zone_name
            _enabled = enabled

        super().__init__(
            rule_id=_rule_id,
            name=_name,
            event_type=EventType.ZONE_INTRUSION,
            severity=Severity.HIGH,
            zone_id=_zone_id,
            enabled=_enabled,
        )
        self._zone_name = _zone_name
        # Track IDs that have already been alerted for current entry
        self._alerted_tracks: Set[int] = set()

    def evaluate(self, context: RuleContext) -> List[RuleViolation]:
        if not self.enabled:
            return []

        violations: List[RuleViolation] = []

        try:
            vehicles_in_zone = context.vehicles_in_zone(self.zone_id)
            in_zone_ids = {tv.track_id for tv in vehicles_in_zone}

            # Clear tracks that have exited the zone
            exited = self._alerted_tracks - in_zone_ids
            self._alerted_tracks -= exited

            # Fire for new entries
            for tv in vehicles_in_zone:
                if tv.track_id not in self._alerted_tracks:
                    self._alerted_tracks.add(tv.track_id)

                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        event_type=EventType.ZONE_INTRUSION,
                        severity=Severity.HIGH,
                        track_id=tv.track_id,
                        zone_id=self.zone_id,
                        zone_name=self._zone_name,
                        class_name=tv.class_name,
                        description=(
                            f"{tv.class_name.capitalize()} #{tv.track_id} entered "
                            f"restricted zone '{self._zone_name}'"
                        ),
                        timestamp=context.timestamp,
                        confidence=tv.confidence,
                        metadata={
                            "centroid": list(tv.centroid),
                            "bbox": list(tv.bbox),
                        },
                    ))
                    logger.info(
                        f"Intrusion detected: track={tv.track_id} zone={self.zone_id}"
                    )

            # Clean up stale tracks no longer being tracked
            all_track_ids = {tv.track_id for tv in context.tracked_vehicles}
            self._alerted_tracks &= all_track_ids

        except Exception as e:
            logger.error(f"IntrusionRule.evaluate error: {e}")

        return violations
