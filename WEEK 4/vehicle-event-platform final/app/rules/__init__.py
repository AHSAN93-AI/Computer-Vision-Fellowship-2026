# app/rules/__init__.py
"""
Rule engine package.

Rule factory builds all enabled rules from config.
Adding a new rule type: implement BaseRule, add to RULE_REGISTRY.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import AppConfig, RuleConfig, ZoneConfig
from app.rules.base_rule import BaseRule
from app.rules.parking_violation import ParkingViolationRule
from app.rules.intrusion import IntrusionRule
from app.rules.loitering import LoiteringRule
from app.rules.direction import WrongDirectionRule
from app.rules.occupancy import OccupancyRule

logger = logging.getLogger(__name__)


def build_rules(config: AppConfig) -> List[BaseRule]:
    """
    Build all configured, enabled rule instances from AppConfig.

    Each RuleConfig maps to a concrete BaseRule subclass.
    Unknown condition strings are logged and skipped (graceful).
    """
    rules: List[BaseRule] = []
    zone_map: Dict[str, ZoneConfig] = {z.id: z for z in config.zones}

    for rc in config.rules:
        if not rc.enabled:
            logger.info(f"Rule '{rc.id}' disabled — skipping")
            continue

        zone = zone_map.get(rc.zone, None)
        zone_name = zone.name if zone else rc.zone

        try:
            rule = _build_rule(rc, zone, zone_name)
            if rule is not None:
                rules.append(rule)
                logger.info(f"Registered rule: {rule.rule_id} ({rule.__class__.__name__})")
        except Exception as e:
            logger.error(f"Failed to build rule '{rc.id}': {e} — skipping")

    return rules


def _build_rule(rc: RuleConfig, zone: Any, zone_name: str) -> BaseRule:
    """Map a RuleConfig to the appropriate BaseRule subclass."""
    condition = rc.condition.lower()

    if condition == "stationary_in_zone":
        return ParkingViolationRule(
            rule_id=rc.id,
            name=rc.name,
            zone_id=rc.zone,
            zone_name=zone_name,
            grace_period_seconds=rc.threshold_seconds,
            stationary_px_threshold=float(rc.stationary_px_threshold),
            stationary_frames=rc.stationary_frames,
            enabled=rc.enabled,
        )

    elif condition == "zone_entry":
        return IntrusionRule(
            rule_id=rc.id,
            name=rc.name,
            zone_id=rc.zone,
            zone_name=zone_name,
            enabled=rc.enabled,
        )

    elif condition == "dwell_exceeded":
        return LoiteringRule(
            rule_id=rc.id,
            name=rc.name,
            zone_id=rc.zone,
            zone_name=zone_name,
            threshold_seconds=rc.threshold_seconds,
            reentry_window_seconds=rc.reentry_window_seconds,
            cooldown_seconds=rc.cooldown_seconds,
            enabled=rc.enabled,
        )

    elif condition == "direction_violation":
        return WrongDirectionRule(
            rule_id=rc.id,
            name=rc.name,
            zone_id=rc.zone,
            zone_name=zone_name,
            expected_direction=rc.expected_direction,
            enabled=rc.enabled,
        )

    elif condition == "occupancy_exceeded":
        cap = zone.max_capacity if zone else rc.threshold
        return OccupancyRule(
            rule_id=rc.id,
            name=rc.name,
            zone_id=rc.zone,
            zone_name=zone_name,
            max_capacity=cap,
            hysteresis=rc.hysteresis,
            enabled=rc.enabled,
        )

    else:
        logger.warning(f"Unknown rule condition '{rc.condition}' for rule '{rc.id}' — skipping")
        return None
