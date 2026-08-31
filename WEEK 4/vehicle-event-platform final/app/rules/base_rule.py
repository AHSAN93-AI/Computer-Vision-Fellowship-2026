"""
app/rules/base_rule.py — Abstract base class for all event rules.

Each rule receives a RuleContext (all current analytics state) and
returns an optional RuleViolation if its condition is met.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

from app.vision.tracker import TrackedVehicle
from app.analytics.zones import ZoneOccupancy
from app.analytics.dwell import DwellTracker
from app.analytics.occupancy import OccupancyMonitor
from app.analytics.lines import LineCrossingEvent


class EventType(str, Enum):
    ZONE_ENTRY = "ZONE_ENTRY"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_INTRUSION = "ZONE_INTRUSION"
    PARKING_VIOLATION = "PARKING_VIOLATION"
    LOITERING = "LOITERING"
    OVER_CAPACITY = "OVER_CAPACITY"
    WRONG_DIRECTION = "WRONG_DIRECTION"
    LINE_CROSS_IN = "LINE_CROSS_IN"
    LINE_CROSS_OUT = "LINE_CROSS_OUT"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RuleContext:
    """
    Snapshot of all analytics state for one frame.
    Passed to every rule's evaluate() method.
    """
    frame_id: int
    timestamp: float
    tracked_vehicles: List[TrackedVehicle]
    zone_occupancies: Dict[str, ZoneOccupancy]      # zone_id → ZoneOccupancy
    dwell_tracker: DwellTracker
    occupancy_monitor: OccupancyMonitor
    line_crossing_events: List[LineCrossingEvent]   # this frame's crossings

    def vehicles_in_zone(self, zone_id: str) -> List[TrackedVehicle]:
        occ = self.zone_occupancies.get(zone_id)
        if occ is None:
            return []
        return [
            tv for tv in self.tracked_vehicles
            if tv.track_id in occ.occupant_track_ids
        ]

    def get_vehicle(self, track_id: int) -> Optional[TrackedVehicle]:
        return next((tv for tv in self.tracked_vehicles if tv.track_id == track_id), None)


@dataclass
class RuleViolation:
    """
    A violation detected by a rule.

    The debouncing/event-manager layer converts these into Events.
    Multiple violations for the same (rule_id, track_id, zone_id) key
    are merged into a single active Event.
    """
    rule_id: str
    event_type: EventType
    severity: Severity
    track_id: Optional[int]        # None for aggregate violations (e.g., over-capacity)
    zone_id: str
    zone_name: str
    class_name: Optional[str]
    description: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def debounce_key(self) -> str:
        """Unique key for this violation type. Used to prevent duplicate events."""
        return f"{self.rule_id}:{self.track_id}:{self.zone_id}"


class BaseRule(ABC):
    """
    Abstract base for all event rules.

    Subclasses implement evaluate() to check their specific condition.
    The rule has no side effects — it only returns violations.
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        event_type: EventType,
        severity: Severity,
        zone_id: str,
        enabled: bool = True,
    ):
        self.rule_id = rule_id
        self.name = name
        self.event_type = event_type
        self.severity = severity
        self.zone_id = zone_id
        self.enabled = enabled

    @abstractmethod
    def evaluate(self, context: RuleContext) -> List[RuleViolation]:
        """
        Evaluate the rule against the current frame context.
        Returns a list of violations (usually 0 or 1 per track).
        Must not raise exceptions — handle errors internally and return [].
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.rule_id}, zone={self.zone_id}, enabled={self.enabled})"
