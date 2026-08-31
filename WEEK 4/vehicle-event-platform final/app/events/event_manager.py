"""
app/events/event_manager.py — Event lifecycle management and debouncing.

Manages the full event lifecycle: Detected → Active → Acknowledged → Resolved.
Key feature: debounces continuous rule violations so that, e.g.,
60 seconds of parking violation at 30fps = 1 event, not 1800.

Uses a debounce_key (rule_id:track_id:zone_id) to merge violations.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.rules.base_rule import EventType, RuleViolation, Severity

logger = logging.getLogger(__name__)


class EventStatus(str, Enum):
    DETECTED = "DETECTED"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class Event:
    """
    A deduplicated event with full lifecycle state.

    Created from a RuleViolation. Subsequent violations with the same
    debounce_key update this event rather than creating new ones.
    """
    event_id: str
    rule_id: str
    event_type: EventType
    severity: Severity
    status: EventStatus
    track_id: Optional[int]
    zone_id: str
    zone_name: str
    class_name: Optional[str]
    description: str
    source_id: str
    confidence: float
    created_at: float
    updated_at: float
    resolved_at: Optional[float] = None
    duration_seconds: float = 0.0
    evidence_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal: tracks how many violations have been merged into this event
    _violation_count: int = 1

    @property
    def debounce_key(self) -> str:
        return f"{self.rule_id}:{self.track_id}:{self.zone_id}"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "rule_id": self.rule_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "class_name": self.class_name,
            "description": self.description,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "evidence_path": self.evidence_path,
            "metadata": self.metadata,
        }


class EventManager:
    """
    Manages the event lifecycle with debouncing.

    Core behavior:
    - First violation with a given debounce_key → new Event (DETECTED)
    - Subsequent violations with same key → update existing Event
    - Event transitions: DETECTED → ACTIVE after min_active_duration
    - Violations stop arriving → Event auto-resolves after cooldown
    - Manual acknowledge/resolve via API

    Callbacks:
    - on_event_created: called when a new event is created
    - on_event_updated: called when an existing event is updated
    - on_event_resolved: called when an event is resolved
    """

    def __init__(
        self,
        min_active_duration_seconds: float = 5.0,
        default_cooldown_seconds: float = 30.0,
        max_active_events: int = 500,
        on_event_created: Optional[Callable] = None,
        on_event_updated: Optional[Callable] = None,
        on_event_resolved: Optional[Callable] = None,
    ):
        self._min_active_duration = min_active_duration_seconds
        self._default_cooldown = default_cooldown_seconds
        self._max_active_events = max_active_events

        self._on_event_created = on_event_created
        self._on_event_updated = on_event_updated
        self._on_event_resolved = on_event_resolved

        # debounce_key → Event
        self._active_events: Dict[str, Event] = {}
        # debounce_key → last_violation_timestamp
        self._last_violation_time: Dict[str, float] = {}
        # debounce_key → cooldown_seconds (rule-specific)
        self._cooldowns: Dict[str, float] = {}

        # Resolved events kept for short history
        self._resolved_events: List[Event] = []

    def process_violations(
        self,
        violations: List[RuleViolation],
        source_id: str = "",
    ) -> List[Event]:
        """
        Process a batch of violations from one frame.

        Returns list of events that were created or updated.
        """
        affected: List[Event] = []
        now = time.time()

        for v in violations:
            key = v.debounce_key

            if key in self._active_events:
                # Update existing event
                event = self._active_events[key]
                event.updated_at = now
                event.duration_seconds = now - event.created_at
                event.description = v.description
                event.confidence = max(event.confidence, v.confidence)
                event._violation_count += 1

                # Merge metadata
                if v.metadata:
                    event.metadata.update(v.metadata)

                # Promote DETECTED → ACTIVE after min_active_duration
                if event.status == EventStatus.DETECTED:
                    if event.duration_seconds >= self._min_active_duration:
                        event.status = EventStatus.ACTIVE
                        logger.debug(f"Event promoted to ACTIVE: {event.event_id}")

                self._last_violation_time[key] = now

                if self._on_event_updated:
                    try:
                        self._on_event_updated(event)
                    except Exception as e:
                        logger.debug(f"on_event_updated callback error: {e}")

                affected.append(event)
            else:
                # Create new event
                if len(self._active_events) >= self._max_active_events:
                    logger.warning(
                        f"Max active events ({self._max_active_events}) reached — "
                        f"dropping violation {key}"
                    )
                    continue

                event = Event(
                    event_id=str(uuid.uuid4()),
                    rule_id=v.rule_id,
                    event_type=v.event_type,
                    severity=v.severity,
                    status=EventStatus.DETECTED,
                    track_id=v.track_id,
                    zone_id=v.zone_id,
                    zone_name=v.zone_name,
                    class_name=v.class_name,
                    description=v.description,
                    source_id=source_id,
                    confidence=v.confidence,
                    created_at=now,
                    updated_at=now,
                    metadata=dict(v.metadata) if v.metadata else {},
                )

                self._active_events[key] = event
                self._last_violation_time[key] = now

                if self._on_event_created:
                    try:
                        self._on_event_created(event)
                    except Exception as e:
                        logger.debug(f"on_event_created callback error: {e}")

                affected.append(event)
                logger.info(
                    f"New event: {event.event_id} type={event.event_type.value} "
                    f"zone={event.zone_id} track={event.track_id}"
                )

        return affected

    def check_auto_resolve(self, cooldown_override: Optional[float] = None) -> List[Event]:
        """
        Check all active events for auto-resolution.

        An event is auto-resolved if no new violation has been received
        for longer than its cooldown period.

        Returns list of resolved events.
        """
        now = time.time()
        cooldown = cooldown_override or self._default_cooldown
        resolved: List[Event] = []

        keys_to_remove: List[str] = []
        for key, event in self._active_events.items():
            last_time = self._last_violation_time.get(key, event.created_at)
            if (now - last_time) >= cooldown:
                event.status = EventStatus.RESOLVED
                event.resolved_at = now
                event.updated_at = now
                event.duration_seconds = now - event.created_at
                keys_to_remove.append(key)
                resolved.append(event)

                if self._on_event_resolved:
                    try:
                        self._on_event_resolved(event)
                    except Exception as e:
                        logger.debug(f"on_event_resolved callback error: {e}")

                logger.info(
                    f"Event auto-resolved: {event.event_id} "
                    f"duration={event.duration_seconds:.1f}s"
                )

        for key in keys_to_remove:
            evt = self._active_events.pop(key)
            self._last_violation_time.pop(key, None)
            self._cooldowns.pop(key, None)
            self._resolved_events.append(evt)

        # Trim resolved history
        if len(self._resolved_events) > 1000:
            self._resolved_events = self._resolved_events[-500:]

        return resolved

    def acknowledge_event(self, event_id: str) -> Optional[Event]:
        """Manually acknowledge an event."""
        for event in self._active_events.values():
            if event.event_id == event_id:
                if event.status in (EventStatus.DETECTED, EventStatus.ACTIVE):
                    event.status = EventStatus.ACKNOWLEDGED
                    event.updated_at = time.time()
                    logger.info(f"Event acknowledged: {event_id}")
                    return event
        return None

    def resolve_event(self, event_id: str) -> Optional[Event]:
        """Manually resolve an event."""
        keys_to_remove = []
        resolved_event = None

        for key, event in self._active_events.items():
            if event.event_id == event_id:
                event.status = EventStatus.RESOLVED
                event.resolved_at = time.time()
                event.updated_at = event.resolved_at
                event.duration_seconds = event.resolved_at - event.created_at
                keys_to_remove.append(key)
                resolved_event = event

                if self._on_event_resolved:
                    try:
                        self._on_event_resolved(event)
                    except Exception as e:
                        logger.debug(f"on_event_resolved callback error: {e}")

                logger.info(f"Event manually resolved: {event_id}")
                break

        for key in keys_to_remove:
            evt = self._active_events.pop(key)
            self._last_violation_time.pop(key, None)
            self._resolved_events.append(evt)

        return resolved_event

    def get_active_events(self) -> List[Event]:
        return list(self._active_events.values())

    def get_event(self, event_id: str) -> Optional[Event]:
        for event in self._active_events.values():
            if event.event_id == event_id:
                return event
        for event in self._resolved_events:
            if event.event_id == event_id:
                return event
        return None

    @property
    def active_count(self) -> int:
        return len(self._active_events)

    def set_evidence_path(self, event_id: str, path: str) -> None:
        """Set the evidence path for an event after evidence capture."""
        for event in self._active_events.values():
            if event.event_id == event_id:
                event.evidence_path = path
                return
