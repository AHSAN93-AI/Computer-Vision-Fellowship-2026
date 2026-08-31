"""
app.events.activity_manager — Per-Person Activity Timeline (§4.14)

Orchestrates all activity recognisers and state machines for each
tracked person.  Maintains:
  • A ``PoseSequenceBuffer`` per person.
  • An ``ActivityStateMachine`` per activity per person.
  • A ``SquatCounter`` per person.
  • An ``ErgonomicMonitor`` per person.
  • An activity timeline (list of completed events).

On each frame, call ``process_person()`` to update everything.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.activities.bending import BendingRecogniser
from app.activities.ergonomic import ErgonomicMonitor
from app.activities.fall import FallRecogniser
from app.activities.hand_raise import HandRaiseRecogniser
from app.activities.sitting import SittingRecogniser
from app.activities.squats import SquatCounter, SquatRecogniser
from app.activities.standing import StandingRecogniser
from app.activities.state_machine import ActivityState, ActivityStateMachine
from app.activities.walking import WalkingRecogniser
from app.activities.waving import WavingRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer

logger = logging.getLogger(__name__)


@dataclass
class ActivityEvent:
    """A completed activity entry for the timeline."""
    activity_type: str
    display_name: str
    start_time: float
    end_time: float
    duration: float
    confidence: float
    person_id: int


@dataclass
class PersonActivityState:
    """All activity-related state for one tracked person."""

    person_id: int
    buffer: PoseSequenceBuffer = field(default_factory=lambda: PoseSequenceBuffer(
        get_settings().sequence_buffer_length
    ))
    state_machines: Dict[str, ActivityStateMachine] = field(default_factory=dict)
    squat_counter: SquatCounter = field(default_factory=SquatCounter)
    ergonomic_monitor: ErgonomicMonitor = field(default_factory=ErgonomicMonitor)
    current_activity: str = "Unknown"
    current_activity_display: str = "Unknown"
    current_confidence: float = 0.0
    previous_activity: str = "Unknown"
    activity_start_time: float = field(default_factory=time.monotonic)
    timeline: List[ActivityEvent] = field(default_factory=list)
    squat_count: int = 0


# Activity priority — higher-priority activities override lower ones
# when multiple fire simultaneously.
_ACTIVITY_PRIORITY = {
    "fall": 100,        # Always top priority
    "squats": 50,       # Specific exercise
    "waving": 40,
    "hand_raised": 35,
    "bending": 30,
    "walking": 20,
    "sitting": 15,
    "standing": 10,
}


class ActivityManager:
    """Central manager that runs all recognisers on each person per frame.

    Usage::

        mgr = ActivityManager()
        # Each frame, for each tracked person:
        result = mgr.process_person(person_id, keypoints, frame_number)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._recognisers: List[ActivityRecogniser] = [
            StandingRecogniser(),
            SittingRecogniser(),
            WalkingRecogniser(),
            HandRaiseRecogniser(),
            FallRecogniser(),
            BendingRecogniser(),
            WavingRecogniser(),
            SquatRecogniser(),
        ]
        self._confirm_frames = settings.activity_confirm_frames
        self._end_frames = settings.activity_end_frames
        self._persons: Dict[int, PersonActivityState] = {}
        logger.info(
            "ActivityManager initialised with %d recognisers",
            len(self._recognisers),
        )

    def _get_or_create(self, person_id: int) -> PersonActivityState:
        """Get or create the activity state for a person."""
        if person_id not in self._persons:
            pas = PersonActivityState(person_id=person_id)
            # Create state machines for each recogniser
            for rec in self._recognisers:
                confirm = self._confirm_frames
                end = self._end_frames
                # Fall needs longer confirmation
                if rec.activity_type == "fall":
                    confirm = get_settings().fall_confirm_frames
                pas.state_machines[rec.activity_type] = ActivityStateMachine(
                    activity_type=rec.activity_type,
                    confirm_frames=confirm,
                    end_frames=end,
                )
            self._persons[person_id] = pas
        return self._persons[person_id]

    def process_person(
        self,
        person_id: int,
        keypoints: PersonKeypoints,
        frame_number: int,
    ) -> PersonActivityState:
        """Run all recognisers and state machines for one person on one frame.

        Returns the updated ``PersonActivityState``.
        """
        pas = self._get_or_create(person_id)

        # 1. Add to sequence buffer
        snap = pas.buffer.add(keypoints, frame_number)

        # 2. Run each recogniser
        candidates: List[ActivityCandidate] = []
        for rec in self._recognisers:
            candidate = rec.evaluate(keypoints, pas.buffer)
            candidates.append(candidate)

            # Update state machine
            sm = pas.state_machines[rec.activity_type]
            sm.update(candidate.is_detected, candidate.confidence)

        # 3. Update squat counter
        avg_knee = pas.buffer.average_knee_angle(last_n=3)
        prev_count = pas.squat_count
        pas.squat_count = pas.squat_counter.update(avg_knee)
        if pas.squat_count > prev_count:
            logger.info("Person #%03d: squat rep #%d", person_id, pas.squat_count)

        # 4. Update ergonomic monitor
        pas.ergonomic_monitor.update(
            snap.torso_angle,
            snap.left_knee_angle,
            snap.right_knee_angle,
        )

        # 5. Determine winning activity (highest priority among active state machines)
        winning_activity = "Unknown"
        winning_display = "Unknown"
        winning_confidence = 0.0

        active_activities: List[Tuple[str, float, int]] = []
        for rec in self._recognisers:
            sm = pas.state_machines[rec.activity_type]
            if sm.is_active:
                priority = _ACTIVITY_PRIORITY.get(rec.activity_type, 0)
                active_activities.append((rec.activity_type, sm.confidence, priority))

        if active_activities:
            # Sort by priority descending
            active_activities.sort(key=lambda x: x[2], reverse=True)
            winning_activity = active_activities[0][0]
            winning_confidence = active_activities[0][1]
            # Get display name
            for rec in self._recognisers:
                if rec.activity_type == winning_activity:
                    winning_display = rec.display_name
                    break

        # 6. Detect activity transitions
        if winning_activity != pas.current_activity:
            old = pas.current_activity
            # Record ended activity in timeline
            if old != "Unknown":
                event = ActivityEvent(
                    activity_type=old,
                    display_name=pas.current_activity_display,
                    start_time=pas.activity_start_time,
                    end_time=time.monotonic(),
                    duration=time.monotonic() - pas.activity_start_time,
                    confidence=pas.current_confidence,
                    person_id=person_id,
                )
                pas.timeline.append(event)
                logger.debug(
                    "Person #%03d: %s ended (%.1fs)",
                    person_id, old, event.duration,
                )

            pas.previous_activity = old
            pas.current_activity = winning_activity
            pas.current_activity_display = winning_display
            pas.current_confidence = winning_confidence
            pas.activity_start_time = time.monotonic()

            if winning_activity != "Unknown":
                logger.info(
                    "Person #%03d: activity -> %s (conf=%.0f%%)",
                    person_id, winning_display, winning_confidence * 100,
                )
        else:
            # Update confidence even if activity hasn't changed
            pas.current_confidence = max(pas.current_confidence, winning_confidence)

        return pas

    def get_person_state(self, person_id: int) -> Optional[PersonActivityState]:
        return self._persons.get(person_id)

    def get_all_states(self) -> Dict[int, PersonActivityState]:
        return dict(self._persons)

    def remove_person(self, person_id: int) -> None:
        """Clean up state for a person who is no longer tracked."""
        if person_id in self._persons:
            del self._persons[person_id]

    def get_fall_state(self, person_id: int) -> Optional[ActivityStateMachine]:
        """Get the fall detection state machine for a person."""
        pas = self._persons.get(person_id)
        if pas:
            return pas.state_machines.get("fall")
        return None
