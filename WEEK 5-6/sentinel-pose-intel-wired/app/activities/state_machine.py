"""
app.activities.state_machine — Temporal Activity State Machine (§4.9)

Prevents per-frame label flipping by requiring an activity to persist
for a configurable number of frames before it is confirmed, and to
be absent for another configurable number of frames before it ends.

State diagram::

    Idle ──(detected)──▶ Candidate ──(persist ≥ confirm_frames)──▶ Confirmed
                              │                                        │
                              ▼                                        ▼
                        (not detected                            Active
                         for end_frames)                            │
                              │                              (not detected
                              ▼                               for end_frames)
                            Idle                                   │
                                                                   ▼
                                                                 Ended ──▶ Idle

``Confirmed`` transitions to ``Active`` immediately (same frame).
``Active`` stays active as long as the recogniser keeps detecting it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ActivityState(Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    ENDED = "ended"


@dataclass
class ActivityStateMachine:
    """Per-activity, per-person state tracker.

    Parameters
    ----------
    activity_type:
        Machine-readable activity name.
    confirm_frames:
        Frames the activity must be continuously detected to move
        Candidate → Confirmed.
    end_frames:
        Frames the activity must be continuously absent to move
        Active → Ended.
    """

    activity_type: str
    confirm_frames: int = 5
    end_frames: int = 8
    state: ActivityState = ActivityState.IDLE
    _detect_count: int = 0        # consecutive frames detected in Candidate
    _absent_count: int = 0        # consecutive frames absent in Active
    _start_time: float = 0.0      # monotonic time of Candidate start
    _active_start_time: float = 0.0  # monotonic time of Active start
    _end_time: float = 0.0        # monotonic time of Ended
    _last_confidence: float = 0.0

    def update(self, is_detected: bool, confidence: float = 0.0) -> ActivityState:
        """Feed one frame's detection result and advance the state machine.

        Parameters
        ----------
        is_detected:
            Whether the recogniser says this activity is occurring now.
        confidence:
            Recogniser's confidence for this frame.

        Returns
        -------
        The new state after this frame.
        """
        self._last_confidence = confidence

        if self.state == ActivityState.IDLE:
            if is_detected:
                self.state = ActivityState.CANDIDATE
                self._detect_count = 1
                self._absent_count = 0
                self._start_time = time.monotonic()

        elif self.state == ActivityState.CANDIDATE:
            if is_detected:
                self._detect_count += 1
                self._absent_count = 0
                if self._detect_count >= self.confirm_frames:
                    self.state = ActivityState.ACTIVE
                    self._active_start_time = time.monotonic()
                    logger.debug(
                        "%s confirmed after %d frames",
                        self.activity_type, self._detect_count,
                    )
            else:
                self._absent_count += 1
                # Allow brief drops (up to 2 frames) while in Candidate
                if self._absent_count >= 3:
                    self.state = ActivityState.IDLE
                    self._detect_count = 0
                    self._absent_count = 0

        elif self.state in (ActivityState.CONFIRMED, ActivityState.ACTIVE):
            if is_detected:
                self._absent_count = 0
            else:
                self._absent_count += 1
                if self._absent_count >= self.end_frames:
                    self.state = ActivityState.ENDED
                    self._end_time = time.monotonic()
                    logger.debug(
                        "%s ended after %d absent frames",
                        self.activity_type, self._absent_count,
                    )

        elif self.state == ActivityState.ENDED:
            # Auto-transition to Idle so the activity can be re-detected
            self.state = ActivityState.IDLE
            self._detect_count = 0
            self._absent_count = 0
            # Immediately check if a new detection starts
            if is_detected:
                self.state = ActivityState.CANDIDATE
                self._detect_count = 1
                self._start_time = time.monotonic()

        return self.state

    @property
    def is_active(self) -> bool:
        """True if the activity is currently confirmed/active."""
        return self.state in (ActivityState.CONFIRMED, ActivityState.ACTIVE)

    @property
    def is_candidate_or_active(self) -> bool:
        return self.state in (
            ActivityState.CANDIDATE, ActivityState.CONFIRMED, ActivityState.ACTIVE,
        )

    @property
    def active_duration(self) -> float:
        """Seconds since the activity became active (0 if not active)."""
        if self.state in (ActivityState.ACTIVE, ActivityState.CONFIRMED):
            return time.monotonic() - self._active_start_time
        return 0.0

    @property
    def total_duration(self) -> float:
        """Seconds from first candidate detection to now/end."""
        if self.state == ActivityState.IDLE:
            return 0.0
        end = self._end_time if self.state == ActivityState.ENDED else time.monotonic()
        return end - self._start_time

    @property
    def confidence(self) -> float:
        return self._last_confidence

    def reset(self) -> None:
        """Force reset to Idle."""
        self.state = ActivityState.IDLE
        self._detect_count = 0
        self._absent_count = 0
