"""
app.activities.squats — Squat Repetition Counter (§4.12)

Counts squat reps using a state machine:

    Standing (knee > up_angle)
         │
         ▼  (knee angle decreasing)
    Descending
         │
         ▼  (knee < down_angle)
    Down
         │
         ▼  (knee increasing past hysteresis_angle)
    Ascending
         │
         ▼  (knee > up_angle)  →  count += 1
    Standing

**Double-count prevention**:
  • Must complete the full cycle (reach Down state) before counting.
  • Hysteresis: separate thresholds for entering Down (knee < 90°) and
    leaving Down (knee > 110°) prevents bouncing at the threshold.
  • Partial reps (never reaching Down) are not counted.

**Required keypoints**: at least one hip-knee-ankle triplet.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer

logger = logging.getLogger(__name__)


class SquatPhase(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    DOWN = "down"
    ASCENDING = "ascending"


class SquatCounter:
    """Per-person squat repetition counter."""

    def __init__(self) -> None:
        settings = get_settings()
        self._down_angle = settings.squat_down_knee_angle          # 90°
        self._up_angle = settings.squat_up_knee_angle              # 160°
        self._hysteresis_angle = settings.squat_hysteresis_angle   # 110°
        self._phase = SquatPhase.STANDING
        self._count = 0
        self._last_knee_angle: Optional[float] = None

    @property
    def count(self) -> int:
        return self._count

    @property
    def phase(self) -> SquatPhase:
        return self._phase

    def update(self, knee_angle: Optional[float]) -> int:
        """Feed one frame's knee angle and return the current rep count.

        Parameters
        ----------
        knee_angle:
            Average of left/right knee angles (degrees), or None if
            not visible.

        Returns
        -------
        Current total rep count.
        """
        if knee_angle is None:
            return self._count

        self._last_knee_angle = knee_angle

        if self._phase == SquatPhase.STANDING:
            if knee_angle < self._up_angle:
                self._phase = SquatPhase.DESCENDING
                logger.debug("Squat: STANDING -> DESCENDING (knee=%.1f deg)", knee_angle)

        elif self._phase == SquatPhase.DESCENDING:
            if knee_angle <= self._down_angle:
                self._phase = SquatPhase.DOWN
                logger.debug("Squat: DESCENDING -> DOWN (knee=%.1f deg)", knee_angle)
            elif knee_angle >= self._up_angle:
                # Aborted descent — back to standing without counting
                self._phase = SquatPhase.STANDING
                logger.debug("Squat: DESCENDING -> STANDING (aborted, knee=%.1f deg)", knee_angle)

        elif self._phase == SquatPhase.DOWN:
            if knee_angle > self._hysteresis_angle:
                self._phase = SquatPhase.ASCENDING
                logger.debug("Squat: DOWN -> ASCENDING (knee=%.1f deg)", knee_angle)

        elif self._phase == SquatPhase.ASCENDING:
            if knee_angle >= self._up_angle:
                self._count += 1
                self._phase = SquatPhase.STANDING
                logger.info("Squat rep #%d completed (knee=%.1f deg)", self._count, knee_angle)
            elif knee_angle <= self._down_angle:
                # Went back down without fully standing
                self._phase = SquatPhase.DOWN
                logger.debug("Squat: ASCENDING -> DOWN (re-descended, knee=%.1f deg)", knee_angle)

        return self._count

    def reset(self) -> None:
        """Reset the counter."""
        self._count = 0
        self._phase = SquatPhase.STANDING


class SquatRecogniser(ActivityRecogniser):
    """Recogniser wrapper — detects "currently doing squats" for the state machine.

    The ``SquatCounter`` itself is managed externally (one per person).
    This recogniser detects whether the person is in an active squat cycle.
    """

    @property
    def activity_type(self) -> str:
        return "squats"

    @property
    def display_name(self) -> str:
        return "Squats"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        avg_knee = buffer.average_knee_angle(last_n=3)
        settings = get_settings()
        reasons = []

        if avg_knee is None:
            return ActivityCandidate(
                activity_type=self.activity_type,
                display_name=self.display_name,
                is_detected=False,
                confidence=0.0,
                rule_explanation="knee_angle=N/A",
            )

        # Detect active squatting: knee is in the squat range
        in_squat_range = avg_knee < settings.squat_up_knee_angle
        # Also check for recent significant knee motion
        min_knee = buffer.min_knee_angle(last_n=10)
        has_deep_bend = min_knee is not None and min_knee < settings.squat_hysteresis_angle

        detected = in_squat_range and has_deep_bend
        confidence = 0.8 if detected else 0.2

        reasons.append(f"knee={avg_knee:.1f}° (range<{settings.squat_up_knee_angle}°: {'yes' if in_squat_range else 'no'})")
        if min_knee is not None:
            reasons.append(f"min_knee_10f={min_knee:.1f}° (deep_bend: {'yes' if has_deep_bend else 'no'})")

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
