"""
app.activities.base_activity — Activity Recogniser ABC (§4.8)

Every activity recogniser inherits from ``ActivityRecogniser`` and
implements a single ``evaluate()`` method that examines the person's
current pose snapshot and sequence buffer to decide whether the
activity is happening.

The recogniser does NOT manage state transitions — that is the job
of the ``ActivityStateMachine`` (state_machine.py, §4.9).  The
recogniser only answers: "does this frame look like my activity?"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer


@dataclass
class ActivityCandidate:
    """Result of one recogniser evaluating one frame for one person.

    Attributes
    ----------
    activity_type : str
        Machine-readable activity name (e.g. ``"standing"``).
    display_name : str
        Human-readable label (e.g. ``"Standing"``).
    is_detected : bool
        True if the recogniser believes this activity is occurring *this frame*.
    confidence : float
        Recogniser's confidence 0..1.  For rule-based recognisers this is
        derived from how many sub-conditions are met and by how much.
    rule_explanation : str
        Human-readable summary of why the recogniser fired (or didn't).
        Useful for debugging and the Activity Rule Specification doc.
    """

    activity_type: str
    display_name: str
    is_detected: bool = False
    confidence: float = 0.0
    rule_explanation: str = ""


class ActivityRecogniser(ABC):
    """Abstract base class for activity recognisers.

    Each subclass implements ``evaluate()`` which receives the latest
    keypoints and the person's pose buffer, and returns an
    ``ActivityCandidate``.
    """

    @property
    @abstractmethod
    def activity_type(self) -> str:
        """Machine-readable activity identifier (e.g. ``"standing"``)."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label (e.g. ``"Standing"``)."""
        ...

    @abstractmethod
    def evaluate(
        self,
        keypoints: PersonKeypoints,
        buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        """Assess whether this activity is occurring in the current frame.

        Parameters
        ----------
        keypoints:
            Current frame's keypoints for this person.
        buffer:
            Rolling sequence buffer (already has the current frame appended).

        Returns
        -------
        ``ActivityCandidate`` with ``is_detected`` and ``confidence``.
        """
        ...
