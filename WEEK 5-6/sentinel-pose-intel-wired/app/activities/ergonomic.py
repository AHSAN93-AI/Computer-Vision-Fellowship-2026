"""
app.activities.ergonomic — Ergonomic Posture Monitoring (§4.13)

Monitors two posture-risk scenarios:

1. **Prolonged bending**: torso angle > 45° for longer than
   ``ergo_bend_warn_seconds`` (default 15 s).
2. **Prolonged crouching**: both knee angles < ``ergo_crouch_max_knee_angle``
   (default 100°) for longer than ``ergo_crouch_warn_seconds`` (default 15 s).

When either threshold is exceeded, the monitor flags a posture risk
so the alert engine can generate a warning.

.. warning::

   This is a prototype posture monitoring tool.  It is **NOT** a certified
   ergonomic assessment and must **NOT** be used as medical or occupational
   health advice.  See ``docs/posture_disclaimer.md`` for the full disclaimer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PostureRiskState:
    """Tracks duration of risky postures for one person."""

    # Bending
    bend_start_time: Optional[float] = None
    bend_warned: bool = False
    bend_duration: float = 0.0

    # Crouching
    crouch_start_time: Optional[float] = None
    crouch_warned: bool = False
    crouch_duration: float = 0.0


class ErgonomicMonitor:
    """Per-person ergonomic posture monitor.

    Call ``update()`` each frame with the current torso angle and
    knee angles.  Query ``is_bend_risk`` / ``is_crouch_risk`` to
    check if a warning should fire.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._bend_angle_threshold = settings.bending_min_torso_angle  # 45°
        self._bend_warn_seconds = settings.ergo_bend_warn_seconds      # 15 s
        self._crouch_knee_threshold = settings.ergo_crouch_max_knee_angle  # 100°
        self._crouch_warn_seconds = settings.ergo_crouch_warn_seconds     # 15 s
        self._state = PostureRiskState()

    @property
    def state(self) -> PostureRiskState:
        return self._state

    @property
    def is_bend_risk(self) -> bool:
        return self._state.bend_warned

    @property
    def is_crouch_risk(self) -> bool:
        return self._state.crouch_warned

    @property
    def bend_duration(self) -> float:
        return self._state.bend_duration

    @property
    def crouch_duration(self) -> float:
        return self._state.crouch_duration

    def update(
        self,
        torso_angle: Optional[float],
        left_knee_angle: Optional[float],
        right_knee_angle: Optional[float],
    ) -> None:
        """Update the posture risk state with this frame's data.

        Parameters
        ----------
        torso_angle:
            Degrees from vertical (0 = upright).
        left_knee_angle, right_knee_angle:
            Knee angles in degrees.  None if not visible.
        """
        now = time.monotonic()

        # ── Bending check ──────────────────────────────
        if torso_angle is not None and torso_angle > self._bend_angle_threshold:
            if self._state.bend_start_time is None:
                self._state.bend_start_time = now
            self._state.bend_duration = now - self._state.bend_start_time
            if self._state.bend_duration >= self._bend_warn_seconds and not self._state.bend_warned:
                self._state.bend_warned = True
                logger.warning(
                    "Ergonomic risk: prolonged bending (%.1fs, torso=%.1f°)",
                    self._state.bend_duration, torso_angle,
                )
        else:
            if self._state.bend_start_time is not None:
                self._state.bend_duration = 0.0
                self._state.bend_start_time = None
                self._state.bend_warned = False

        # ── Crouching check ────────────────────────────
        knees_bent = True
        if left_knee_angle is not None and left_knee_angle > self._crouch_knee_threshold:
            knees_bent = False
        if right_knee_angle is not None and right_knee_angle > self._crouch_knee_threshold:
            knees_bent = False
        # If both knees are invisible, don't trigger crouching
        if left_knee_angle is None and right_knee_angle is None:
            knees_bent = False

        if knees_bent:
            if self._state.crouch_start_time is None:
                self._state.crouch_start_time = now
            self._state.crouch_duration = now - self._state.crouch_start_time
            if self._state.crouch_duration >= self._crouch_warn_seconds and not self._state.crouch_warned:
                self._state.crouch_warned = True
                logger.warning(
                    "Ergonomic risk: prolonged crouching (%.1fs)",
                    self._state.crouch_duration,
                )
        else:
            if self._state.crouch_start_time is not None:
                self._state.crouch_duration = 0.0
                self._state.crouch_start_time = None
                self._state.crouch_warned = False

    def reset(self) -> None:
        """Reset all posture timers."""
        self._state = PostureRiskState()
