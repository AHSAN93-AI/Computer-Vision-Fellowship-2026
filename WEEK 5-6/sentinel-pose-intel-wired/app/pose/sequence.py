"""
app.pose.sequence — Rolling Pose Sequence Buffer (§4.5)

Maintains a fixed-size sliding window of pose snapshots per person.
Each snapshot stores the normalised keypoints, raw keypoints, computed
angles, hip position, and timestamp so downstream activity recognisers
can analyse temporal patterns without re-computing features.

**Buffer size — 30 frames (rationale)**:
  • At 30 FPS this covers 1 second — long enough for a complete walking
    gait cycle (~0.5–1 s) and for fall detection (which happens in < 1 s).
  • Memory cost: ~50 KB per person (30 × 17 × 2 floats + metadata).
  • Longer buffers (60 frames) would help slow activities but increase
    latency for fast ones; 30 is a practical balance.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from app.pose.keypoints import PersonKeypoints
from app.pose.normalization import compute_velocity_from_raw, normalize_keypoints
from app.pose import angles as angle_mod

logger = logging.getLogger(__name__)


@dataclass
class PoseSnapshot:
    """One frame's worth of pose data for a single person."""

    timestamp: float                                   # monotonic time
    frame_number: int
    raw_keypoints: PersonKeypoints                     # pixel-space
    normalised: Optional[np.ndarray]                   # (17, 2) normalised coords
    mid_hip: Optional[Tuple[float, float]]             # raw pixel mid-hip
    mid_shoulder: Optional[Tuple[float, float]]        # raw pixel mid-shoulder
    torso_scale: Optional[float]                       # torso length (pixels)
    velocity: Optional[float]                          # normalised hip velocity
    torso_angle: Optional[float]                       # degrees from vertical
    left_knee_angle: Optional[float]
    right_knee_angle: Optional[float]
    left_elbow_angle: Optional[float]
    right_elbow_angle: Optional[float]
    hip_angle: Optional[float]
    bbox_aspect_ratio: Optional[float]
    head_to_hip_vert: Optional[float]


class PoseSequenceBuffer:
    """Per-person rolling buffer of ``PoseSnapshot`` objects.

    Usage::

        buf = PoseSequenceBuffer(max_length=30)
        buf.add(person_keypoints, frame_number)
        v = buf.current_velocity
        angles = buf.current_angles
    """

    def __init__(self, max_length: int = 30) -> None:
        self._max_length = max_length
        self._buffer: Deque[PoseSnapshot] = deque(maxlen=max_length)

    def add(self, pk: PersonKeypoints, frame_number: int) -> PoseSnapshot:
        """Compute features and append a new snapshot to the buffer.

        Returns the newly created ``PoseSnapshot``.
        """
        now = time.monotonic()

        # ── Normalisation ──────────────────────────────
        normalised = normalize_keypoints(pk)
        mid_hip = pk.mid_hip
        mid_shoulder = pk.mid_shoulder
        torso_scale = angle_mod.torso_length(pk)

        # ── Velocity (hip displacement vs previous frame) ──
        velocity: Optional[float] = None
        if len(self._buffer) > 0:
            prev = self._buffer[-1]
            velocity = compute_velocity_from_raw(prev.mid_hip, mid_hip, torso_scale)

        # ── Joint angles ──────────────────────────────
        t_angle = angle_mod.torso_angle(pk)
        lk = angle_mod.left_knee_angle(pk)
        rk = angle_mod.right_knee_angle(pk)
        le = angle_mod.left_elbow_angle(pk)
        re = angle_mod.right_elbow_angle(pk)
        ha = angle_mod.hip_angle(pk)
        bar = angle_mod.bbox_aspect_ratio(pk)
        hthv = angle_mod.head_to_hip_vertical(pk)

        snap = PoseSnapshot(
            timestamp=now,
            frame_number=frame_number,
            raw_keypoints=pk,
            normalised=normalised,
            mid_hip=mid_hip,
            mid_shoulder=mid_shoulder,
            torso_scale=torso_scale,
            velocity=velocity,
            torso_angle=t_angle,
            left_knee_angle=lk,
            right_knee_angle=rk,
            left_elbow_angle=le,
            right_elbow_angle=re,
            hip_angle=ha,
            bbox_aspect_ratio=bar,
            head_to_hip_vert=hthv,
        )
        self._buffer.append(snap)
        return snap

    # ── Current-frame accessors ────────────────────────

    @property
    def latest(self) -> Optional[PoseSnapshot]:
        return self._buffer[-1] if self._buffer else None

    @property
    def current_velocity(self) -> Optional[float]:
        s = self.latest
        return s.velocity if s else None

    @property
    def current_torso_angle(self) -> Optional[float]:
        s = self.latest
        return s.torso_angle if s else None

    # ── Temporal analysis helpers ──────────────────────

    @property
    def length(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._max_length

    @property
    def duration_seconds(self) -> float:
        """Time span covered by the buffer in seconds."""
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp

    def average_velocity(self, last_n: Optional[int] = None) -> Optional[float]:
        """Mean hip velocity over the last ``last_n`` snapshots (or entire buffer)."""
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]
        vals = [s.velocity for s in slc if s.velocity is not None]
        return sum(vals) / len(vals) if vals else None

    def max_velocity(self, last_n: Optional[int] = None) -> Optional[float]:
        """Peak hip velocity over the last ``last_n`` snapshots."""
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]
        vals = [s.velocity for s in slc if s.velocity is not None]
        return max(vals) if vals else None

    def hip_stability(self, last_n: Optional[int] = None) -> Optional[float]:
        """Variance of hip position over recent frames (lower = more stable).

        Uses normalised velocity values since they are scale-invariant.
        """
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]
        vals = [s.velocity for s in slc if s.velocity is not None]
        if len(vals) < 2:
            return None
        arr = np.array(vals)
        return float(np.var(arr))

    def movement_direction(self, last_n: int = 5) -> Optional[float]:
        """Dominant movement direction in degrees (0=right, 90=down, etc.).

        Computed from the overall hip displacement over the last ``last_n`` frames.
        Returns None if hip data is insufficient.
        """
        slc = list(self._buffer)[-last_n:]
        hips = [s.mid_hip for s in slc if s.mid_hip is not None]
        if len(hips) < 2:
            return None
        dx = hips[-1][0] - hips[0][0]
        dy = hips[-1][1] - hips[0][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return math.degrees(math.atan2(dy, dx)) % 360

    def average_torso_angle(self, last_n: Optional[int] = None) -> Optional[float]:
        """Mean torso angle over recent frames."""
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]
        vals = [s.torso_angle for s in slc if s.torso_angle is not None]
        return sum(vals) / len(vals) if vals else None

    def average_knee_angle(self, last_n: Optional[int] = None) -> Optional[float]:
        """Mean of available left/right knee angles over recent frames."""
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]
        vals = []
        for s in slc:
            if s.left_knee_angle is not None:
                vals.append(s.left_knee_angle)
            if s.right_knee_angle is not None:
                vals.append(s.right_knee_angle)
        return sum(vals) / len(vals) if vals else None

    def min_knee_angle(self, last_n: Optional[int] = None) -> Optional[float]:
        """Minimum knee angle (either side) over recent frames."""
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]
        vals = []
        for s in slc:
            if s.left_knee_angle is not None:
                vals.append(s.left_knee_angle)
            if s.right_knee_angle is not None:
                vals.append(s.right_knee_angle)
        return min(vals) if vals else None

    def wrist_oscillations(self, last_n: Optional[int] = None) -> int:
        """Count lateral wrist direction changes (for waving detection).

        Examines left and right wrist x-positions in normalised coords
        and counts how many times the x-direction reverses.
        """
        slc = list(self._buffer)
        if last_n is not None:
            slc = slc[-last_n:]

        max_osc = 0
        for wrist_idx in [9, 10]:  # LEFT_WRIST, RIGHT_WRIST
            xs = []
            for s in slc:
                if s.normalised is not None and not np.isnan(s.normalised[wrist_idx, 0]):
                    xs.append(float(s.normalised[wrist_idx, 0]))
            if len(xs) < 3:
                continue

            # Count direction changes. A naive adjacent-pair sign check
            # (d_prev * d_curr < 0) breaks whenever two consecutive samples
            # are equal (d == 0) -- e.g. a wrist that briefly holds still at
            # the top/bottom of a wave, or samples that land exactly on a
            # symmetric point of the motion. Track the last *non-zero*
            # direction instead so flat frames don't mask a real reversal.
            changes = 0
            last_dir = 0
            for i in range(1, len(xs)):
                delta = xs[i] - xs[i - 1]
                if delta == 0:
                    continue
                direction = 1 if delta > 0 else -1
                if last_dir != 0 and direction != last_dir:
                    changes += 1
                last_dir = direction
            max_osc = max(max_osc, changes)

        return max_osc

    def get_snapshots(self, last_n: Optional[int] = None) -> List[PoseSnapshot]:
        """Return recent snapshots as a list."""
        slc = list(self._buffer)
        if last_n is not None:
            return slc[-last_n:]
        return slc

    def clear(self) -> None:
        """Empty the buffer."""
        self._buffer.clear()
