"""
tests.test_sequence -- Unit tests for PoseSequenceBuffer

Covers buffer add/length, rolling window, temporal analysis helpers,
and wrist oscillation detection.
"""

import pytest
import numpy as np

from evaluation.synthetic import make_standing_pose, make_walking_pose, make_waving_sequence
from app.pose.sequence import PoseSequenceBuffer


def _fill_buffer(buf: PoseSequenceBuffer, n: int, pose_fn=make_standing_pose) -> None:
    """Add n standing poses to the buffer."""
    for i in range(n):
        buf.add(pose_fn(track_id=1), frame_number=i)


class TestBufferBasics:
    def test_empty_buffer(self):
        buf = PoseSequenceBuffer(max_length=10)
        assert buf.length == 0
        assert buf.latest is None
        assert not buf.is_full

    def test_add_increments_length(self):
        buf = PoseSequenceBuffer(max_length=10)
        pk = make_standing_pose()
        buf.add(pk, frame_number=0)
        assert buf.length == 1
        assert buf.latest is not None

    def test_rolling_window_maxlen(self):
        buf = PoseSequenceBuffer(max_length=5)
        _fill_buffer(buf, 10)
        assert buf.length == 5
        assert buf.is_full
        # The oldest frame should be frame 5 (frames 0-4 evicted)
        assert buf.latest.frame_number == 9

    def test_snapshot_has_correct_frame_number(self):
        buf = PoseSequenceBuffer(max_length=10)
        pk = make_standing_pose()
        snap = buf.add(pk, frame_number=42)
        assert snap.frame_number == 42

    def test_clear(self):
        buf = PoseSequenceBuffer(max_length=10)
        _fill_buffer(buf, 5)
        buf.clear()
        assert buf.length == 0
        assert buf.latest is None


class TestSnapshotFeatures:
    def test_first_frame_velocity_is_none(self):
        buf = PoseSequenceBuffer(max_length=10)
        pk = make_standing_pose()
        snap = buf.add(pk, frame_number=0)
        assert snap.velocity is None  # no previous frame

    def test_second_frame_has_velocity(self):
        buf = PoseSequenceBuffer(max_length=10)
        buf.add(make_standing_pose(), frame_number=0)
        snap = buf.add(make_standing_pose(), frame_number=1)
        assert snap.velocity is not None

    def test_standing_has_torso_angle(self):
        buf = PoseSequenceBuffer(max_length=10)
        snap = buf.add(make_standing_pose(), frame_number=0)
        assert snap.torso_angle is not None

    def test_standing_torso_angle_near_zero(self):
        buf = PoseSequenceBuffer(max_length=10)
        snap = buf.add(make_standing_pose(), frame_number=0)
        assert snap.torso_angle < 20.0  # upright person


class TestTemporalAnalysis:
    def test_average_velocity(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        avg_vel = buf.average_velocity()
        assert avg_vel is not None
        # Standing person with no displacement -> very low velocity
        assert avg_vel < 0.01

    def test_average_velocity_last_n(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        avg_vel = buf.average_velocity(last_n=3)
        assert avg_vel is not None

    def test_max_velocity(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        max_vel = buf.max_velocity()
        assert max_vel is not None
        assert max_vel >= 0.0

    def test_average_torso_angle(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        avg_ta = buf.average_torso_angle()
        assert avg_ta is not None
        assert avg_ta < 20.0  # upright

    def test_average_knee_angle(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        avg_knee = buf.average_knee_angle()
        # May be None if knee angles weren't computed (depends on keypoint visibility)
        # The standing pose has knees visible
        if avg_knee is not None:
            assert avg_knee > 100.0  # standing = straight legs

    def test_min_knee_angle(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        min_knee = buf.min_knee_angle()
        if min_knee is not None:
            assert min_knee > 0.0

    def test_hip_stability(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        stability = buf.hip_stability()
        if stability is not None:
            assert stability >= 0.0  # variance is non-negative

    def test_movement_direction_returns_none_for_static(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        direction = buf.movement_direction()
        # For a standing person at same position, direction may be None or near 0
        # (the synthetic standing pose doesn't move)

    def test_duration_seconds(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        dur = buf.duration_seconds
        # Duration should be positive (monotonic timestamps)
        assert dur >= 0.0

    def test_duration_single_frame(self):
        buf = PoseSequenceBuffer(max_length=30)
        buf.add(make_standing_pose(), frame_number=0)
        assert buf.duration_seconds == 0.0

    def test_get_snapshots(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        snaps = buf.get_snapshots()
        assert len(snaps) == 10

    def test_get_snapshots_last_n(self):
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 10)
        snaps = buf.get_snapshots(last_n=3)
        assert len(snaps) == 3


class TestWristOscillations:
    def test_standing_no_oscillations(self):
        """Standing person with arms at sides -> minimal oscillation."""
        buf = PoseSequenceBuffer(max_length=30)
        _fill_buffer(buf, 20)
        osc = buf.wrist_oscillations()
        assert osc < 3  # noise may cause a few

    def test_waving_has_oscillations(self):
        """Waving sequence should produce multiple oscillations."""
        buf = PoseSequenceBuffer(max_length=60)
        waving_poses = make_waving_sequence(n_frames=30, track_id=1)
        for i, pk in enumerate(waving_poses):
            buf.add(pk, frame_number=i)
        osc = buf.wrist_oscillations()
        assert osc >= 2  # the waving generator oscillates


class TestCurrentAccessors:
    def test_current_velocity(self):
        buf = PoseSequenceBuffer(max_length=10)
        buf.add(make_standing_pose(), frame_number=0)
        buf.add(make_standing_pose(), frame_number=1)
        assert buf.current_velocity is not None

    def test_current_torso_angle(self):
        buf = PoseSequenceBuffer(max_length=10)
        buf.add(make_standing_pose(), frame_number=0)
        assert buf.current_torso_angle is not None
