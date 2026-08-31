"""
tests.test_normalization -- Unit tests for app.pose.normalization

Covers hip-centred translation, torso-length scaling, fallback to
shoulder width, None returns, and velocity computation.
"""

import math
import pytest
import numpy as np

from app.pose.keypoints import (
    Keypoint, PersonKeypoints, KEYPOINT_NAMES, NUM_KEYPOINTS,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
)
from app.pose.normalization import normalize_keypoints, compute_velocity_from_raw


def _make_kp(x: float, y: float, idx: int, conf: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=conf, index=idx, name=KEYPOINT_NAMES[idx])


def _make_nan_kp(idx: int) -> Keypoint:
    return Keypoint(x=float("nan"), y=float("nan"), confidence=0.0, index=idx, name=KEYPOINT_NAMES[idx])


def _make_pk_with_body(
    hip_x: float = 100.0,
    hip_y: float = 200.0,
    torso_len: float = 100.0,
    shoulder_width: float = 60.0,
) -> PersonKeypoints:
    """Build a minimal PK with shoulders and hips for normalisation tests."""
    kps = [_make_nan_kp(i) for i in range(NUM_KEYPOINTS)]
    kps[LEFT_HIP] = _make_kp(hip_x - 20, hip_y, LEFT_HIP)
    kps[RIGHT_HIP] = _make_kp(hip_x + 20, hip_y, RIGHT_HIP)
    kps[LEFT_SHOULDER] = _make_kp(hip_x - shoulder_width / 2, hip_y - torso_len, LEFT_SHOULDER)
    kps[RIGHT_SHOULDER] = _make_kp(hip_x + shoulder_width / 2, hip_y - torso_len, RIGHT_SHOULDER)
    return PersonKeypoints(keypoints=kps)


class TestNormalizeKeypoints:
    def test_returns_array_shape(self):
        pk = _make_pk_with_body()
        result = normalize_keypoints(pk)
        assert result is not None
        assert result.shape == (17, 2)

    def test_hip_centred_origin(self):
        """After normalisation, mid-hip should be at approximately (0, 0)."""
        pk = _make_pk_with_body(hip_x=300, hip_y=400)
        result = normalize_keypoints(pk)
        assert result is not None
        # Mid-hip keypoints (LEFT_HIP, RIGHT_HIP) should average to near 0
        hip_mid_x = (result[LEFT_HIP, 0] + result[RIGHT_HIP, 0]) / 2
        hip_mid_y = (result[LEFT_HIP, 1] + result[RIGHT_HIP, 1]) / 2
        assert abs(hip_mid_x) < 0.5
        assert abs(hip_mid_y) < 0.5

    def test_torso_length_scaling(self):
        """Normalised shoulder-hip distance should be ~1.0."""
        pk = _make_pk_with_body(torso_len=150)
        result = normalize_keypoints(pk)
        assert result is not None
        # Mid-shoulder should be approximately 1.0 units above mid-hip in normalised space
        shoulder_mid_y = (result[LEFT_SHOULDER, 1] + result[RIGHT_SHOULDER, 1]) / 2
        hip_mid_y = (result[LEFT_HIP, 1] + result[RIGHT_HIP, 1]) / 2
        dist = abs(shoulder_mid_y - hip_mid_y)
        assert abs(dist - 1.0) < 0.2

    def test_nan_keypoints_stay_nan(self):
        """Missing keypoints should remain NaN after normalisation."""
        pk = _make_pk_with_body()
        result = normalize_keypoints(pk)
        assert result is not None
        # NOSE (index 0) was not set -> should be NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])

    def test_none_when_no_reference_points(self):
        """No hip AND no shoulder keypoints -> normalisation should return None."""
        kps = [_make_nan_kp(i) for i in range(NUM_KEYPOINTS)]
        pk = PersonKeypoints(keypoints=kps)
        result = normalize_keypoints(pk)
        assert result is None

    def test_fallback_to_shoulders_when_no_hips(self):
        """No hip keypoints but shoulders visible -> uses mid-shoulder as origin."""
        kps = [_make_nan_kp(i) for i in range(NUM_KEYPOINTS)]
        kps[LEFT_SHOULDER] = _make_kp(100, 100, LEFT_SHOULDER)
        kps[RIGHT_SHOULDER] = _make_kp(200, 100, RIGHT_SHOULDER)
        pk = PersonKeypoints(keypoints=kps)
        result = normalize_keypoints(pk)
        # Falls back to shoulder-based normalisation, not None
        assert result is not None

    def test_different_positions_produce_same_normalised(self):
        """Two people at different pixel positions with same pose should
        normalise to the same coordinates."""
        pk1 = _make_pk_with_body(hip_x=100, hip_y=200, torso_len=100)
        pk2 = _make_pk_with_body(hip_x=400, hip_y=300, torso_len=100)
        n1 = normalize_keypoints(pk1)
        n2 = normalize_keypoints(pk2)
        assert n1 is not None and n2 is not None
        # Shoulder positions should be very similar in normalised space
        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
            if not np.isnan(n1[idx, 0]):
                assert abs(n1[idx, 0] - n2[idx, 0]) < 0.5
                assert abs(n1[idx, 1] - n2[idx, 1]) < 0.5


class TestComputeVelocity:
    def test_zero_displacement(self):
        """Same hip position -> 0 velocity."""
        vel = compute_velocity_from_raw((100, 200), (100, 200), 100.0)
        assert vel is not None
        assert vel == 0.0

    def test_known_displacement(self):
        """10px displacement with 100px torso scale -> velocity = 0.1."""
        vel = compute_velocity_from_raw((100, 200), (110, 200), 100.0)
        assert vel is not None
        assert abs(vel - 0.1) < 0.01

    def test_diagonal_displacement(self):
        """Diagonal: sqrt(3^2 + 4^2) = 5px displacement, scale=100 -> 0.05."""
        vel = compute_velocity_from_raw((100, 200), (103, 204), 100.0)
        assert vel is not None
        assert abs(vel - 0.05) < 0.01

    def test_none_when_prev_missing(self):
        vel = compute_velocity_from_raw(None, (100, 200), 100.0)
        assert vel is None

    def test_none_when_curr_missing(self):
        vel = compute_velocity_from_raw((100, 200), None, 100.0)
        assert vel is None

    def test_none_when_scale_missing(self):
        vel = compute_velocity_from_raw((100, 200), (110, 200), None)
        assert vel is None

    def test_none_when_scale_zero(self):
        vel = compute_velocity_from_raw((100, 200), (110, 200), 0.0)
        assert vel is None
