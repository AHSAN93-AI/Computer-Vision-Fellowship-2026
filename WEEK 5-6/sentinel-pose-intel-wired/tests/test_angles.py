"""
tests.test_angles -- Unit tests for app.pose.angles

Covers calculate_angle with known geometries, all named angle functions,
torso_angle sign convention, and auxiliary functions.
"""

import math
import pytest
import numpy as np

from app.pose.keypoints import (
    Keypoint, PersonKeypoints, KEYPOINT_NAMES, NUM_KEYPOINTS,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, NOSE,
)
from app.pose.angles import (
    calculate_angle,
    left_elbow_angle,
    right_elbow_angle,
    left_knee_angle,
    right_knee_angle,
    hip_angle,
    torso_angle,
    torso_length,
    shoulder_width,
    head_to_hip_vertical,
    bbox_aspect_ratio,
)


def _make_kp(x: float, y: float, idx: int, conf: float = 0.95) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=conf, index=idx, name=KEYPOINT_NAMES[idx])


def _make_nan_kp(idx: int) -> Keypoint:
    return Keypoint(x=float("nan"), y=float("nan"), confidence=0.0, index=idx, name=KEYPOINT_NAMES[idx])


def _make_pk_from_dict(coords: dict) -> PersonKeypoints:
    """Build PersonKeypoints from a dict of {index: (x, y)}."""
    kps = []
    for i in range(NUM_KEYPOINTS):
        if i in coords:
            kps.append(_make_kp(coords[i][0], coords[i][1], i))
        else:
            kps.append(_make_nan_kp(i))
    return PersonKeypoints(keypoints=kps)


# ── calculate_angle ──────────────────────────────────────

class TestCalculateAngle:
    def test_right_angle(self):
        """90-degree angle at origin."""
        angle = calculate_angle((1, 0), (0, 0), (0, 1))
        assert abs(angle - 90.0) < 0.1

    def test_straight_line(self):
        """180-degree angle (collinear points)."""
        angle = calculate_angle((-1, 0), (0, 0), (1, 0))
        assert abs(angle - 180.0) < 0.1

    def test_zero_angle(self):
        """0-degree angle (same direction)."""
        angle = calculate_angle((1, 0), (0, 0), (2, 0))
        assert angle < 1.0  # near 0

    def test_45_degrees(self):
        angle = calculate_angle((1, 0), (0, 0), (1, 1))
        assert abs(angle - 45.0) < 0.5

    def test_60_degrees(self):
        angle = calculate_angle((1, 0), (0, 0), (0.5, math.sqrt(3) / 2))
        assert abs(angle - 60.0) < 0.5

    def test_coincident_points_returns_zero(self):
        """If two points coincide, return 0 (degenerate case)."""
        angle = calculate_angle((0, 0), (0, 0), (1, 0))
        assert angle == 0.0


# ── Named angle functions ────────────────────────────────

class TestNamedAngles:
    def test_left_elbow_angle_straight(self):
        """Straight arm: shoulder-elbow-wrist in a line -> ~180 deg."""
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (100, 100),
            LEFT_ELBOW: (100, 150),
            LEFT_WRIST: (100, 200),
        })
        angle = left_elbow_angle(pk)
        assert angle is not None
        assert abs(angle - 180.0) < 1.0

    def test_left_elbow_angle_bent(self):
        """Bent arm at ~90 degrees."""
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (100, 100),
            LEFT_ELBOW: (100, 150),
            LEFT_WRIST: (150, 150),
        })
        angle = left_elbow_angle(pk)
        assert angle is not None
        assert abs(angle - 90.0) < 1.0

    def test_right_elbow_angle_missing_keypoint(self):
        """Missing wrist returns None."""
        pk = _make_pk_from_dict({
            RIGHT_SHOULDER: (200, 100),
            RIGHT_ELBOW: (200, 150),
        })
        assert right_elbow_angle(pk) is None

    def test_left_knee_angle_straight(self):
        """Straight leg -> ~180 deg."""
        pk = _make_pk_from_dict({
            LEFT_HIP: (100, 100),
            LEFT_KNEE: (100, 200),
            LEFT_ANKLE: (100, 300),
        })
        angle = left_knee_angle(pk)
        assert angle is not None
        assert abs(angle - 180.0) < 1.0

    def test_right_knee_angle_bent(self):
        """Bent knee at ~90 degrees."""
        pk = _make_pk_from_dict({
            RIGHT_HIP: (200, 100),
            RIGHT_KNEE: (200, 200),
            RIGHT_ANKLE: (300, 200),
        })
        angle = right_knee_angle(pk)
        assert angle is not None
        assert abs(angle - 90.0) < 1.0

    def test_hip_angle_with_all_midpoints(self):
        """Hip angle with visible shoulders, hips, knees."""
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (90, 100),
            RIGHT_SHOULDER: (110, 100),
            LEFT_HIP: (90, 200),
            RIGHT_HIP: (110, 200),
            LEFT_KNEE: (90, 300),
            RIGHT_KNEE: (110, 300),
        })
        angle = hip_angle(pk)
        assert angle is not None
        assert abs(angle - 180.0) < 1.0  # straight body

    def test_hip_angle_missing_knees(self):
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (90, 100),
            RIGHT_SHOULDER: (110, 100),
            LEFT_HIP: (90, 200),
            RIGHT_HIP: (110, 200),
        })
        assert hip_angle(pk) is None


# ── torso_angle ──────────────────────────────────────────

class TestTorsoAngle:
    def test_upright_is_near_zero(self):
        """Person standing straight -> ~0 deg from vertical."""
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (90, 100),
            RIGHT_SHOULDER: (110, 100),
            LEFT_HIP: (90, 200),
            RIGHT_HIP: (110, 200),
        })
        angle = torso_angle(pk)
        assert angle is not None
        assert angle < 5.0  # near 0

    def test_horizontal_is_near_90(self):
        """Person lying flat -> ~90 deg."""
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (200, 200),
            RIGHT_SHOULDER: (200, 220),
            LEFT_HIP: (100, 200),
            RIGHT_HIP: (100, 220),
        })
        angle = torso_angle(pk)
        assert angle is not None
        assert abs(angle - 90.0) < 5.0

    def test_tilted_45_degrees(self):
        """Person tilted 45 deg."""
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (170, 100),
            RIGHT_SHOULDER: (170, 100),
            LEFT_HIP: (100, 170),
            RIGHT_HIP: (100, 170),
        })
        angle = torso_angle(pk)
        assert angle is not None
        assert abs(angle - 45.0) < 5.0

    def test_missing_shoulders_returns_none(self):
        pk = _make_pk_from_dict({
            LEFT_HIP: (100, 200),
            RIGHT_HIP: (120, 200),
        })
        assert torso_angle(pk) is None


# ── Auxiliary functions ──────────────────────────────────

class TestAuxiliaryFunctions:
    def test_torso_length(self):
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (90, 100),
            RIGHT_SHOULDER: (110, 100),
            LEFT_HIP: (90, 200),
            RIGHT_HIP: (110, 200),
        })
        length = torso_length(pk)
        assert length is not None
        assert abs(length - 100.0) < 1.0  # vertical distance = 100px

    def test_shoulder_width(self):
        pk = _make_pk_from_dict({
            LEFT_SHOULDER: (100, 100),
            RIGHT_SHOULDER: (150, 100),
        })
        width = shoulder_width(pk)
        assert width is not None
        assert abs(width - 50.0) < 0.1

    def test_head_to_hip_vertical_positive_when_above(self):
        """Nose above hip -> positive value."""
        pk = _make_pk_from_dict({
            NOSE: (100, 50),
            LEFT_HIP: (90, 200),
            RIGHT_HIP: (110, 200),
        })
        val = head_to_hip_vertical(pk)
        assert val is not None
        assert val > 0  # head above hip

    def test_head_to_hip_vertical_negative_when_below(self):
        """Nose below hip (e.g., head hanging) -> negative value."""
        pk = _make_pk_from_dict({
            NOSE: (100, 250),
            LEFT_HIP: (90, 200),
            RIGHT_HIP: (110, 200),
        })
        val = head_to_hip_vertical(pk)
        assert val is not None
        assert val < 0  # head below hip

    def test_bbox_aspect_ratio_portrait(self):
        """Upright person bbox: taller than wide -> ratio < 1."""
        pk = PersonKeypoints(
            keypoints=[_make_nan_kp(i) for i in range(NUM_KEYPOINTS)],
            bbox=(100, 50, 200, 400),  # w=100, h=350
        )
        ratio = bbox_aspect_ratio(pk)
        assert ratio is not None
        assert ratio < 1.0

    def test_bbox_aspect_ratio_landscape(self):
        """Fallen person bbox: wider than tall -> ratio > 1."""
        pk = PersonKeypoints(
            keypoints=[_make_nan_kp(i) for i in range(NUM_KEYPOINTS)],
            bbox=(50, 200, 400, 280),  # w=350, h=80
        )
        ratio = bbox_aspect_ratio(pk)
        assert ratio is not None
        assert ratio > 1.0

    def test_bbox_aspect_ratio_none_when_no_bbox(self):
        pk = PersonKeypoints(
            keypoints=[_make_nan_kp(i) for i in range(NUM_KEYPOINTS)],
            bbox=None,
        )
        assert bbox_aspect_ratio(pk) is None
