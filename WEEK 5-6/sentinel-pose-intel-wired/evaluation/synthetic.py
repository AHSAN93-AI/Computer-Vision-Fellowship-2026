"""
evaluation.synthetic -- Synthetic Pose Generators

Produces ``PersonKeypoints`` objects with realistic joint positions for
each supported activity.  Used by the unit tests and evaluation harness
to exercise activity recognisers without needing real video or a GPU.

All coordinates are in a 640x480 pixel space with the person roughly
centred.  Joint angles are chosen to clearly trigger (or clearly miss)
the thresholds defined in ``app.config``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from app.pose.keypoints import (
    KEYPOINT_NAMES,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_EAR,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Keypoint,
    PersonKeypoints,
)


def _make_pk(
    coords: dict[int, Tuple[float, float]],
    confidence: float = 0.95,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    track_id: Optional[int] = None,
) -> PersonKeypoints:
    """Build a PersonKeypoints from a sparse dict of {index: (x, y)}.

    Missing keypoints are set to NaN / confidence=0.
    """
    kps: List[Keypoint] = []
    for i in range(NUM_KEYPOINTS):
        if i in coords:
            x, y = coords[i]
            kps.append(Keypoint(x=x, y=y, confidence=confidence, index=i, name=KEYPOINT_NAMES[i]))
        else:
            kps.append(Keypoint(x=float("nan"), y=float("nan"), confidence=0.0, index=i, name=KEYPOINT_NAMES[i]))

    if bbox is None:
        # Auto-compute bbox from visible keypoints
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        if xs and ys:
            pad = 20
            bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    pk = PersonKeypoints(keypoints=kps, bbox=bbox, detection_confidence=confidence, track_id=track_id)
    return pk


# ---------- Core body point helpers ----------

def _upright_body(
    cx: float = 320.0,
    hip_y: float = 280.0,
    torso_len: float = 120.0,
    leg_len: float = 130.0,
    arm_len: float = 100.0,
) -> dict[int, Tuple[float, float]]:
    """Return a dictionary of keypoint positions for an upright standing person.

    The person faces the camera with arms at sides and legs straight.
    """
    shoulder_width = 60.0
    hip_width = 40.0

    shoulder_y = hip_y - torso_len
    knee_y = hip_y + leg_len * 0.55
    ankle_y = hip_y + leg_len
    nose_y = shoulder_y - 40.0
    eye_y = nose_y - 8.0
    ear_y = nose_y - 4.0

    elbow_y = shoulder_y + arm_len * 0.45
    wrist_y = shoulder_y + arm_len

    return {
        NOSE: (cx, nose_y),
        LEFT_EYE: (cx - 8, eye_y),
        RIGHT_EYE: (cx + 8, eye_y),
        LEFT_EAR: (cx - 18, ear_y),
        RIGHT_EAR: (cx + 18, ear_y),
        LEFT_SHOULDER: (cx - shoulder_width / 2, shoulder_y),
        RIGHT_SHOULDER: (cx + shoulder_width / 2, shoulder_y),
        LEFT_ELBOW: (cx - shoulder_width / 2 - 10, elbow_y),
        RIGHT_ELBOW: (cx + shoulder_width / 2 + 10, elbow_y),
        LEFT_WRIST: (cx - shoulder_width / 2 - 12, wrist_y),
        RIGHT_WRIST: (cx + shoulder_width / 2 + 12, wrist_y),
        LEFT_HIP: (cx - hip_width / 2, hip_y),
        RIGHT_HIP: (cx + hip_width / 2, hip_y),
        LEFT_KNEE: (cx - hip_width / 2, knee_y),
        RIGHT_KNEE: (cx + hip_width / 2, knee_y),
        LEFT_ANKLE: (cx - hip_width / 2, ankle_y),
        RIGHT_ANKLE: (cx + hip_width / 2, ankle_y),
    }


# ---------- Public pose factories ----------

def make_standing_pose(track_id: int = 1) -> PersonKeypoints:
    """Upright person with straight legs, low velocity expected."""
    return _make_pk(_upright_body(), track_id=track_id)


def make_sitting_pose(track_id: int = 1) -> PersonKeypoints:
    """Seated person: torso upright, knees bent < 120 deg."""
    body = _upright_body()
    # Bend knees to ~90 degrees by moving ankles forward
    body[LEFT_KNEE] = (body[LEFT_HIP][0] - 5, body[LEFT_HIP][1] + 50)
    body[RIGHT_KNEE] = (body[RIGHT_HIP][0] + 5, body[RIGHT_HIP][1] + 50)
    body[LEFT_ANKLE] = (body[LEFT_KNEE][0] + 50, body[LEFT_KNEE][1] + 10)
    body[RIGHT_ANKLE] = (body[RIGHT_KNEE][0] + 50, body[RIGHT_KNEE][1] + 10)
    return _make_pk(body, track_id=track_id)


def make_walking_pose(frame_offset: int = 0, track_id: int = 1) -> PersonKeypoints:
    """Walking person: upright torso, offset hips simulate displacement.

    Call with incrementing ``frame_offset`` to produce a sequence with
    hip velocity above the walking threshold.
    """
    cx = 320.0 + frame_offset * 5.0  # 5 px/frame lateral displacement
    body = _upright_body(cx=cx)
    # Alternate leg positions for gait
    phase = frame_offset % 4
    if phase < 2:
        body[LEFT_KNEE] = (body[LEFT_KNEE][0] - 5, body[LEFT_KNEE][1] - 10)
        body[RIGHT_KNEE] = (body[RIGHT_KNEE][0] + 5, body[RIGHT_KNEE][1] + 5)
    else:
        body[LEFT_KNEE] = (body[LEFT_KNEE][0] + 5, body[LEFT_KNEE][1] + 5)
        body[RIGHT_KNEE] = (body[RIGHT_KNEE][0] - 5, body[RIGHT_KNEE][1] - 10)
    return _make_pk(body, track_id=track_id)


def make_hand_raise_pose(both: bool = False, track_id: int = 1) -> PersonKeypoints:
    """Person with one or both arms raised above shoulders."""
    body = _upright_body()
    shoulder_y = body[LEFT_SHOULDER][1]
    # Raise left arm above head
    body[LEFT_ELBOW] = (body[LEFT_SHOULDER][0] - 10, shoulder_y - 50)
    body[LEFT_WRIST] = (body[LEFT_SHOULDER][0] - 5, shoulder_y - 90)
    if both:
        body[RIGHT_ELBOW] = (body[RIGHT_SHOULDER][0] + 10, shoulder_y - 50)
        body[RIGHT_WRIST] = (body[RIGHT_SHOULDER][0] + 5, shoulder_y - 90)
    return _make_pk(body, track_id=track_id)


def make_bending_pose(track_id: int = 1) -> PersonKeypoints:
    """Person bent forward: torso > 45 deg from vertical, legs straight."""
    body = _upright_body()
    # Tilt torso forward by moving shoulders forward and down
    hip_y = body[LEFT_HIP][1]
    body[LEFT_SHOULDER] = (body[LEFT_SHOULDER][0] + 80, hip_y - 40)
    body[RIGHT_SHOULDER] = (body[RIGHT_SHOULDER][0] + 80, hip_y - 40)
    body[NOSE] = (body[NOSE][0] + 100, hip_y - 30)
    body[LEFT_EYE] = (body[NOSE][0] - 5, body[NOSE][1] - 5)
    body[RIGHT_EYE] = (body[NOSE][0] + 5, body[NOSE][1] - 5)
    body[LEFT_ELBOW] = (body[LEFT_SHOULDER][0] - 10, body[LEFT_SHOULDER][1] + 40)
    body[RIGHT_ELBOW] = (body[RIGHT_SHOULDER][0] + 10, body[RIGHT_SHOULDER][1] + 40)
    body[LEFT_WRIST] = (body[LEFT_ELBOW][0] - 5, body[LEFT_ELBOW][1] + 40)
    body[RIGHT_WRIST] = (body[RIGHT_ELBOW][0] + 5, body[RIGHT_ELBOW][1] + 40)
    return _make_pk(body, track_id=track_id)


def make_fall_pose(track_id: int = 1) -> PersonKeypoints:
    """Fallen person: horizontal orientation, head near hip level.

    Bbox is wider than tall, torso nearly horizontal.
    """
    cx, cy = 320.0, 350.0
    # Person lying on the ground, spread horizontally
    coords = {
        NOSE: (cx - 100, cy - 5),
        LEFT_EYE: (cx - 108, cy - 10),
        RIGHT_EYE: (cx - 108, cy),
        LEFT_EAR: (cx - 115, cy - 12),
        RIGHT_EAR: (cx - 115, cy + 2),
        LEFT_SHOULDER: (cx - 60, cy - 20),
        RIGHT_SHOULDER: (cx - 60, cy + 20),
        LEFT_ELBOW: (cx - 80, cy - 40),
        RIGHT_ELBOW: (cx - 80, cy + 40),
        LEFT_WRIST: (cx - 95, cy - 50),
        RIGHT_WRIST: (cx - 95, cy + 50),
        LEFT_HIP: (cx + 20, cy - 15),
        RIGHT_HIP: (cx + 20, cy + 15),
        LEFT_KNEE: (cx + 70, cy - 20),
        RIGHT_KNEE: (cx + 70, cy + 20),
        LEFT_ANKLE: (cx + 120, cy - 18),
        RIGHT_ANKLE: (cx + 120, cy + 18),
    }
    bbox = (cx - 130, cy - 60, cx + 140, cy + 60)  # wide, short
    return _make_pk(coords, bbox=bbox, track_id=track_id)


def make_falling_sequence(
    n_frames: int = 20,
    track_id: int = 1,
) -> List[PersonKeypoints]:
    """Generate a sequence simulating a fall: upright -> rapid descent -> horizontal.

    The first half shows the person transitioning from upright to falling,
    the second half shows them on the ground (still).
    """
    poses = []
    mid = n_frames // 2
    for i in range(n_frames):
        if i < mid:
            # Transition: interpolate hip_y downward with accelerating speed
            t = i / max(mid - 1, 1)
            hip_y = 280.0 + t * 80.0  # move hip down rapidly
            tilt = t * 80.0  # tilt torso
            body = _upright_body(hip_y=hip_y)
            # Tilt shoulders
            sx_offset = tilt * 0.8
            body[LEFT_SHOULDER] = (body[LEFT_SHOULDER][0] + sx_offset, body[LEFT_SHOULDER][1] + tilt * 0.3)
            body[RIGHT_SHOULDER] = (body[RIGHT_SHOULDER][0] + sx_offset, body[RIGHT_SHOULDER][1] + tilt * 0.3)
            body[NOSE] = (body[NOSE][0] + sx_offset, body[NOSE][1] + tilt * 0.4)
            # Make bbox wider as person falls
            bbox_w = 80 + t * 120
            bbox_h = 200 - t * 80
            bbox = (320 - bbox_w / 2, hip_y - bbox_h * 0.6, 320 + bbox_w / 2, hip_y + bbox_h * 0.4)
            poses.append(_make_pk(body, bbox=bbox, track_id=track_id))
        else:
            # On ground (static)
            poses.append(make_fall_pose(track_id=track_id))
    return poses


def make_squat_sequence(
    n_reps: int = 2,
    frames_per_phase: int = 5,
    track_id: int = 1,
) -> List[PersonKeypoints]:
    """Generate a squat sequence: standing -> descend -> bottom -> ascend -> standing.

    Returns enough frames for ``n_reps`` complete repetitions.
    """
    poses = []
    for rep in range(n_reps):
        # Standing (knees straight ~170 deg)
        for _ in range(frames_per_phase):
            poses.append(make_standing_pose(track_id=track_id))

        # Descending
        for f in range(frames_per_phase):
            body = _upright_body()
            t = (f + 1) / frames_per_phase
            knee_bend = 50.0 * t  # progressively bend
            body[LEFT_KNEE] = (body[LEFT_KNEE][0], body[LEFT_KNEE][1] - knee_bend * 0.3)
            body[RIGHT_KNEE] = (body[RIGHT_KNEE][0], body[RIGHT_KNEE][1] - knee_bend * 0.3)
            body[LEFT_ANKLE] = (body[LEFT_ANKLE][0] + knee_bend * 0.5, body[LEFT_ANKLE][1] - knee_bend * 0.1)
            body[RIGHT_ANKLE] = (body[RIGHT_ANKLE][0] + knee_bend * 0.5, body[RIGHT_ANKLE][1] - knee_bend * 0.1)
            # Lower hips
            body[LEFT_HIP] = (body[LEFT_HIP][0], body[LEFT_HIP][1] + knee_bend * 0.3)
            body[RIGHT_HIP] = (body[RIGHT_HIP][0], body[RIGHT_HIP][1] + knee_bend * 0.3)
            poses.append(_make_pk(body, track_id=track_id))

        # Bottom (knees at ~80 deg)
        for _ in range(frames_per_phase):
            body = _upright_body()
            body[LEFT_HIP] = (body[LEFT_HIP][0], body[LEFT_HIP][1] + 30)
            body[RIGHT_HIP] = (body[RIGHT_HIP][0], body[RIGHT_HIP][1] + 30)
            body[LEFT_KNEE] = (body[LEFT_KNEE][0] + 20, body[LEFT_KNEE][1] - 10)
            body[RIGHT_KNEE] = (body[RIGHT_KNEE][0] + 20, body[RIGHT_KNEE][1] - 10)
            body[LEFT_ANKLE] = (body[LEFT_ANKLE][0] + 30, body[LEFT_ANKLE][1] - 15)
            body[RIGHT_ANKLE] = (body[RIGHT_ANKLE][0] + 30, body[RIGHT_ANKLE][1] - 15)
            poses.append(_make_pk(body, track_id=track_id))

        # Ascending
        for f in range(frames_per_phase):
            body = _upright_body()
            t = 1.0 - (f + 1) / frames_per_phase
            knee_bend = 50.0 * t
            body[LEFT_KNEE] = (body[LEFT_KNEE][0], body[LEFT_KNEE][1] - knee_bend * 0.3)
            body[RIGHT_KNEE] = (body[RIGHT_KNEE][0], body[RIGHT_KNEE][1] - knee_bend * 0.3)
            body[LEFT_HIP] = (body[LEFT_HIP][0], body[LEFT_HIP][1] + knee_bend * 0.3)
            body[RIGHT_HIP] = (body[RIGHT_HIP][0], body[RIGHT_HIP][1] + knee_bend * 0.3)
            poses.append(_make_pk(body, track_id=track_id))

    # Final standing
    for _ in range(frames_per_phase):
        poses.append(make_standing_pose(track_id=track_id))

    return poses


def make_waving_sequence(
    n_frames: int = 30,
    track_id: int = 1,
) -> List[PersonKeypoints]:
    """Generate a waving sequence: raised arm with lateral oscillation."""
    poses = []
    for f in range(n_frames):
        body = _upright_body()
        shoulder_y = body[LEFT_SHOULDER][1]
        # Raise right arm
        body[RIGHT_ELBOW] = (body[RIGHT_SHOULDER][0] + 10, shoulder_y - 50)
        # Oscillate wrist laterally with large amplitude
        osc = 80.0 * math.sin(f * math.pi / 3)  # fast oscillation, large amplitude
        body[RIGHT_WRIST] = (body[RIGHT_SHOULDER][0] + osc, shoulder_y - 90)
        poses.append(_make_pk(body, track_id=track_id))
    return poses
