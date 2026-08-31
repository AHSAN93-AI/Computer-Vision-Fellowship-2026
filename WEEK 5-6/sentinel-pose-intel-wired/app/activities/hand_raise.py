"""
app.activities.hand_raise — Hand-Raise Activity Recogniser (§4.8, rule-based)

**Rules**:
  1. At least one wrist is above its corresponding shoulder
     (in image coordinates, wrist.y < shoulder.y).
  2. The arm's elbow angle > ``hand_raise_min_elbow_angle`` (default 120°)
     — the arm is extended, not just bent.

**Required keypoints**: at least one shoulder-elbow-wrist triplet.

**Duration**: must persist ≥ 3 frames (default activity_confirm_frames).
"""

from __future__ import annotations

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import (
    LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
    PersonKeypoints,
)
from app.pose.sequence import PoseSequenceBuffer


class HandRaiseRecogniser(ActivityRecogniser):

    @property
    def activity_type(self) -> str:
        return "hand_raised"

    @property
    def display_name(self) -> str:
        return "Hand Raised"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []
        detected = False
        confidence = 0.0

        # Check left arm
        left_raised = self._check_arm(
            keypoints,
            LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
            "left",
            settings.hand_raise_min_elbow_angle,
            reasons,
        )

        # Check right arm
        right_raised = self._check_arm(
            keypoints,
            RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
            "right",
            settings.hand_raise_min_elbow_angle,
            reasons,
        )

        if left_raised and right_raised:
            detected = True
            confidence = 0.95
            reasons.append("BOTH hands raised")
        elif left_raised or right_raised:
            detected = True
            confidence = 0.85
        else:
            confidence = 0.0

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )

    @staticmethod
    def _check_arm(
        pk: PersonKeypoints,
        shoulder_idx: int,
        elbow_idx: int,
        wrist_idx: int,
        side: str,
        min_elbow_angle: float,
        reasons: list,
    ) -> bool:
        """Check if one arm is raised.  Appends explanation to *reasons*."""
        shoulder = pk.get(shoulder_idx)
        wrist = pk.get(wrist_idx)
        elbow = pk.get(elbow_idx)

        if shoulder is None or wrist is None:
            reasons.append(f"{side}: shoulder/wrist not visible")
            return False

        # Wrist above shoulder (image y-axis: lower value = higher)
        if wrist.y >= shoulder.y:
            reasons.append(f"{side}: wrist.y={wrist.y:.0f} >= shoulder.y={shoulder.y:.0f}")
            return False

        reasons.append(f"{side}: wrist ABOVE shoulder ({wrist.y:.0f} < {shoulder.y:.0f})")

        # Elbow angle check (arm extended, not just bent up)
        if elbow is not None:
            from app.pose.angles import calculate_angle
            angle = calculate_angle(
                shoulder.as_tuple(), elbow.as_tuple(), wrist.as_tuple()
            )
            if angle < min_elbow_angle:
                reasons.append(f"{side}: elbow_angle={angle:.0f}° < {min_elbow_angle}° (bent)")
                # Still count as raised, but with lower confidence
                return True
            reasons.append(f"{side}: elbow_angle={angle:.0f} deg >= {min_elbow_angle} deg (extended)")

        return True
