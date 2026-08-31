"""
app.activities.standing — Standing Activity Recogniser (§4.8, rule-based)

**Rules**:
  1. Torso angle < ``standing_max_torso_angle`` (default 15°) from vertical.
  2. Average hip velocity < ``standing_max_velocity`` (default 0.008 normalised).
  3. Average knee angle > 150° (legs mostly straight).

**Required keypoints**: shoulders, hips (for torso angle + velocity),
  at least one knee-ankle pair.

**Known limitation**: Cannot distinguish standing from very slow walking
  (velocity near the threshold).  Walking recogniser has priority when
  both fire simultaneously.
"""

from __future__ import annotations

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer


class StandingRecogniser(ActivityRecogniser):

    @property
    def activity_type(self) -> str:
        return "standing"

    @property
    def display_name(self) -> str:
        return "Standing"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []
        score = 0.0
        total_checks = 3

        # 1. Torso angle
        ta = buffer.average_torso_angle(last_n=5)
        if ta is not None:
            if ta < settings.standing_max_torso_angle:
                score += 1.0
                reasons.append(f"torso_angle={ta:.1f}° < {settings.standing_max_torso_angle}°")
            else:
                reasons.append(f"FAIL torso_angle={ta:.1f} deg >= {settings.standing_max_torso_angle} deg")
        else:
            reasons.append("torso_angle=N/A (missing keypoints)")

        # 2. Low velocity
        vel = buffer.average_velocity(last_n=5)
        if vel is not None:
            if vel < settings.standing_max_velocity:
                score += 1.0
                reasons.append(f"velocity={vel:.4f} < {settings.standing_max_velocity}")
            else:
                reasons.append(f"FAIL velocity={vel:.4f} >= {settings.standing_max_velocity}")
        else:
            reasons.append("velocity=N/A")

        # 3. Knees mostly straight
        avg_knee = buffer.average_knee_angle(last_n=5)
        if avg_knee is not None:
            if avg_knee > 150.0:
                score += 1.0
                reasons.append(f"knee_angle={avg_knee:.1f}° > 150°")
            else:
                reasons.append(f"FAIL knee_angle={avg_knee:.1f}° ≤ 150°")
        else:
            # If knees aren't visible, don't penalise but don't reward
            total_checks -= 1
            reasons.append("knee_angle=N/A (not penalised)")

        confidence = score / max(total_checks, 1)
        detected = total_checks > 0 and score >= total_checks  # All visible checks must pass

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
