"""
app.activities.sitting — Sitting Activity Recogniser (§4.8, rule-based)

**Rules**:
  1. Average knee angle < ``sitting_max_knee_angle`` (default 120°).
  2. Torso approximately upright (< 30° from vertical).
  3. Low hip velocity (< standing_max_velocity — seated people don't move much).

**Required keypoints**: hips, shoulders, at least one knee.

**Known limitation**: A person crouching low may trigger sitting if
  their torso stays upright.  Crouching is handled by the ergonomic
  monitor, not as a separate activity.
"""

from __future__ import annotations

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer


class SittingRecogniser(ActivityRecogniser):

    @property
    def activity_type(self) -> str:
        return "sitting"

    @property
    def display_name(self) -> str:
        return "Sitting"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []
        score = 0.0
        total_checks = 3

        # 1. Bent knees
        avg_knee = buffer.average_knee_angle(last_n=5)
        if avg_knee is not None:
            if avg_knee < settings.sitting_max_knee_angle:
                score += 1.0
                reasons.append(f"knee_angle={avg_knee:.1f}° < {settings.sitting_max_knee_angle}°")
            else:
                reasons.append(f"FAIL knee_angle={avg_knee:.1f} deg >= {settings.sitting_max_knee_angle} deg")
        else:
            reasons.append("knee_angle=N/A")
            total_checks -= 1

        # 2. Upright torso (< 30°)
        ta = buffer.average_torso_angle(last_n=5)
        if ta is not None:
            if ta < 30.0:
                score += 1.0
                reasons.append(f"torso_angle={ta:.1f}° < 30°")
            else:
                reasons.append(f"FAIL torso_angle={ta:.1f} deg >= 30 deg")
        else:
            reasons.append("torso_angle=N/A")
            total_checks -= 1

        # 3. Low velocity
        vel = buffer.average_velocity(last_n=5)
        if vel is not None:
            if vel < settings.standing_max_velocity:
                score += 1.0
                reasons.append(f"velocity={vel:.4f} < {settings.standing_max_velocity}")
            else:
                reasons.append(f"FAIL velocity={vel:.4f} >= {settings.standing_max_velocity}")
        else:
            reasons.append("velocity=N/A")
            total_checks -= 1

        confidence = score / max(total_checks, 1)
        # Knee angle is the primary condition — must be present and pass
        knee_ok = avg_knee is not None and avg_knee < settings.sitting_max_knee_angle
        detected = knee_ok and total_checks > 0 and score >= total_checks

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
