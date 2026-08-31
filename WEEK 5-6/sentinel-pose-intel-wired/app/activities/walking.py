"""
app.activities.walking — Walking Activity Recogniser (§4.8, rule-based)

**Rules**:
  1. Average hip velocity > ``walking_velocity_threshold`` (default 0.015).
  2. Torso approximately upright (< 25° from vertical).
  3. Knees show gait-like variation (not both locked straight, not both deeply bent).

**Required keypoints**: hips, shoulders (velocity + torso angle).

**Known limitation**: Very slow walking near the threshold may flicker
  between Standing and Walking — the state machine handles this via
  persistence requirements.
"""

from __future__ import annotations

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer


class WalkingRecogniser(ActivityRecogniser):

    @property
    def activity_type(self) -> str:
        return "walking"

    @property
    def display_name(self) -> str:
        return "Walking"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []
        score = 0.0
        total_checks = 2  # velocity + torso are mandatory

        # 1. Hip velocity above threshold
        vel = buffer.average_velocity(last_n=5)
        if vel is not None:
            if vel > settings.walking_velocity_threshold:
                score += 1.0
                reasons.append(f"velocity={vel:.4f} > {settings.walking_velocity_threshold}")
            else:
                reasons.append(f"FAIL velocity={vel:.4f} ≤ {settings.walking_velocity_threshold}")
        else:
            reasons.append("velocity=N/A")

        # 2. Upright torso
        ta = buffer.average_torso_angle(last_n=5)
        if ta is not None:
            if ta < 25.0:
                score += 1.0
                reasons.append(f"torso_angle={ta:.1f}° < 25°")
            else:
                reasons.append(f"FAIL torso_angle={ta:.1f} deg >= 25 deg")
        else:
            reasons.append("torso_angle=N/A")

        # 3. Bonus: knee angle variation (gait signature)
        avg_knee = buffer.average_knee_angle(last_n=5)
        if avg_knee is not None:
            if 100.0 < avg_knee < 175.0:
                score += 0.3  # bonus, not mandatory
                reasons.append(f"knee_angle={avg_knee:.1f}° (gait range)")

        confidence = min(1.0, score / max(total_checks, 1))
        detected = vel is not None and vel > settings.walking_velocity_threshold and (
            ta is None or ta < 25.0  # allow missing torso
        )

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
