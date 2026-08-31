"""
app.activities.bending — Bending Activity Recogniser (bonus #1, rule-based)

**Rules**:
  1. Torso angle > ``bending_min_torso_angle`` (default 45°) from vertical.
  2. Hip Y relatively stable (not falling rapidly) — velocity < fall_speed_threshold.
  3. Knee angles > 140° (legs mostly straight — distinguishes from sitting/crouching).

**Required keypoints**: shoulders, hips (torso angle), ideally knees.

**Known limitation**: A person bending with very bent knees may look
  like crouching instead.
"""

from __future__ import annotations

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer


class BendingRecogniser(ActivityRecogniser):

    @property
    def activity_type(self) -> str:
        return "bending"

    @property
    def display_name(self) -> str:
        return "Bending"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []
        score = 0.0
        total_checks = 3

        # 1. Torso leaning forward
        ta = buffer.average_torso_angle(last_n=5)
        if ta is not None:
            if ta > settings.bending_min_torso_angle:
                score += 1.0
                reasons.append(f"torso_angle={ta:.1f}° > {settings.bending_min_torso_angle}°")
            else:
                reasons.append(f"FAIL torso_angle={ta:.1f}° ≤ {settings.bending_min_torso_angle}°")
        else:
            reasons.append("torso_angle=N/A")
            total_checks -= 1

        # 2. Not falling (velocity below fall threshold)
        vel = buffer.average_velocity(last_n=5)
        if vel is not None:
            if vel < settings.fall_speed_threshold:
                score += 1.0
                reasons.append(f"velocity={vel:.4f} < {settings.fall_speed_threshold} (not falling)")
            else:
                reasons.append(f"FAIL velocity={vel:.4f} >= {settings.fall_speed_threshold}")
        else:
            reasons.append("velocity=N/A")
            total_checks -= 1

        # 3. Legs mostly straight (not crouching)
        avg_knee = buffer.average_knee_angle(last_n=5)
        if avg_knee is not None:
            if avg_knee > 140.0:
                score += 1.0
                reasons.append(f"knee_angle={avg_knee:.1f}° > 140° (straight legs)")
            else:
                reasons.append(f"FAIL knee_angle={avg_knee:.1f}° ≤ 140° (bent — crouching?)")
        else:
            total_checks -= 1
            reasons.append("knee_angle=N/A (not penalised)")

        confidence = score / max(total_checks, 1)
        # Torso angle is the primary condition
        torso_ok = ta is not None and ta > settings.bending_min_torso_angle
        detected = torso_ok and total_checks > 0 and score >= total_checks

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
