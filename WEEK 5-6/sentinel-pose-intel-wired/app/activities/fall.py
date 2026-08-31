"""
app.activities.fall — Fall Detection & Alert Lifecycle (§4.10, §4.11)

Multi-factor analysis — scores 5 indicators and requires ≥ 3 to fire:

  1. **Torso orientation**: torso angle > 60° from vertical.
  2. **Rapid descent**: max hip velocity in recent buffer exceeds threshold.
  3. **Aspect ratio**: bounding box width/height > 1.0 (wider than tall).
  4. **Head-to-hip**: head Y approaches or goes below hip Y.
  5. **Post-fall stillness**: velocity drops after the rapid descent.

**Distinguishes from (where feasible)**:
  • Sitting down: slower descent + knee flexion + torso stays more upright.
  • Bending: torso tilts but hip Y doesn't drop rapidly; returns upright.
  • Lying normally: no rapid descent (already on ground when first seen).

**Alert lifecycle** (§4.11):
  Possible Fall → Fall Confirmed (after confirm_frames of still-down) →
  Alert Active → Acknowledged (user action) → Resolved.

  No repeated alerts for the same person within ``fall_alert_cooldown_seconds``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import PersonKeypoints
from app.pose.sequence import PoseSequenceBuffer

logger = logging.getLogger(__name__)


class FallRecogniser(ActivityRecogniser):
    """Multi-factor fall detector.

    This is the *recogniser* — it answers "does this frame look like a
    fall?"  The alert lifecycle (Possible → Confirmed → Active →
    Acknowledged → Resolved) is managed by the event/alert layer.
    """

    @property
    def activity_type(self) -> str:
        return "fall"

    @property
    def display_name(self) -> str:
        return "Fall Detected"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []
        factors_met = 0
        total_factors = 5

        # ── Factor 1: Torso orientation ──────────────
        ta = buffer.current_torso_angle
        if ta is not None:
            if ta > settings.fall_torso_angle_threshold:
                factors_met += 1
                reasons.append(f"F1-torso: {ta:.1f}° > {settings.fall_torso_angle_threshold}° ✓")
            else:
                reasons.append(f"F1-torso: {ta:.1f}° ≤ {settings.fall_torso_angle_threshold}° ✗")
        else:
            reasons.append("F1-torso: N/A")

        # ── Factor 2: Rapid descent (max velocity in buffer) ──
        max_vel = buffer.max_velocity(last_n=15)
        if max_vel is not None:
            if max_vel > settings.fall_speed_threshold:
                factors_met += 1
                reasons.append(f"F2-speed: {max_vel:.4f} > {settings.fall_speed_threshold} ✓")
            else:
                reasons.append(f"F2-speed: {max_vel:.4f} ≤ {settings.fall_speed_threshold} ✗")
        else:
            reasons.append("F2-speed: N/A")

        # ── Factor 3: Aspect ratio (wide = fallen) ──
        snap = buffer.latest
        if snap is not None and snap.bbox_aspect_ratio is not None:
            ar = snap.bbox_aspect_ratio
            if ar > 1.0:
                factors_met += 1
                reasons.append(f"F3-aspect: {ar:.2f} > 1.0 (wider than tall) ✓")
            else:
                reasons.append(f"F3-aspect: {ar:.2f} ≤ 1.0 ✗")
        else:
            reasons.append("F3-aspect: N/A")

        # ── Factor 4: Head near or below hip level ──
        if snap is not None and snap.head_to_hip_vert is not None:
            hth = snap.head_to_hip_vert
            if hth < 10.0:  # head at or below hip level
                factors_met += 1
                reasons.append(f"F4-head: head_to_hip={hth:.0f}px (head ≈ hip level) ✓")
            else:
                reasons.append(f"F4-head: head_to_hip={hth:.0f}px (head above hip) ✗")
        else:
            reasons.append("F4-head: N/A")

        # ── Factor 5: Post-fall stillness ──
        # Recent velocity should be low (person stopped after rapid descent)
        recent_vel = buffer.average_velocity(last_n=5)
        if recent_vel is not None and max_vel is not None:
            # There was a rapid event AND now they're still
            if recent_vel < settings.standing_max_velocity and max_vel > settings.fall_speed_threshold:
                factors_met += 1
                reasons.append(f"F5-still: recent_vel={recent_vel:.4f} (still after rapid motion) ✓")
            else:
                reasons.append(f"F5-still: recent_vel={recent_vel:.4f} ✗")
        else:
            reasons.append("F5-still: N/A")

        # ── Decision ──
        detected = factors_met >= settings.fall_min_factors
        confidence = factors_met / total_factors

        summary = f"{factors_met}/{total_factors} factors met (need >={settings.fall_min_factors})"
        reasons.insert(0, summary)

        if detected:
            logger.info("Fall detected: %s", summary)

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
