"""
app.activities.waving — Waving Activity Recogniser (bonus #2, rule-based)

**Rules**:
  1. One wrist is above its corresponding shoulder (hand raised).
  2. Lateral wrist oscillation detected in the buffer — at least
     ``waving_min_oscillations`` (default 2) direction changes within
     the buffer window.

**Required keypoints**: at least one shoulder + wrist pair, plus
  enough buffer history to detect oscillation.

**Known limitation**: Very slow or very small waves may not produce
  enough oscillation; a single sustained hand-raise will not trigger
  waving (only Hand Raised).
"""

from __future__ import annotations

from app.activities.base_activity import ActivityCandidate, ActivityRecogniser
from app.config import get_settings
from app.pose.keypoints import (
    LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_WRIST,
    PersonKeypoints,
)
from app.pose.sequence import PoseSequenceBuffer


class WavingRecogniser(ActivityRecogniser):

    @property
    def activity_type(self) -> str:
        return "waving"

    @property
    def display_name(self) -> str:
        return "Waving"

    def evaluate(
        self, keypoints: PersonKeypoints, buffer: PoseSequenceBuffer,
    ) -> ActivityCandidate:
        settings = get_settings()
        reasons = []

        # 1. At least one wrist above shoulder
        hand_up = False
        for s_idx, w_idx, side in [
            (LEFT_SHOULDER, LEFT_WRIST, "left"),
            (RIGHT_SHOULDER, RIGHT_WRIST, "right"),
        ]:
            s = keypoints.get(s_idx)
            w = keypoints.get(w_idx)
            if s is not None and w is not None and w.y < s.y:
                hand_up = True
                reasons.append(f"{side} wrist above shoulder")
                break

        if not hand_up:
            return ActivityCandidate(
                activity_type=self.activity_type,
                display_name=self.display_name,
                is_detected=False,
                confidence=0.0,
                rule_explanation="No wrist above shoulder",
            )

        # 2. Lateral oscillation in buffer
        osc = buffer.wrist_oscillations(last_n=20)
        min_osc = settings.waving_min_oscillations
        if osc >= min_osc:
            reasons.append(f"oscillations={osc} >= {min_osc}")
            detected = True
            confidence = min(1.0, 0.7 + 0.1 * osc)
        else:
            reasons.append(f"oscillations={osc} < {min_osc} (not enough)")
            detected = False
            confidence = 0.3

        return ActivityCandidate(
            activity_type=self.activity_type,
            display_name=self.display_name,
            is_detected=detected,
            confidence=confidence,
            rule_explanation=" | ".join(reasons),
        )
