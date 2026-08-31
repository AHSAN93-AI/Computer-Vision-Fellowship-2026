"""
app.vision.fall_classifier — CNN-based Fall/NotFall Classifier

Wraps the trained Ultralytics YOLOv8-cls checkpoint (2 classes:
``Fall`` / ``NotFall``) and combines it with the rule-based
``FallRecogniser`` (app.activities.fall) as a second, independent
signal.

Design
------
The rule-based recogniser looks at *pose geometry over time* (torso
angle, velocity, aspect ratio, ...). This classifier looks at the
*raw appearance* of the cropped person bounding box on a single
frame. Requiring either signal to agree (an OR-ensemble, each with
its own temporal confirmation) reduces false negatives from either
approach alone while the per-track voting window still guards
against single-frame classifier noise triggering a false alarm.

Usage::

    clf = FallClassifier()
    clf.observe(track_id=3, frame=frame, bbox=(x1, y1, x2, y2))
    is_confirmed, confidence = clf.is_confirmed(track_id=3)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


class FallClassifier:
    """Lazily-loaded YOLOv8-cls fall/not-fall classifier with voting."""

    def __init__(self) -> None:
        self._model = None
        self._load_failed = False
        self._votes: Dict[int, Deque[bool]] = defaultdict(deque)
        self._last_conf: Dict[int, float] = {}
        self._last_label: Dict[int, str] = {}

    # ── Model loading ────────────────────────────────────
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        settings = get_settings()
        if not settings.use_fall_classifier:
            return False

        try:
            from ultralytics import YOLO  # imported lazily — heavy import

            self._model = YOLO(settings.fall_classifier_model)
            logger.info(
                "Fall classifier loaded: %s (classes=%s)",
                settings.fall_classifier_model, self._model.names,
            )
            return True
        except Exception:
            logger.warning(
                "Could not load fall classifier model at %s — "
                "continuing with rule-based fall detection only.",
                settings.fall_classifier_model, exc_info=True,
            )
            self._load_failed = True
            return False

    @property
    def is_available(self) -> bool:
        return self._ensure_loaded()

    # ── Per-frame observation ───────────────────────────
    def observe(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[Tuple[str, float]]:
        """Run the classifier on one person crop and record a vote.

        Returns ``(label, confidence)`` for this single frame, or
        ``None`` if the classifier is unavailable or the crop is
        invalid.
        """
        if not self._ensure_loaded():
            return None

        settings = get_settings()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        try:
            result = self._model.predict(crop, verbose=False)[0]
            top1 = int(result.probs.top1)
            conf = float(result.probs.top1conf)
            label = result.names[top1]
        except Exception:
            logger.debug("Fall classifier inference failed", exc_info=True)
            return None

        is_fall_vote = label == "Fall" and conf >= settings.fall_classifier_confidence_threshold

        votes = self._votes[track_id]
        votes.append(is_fall_vote)
        while len(votes) > settings.fall_classifier_vote_window:
            votes.popleft()

        self._last_conf[track_id] = conf
        self._last_label[track_id] = label

        return label, conf

    def is_confirmed(self, track_id: int) -> Tuple[bool, float]:
        """Return (confirmed, confidence) based on the recent vote window.

        Confirmed when at least ``fall_classifier_vote_required`` of
        the last ``fall_classifier_vote_window`` frames were classified
        as ``Fall`` above the confidence threshold.
        """
        settings = get_settings()
        votes = self._votes.get(track_id)
        if not votes:
            return False, 0.0

        vote_count = sum(votes)
        confirmed = vote_count >= settings.fall_classifier_vote_required
        conf = self._last_conf.get(track_id, 0.0)
        return confirmed, conf

    def last_result(self, track_id: int) -> Tuple[str, float]:
        """Most recent single-frame (label, confidence) for a track."""
        return self._last_label.get(track_id, "Unknown"), self._last_conf.get(track_id, 0.0)

    def tracked_ids(self):
        """IDs currently holding vote history (for stale-track cleanup)."""
        return list(self._votes.keys())

    def forget(self, track_id: int) -> None:
        """Drop vote history for a track that is no longer active."""
        self._votes.pop(track_id, None)
        self._last_conf.pop(track_id, None)
        self._last_label.pop(track_id, None)
