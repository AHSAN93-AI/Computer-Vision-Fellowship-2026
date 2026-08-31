"""
evaluation.evaluate -- Evaluation Harness

Runs the activity recognition pipeline against synthetic pose sequences
or recorded test clips and reports accuracy metrics per activity type.

Usage::

    # Synthetic mode (no video or GPU required):
    python -m evaluation.evaluate --synthetic

    # Video mode (requires YOLO model):
    python -m evaluation.evaluate --video-dir path/to/clips/

Each clip in video mode expects a JSON sidecar with ground-truth labels::

    video_name.mp4
    video_name.json  ->  {"labels": [{"frame": 0, "activity": "standing"}, ...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app import setup_logging
from app.config import get_settings
from app.events.activity_manager import ActivityManager
from app.pose.sequence import PoseSequenceBuffer

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


@dataclass
class ActivityMetrics:
    """Per-activity precision / recall / F1."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation output."""
    per_activity: Dict[str, ActivityMetrics] = field(default_factory=dict)
    total_frames: int = 0
    correct_frames: int = 0
    latency_ms: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct_frames / self.total_frames if self.total_frames > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        s = sorted(self.latency_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        s = sorted(self.latency_ms)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]


def evaluate_synthetic() -> EvaluationResult:
    """Run evaluation using synthetic pose sequences.

    Tests each activity type by generating known-good synthetic poses
    and verifying that the activity recognisers identify them correctly.
    """
    from evaluation.synthetic import (
        make_standing_pose,
        make_sitting_pose,
        make_walking_pose,
        make_hand_raise_pose,
        make_bending_pose,
        make_fall_pose,
        make_falling_sequence,
        make_squat_sequence,
        make_waving_sequence,
    )

    result = EvaluationResult()
    mgr = ActivityManager()
    person_id = 1

    # Define test scenarios: (expected_activity, pose_generator_frames)
    scenarios: List[Tuple[str, str, list]] = []

    # Standing: 30 frames of standing pose
    standing_frames = [make_standing_pose(track_id=person_id) for _ in range(30)]
    scenarios.append(("standing", "Standing (30 frames)", standing_frames))

    # Sitting: 30 frames of seated pose
    sitting_frames = [make_sitting_pose(track_id=person_id) for _ in range(30)]
    scenarios.append(("sitting", "Sitting (30 frames)", sitting_frames))

    # Walking: 30 frames with lateral displacement
    walking_frames = [make_walking_pose(frame_offset=i, track_id=person_id) for i in range(30)]
    scenarios.append(("walking", "Walking (30 frames)", walking_frames))

    # Hand raised: 30 frames with arm above shoulder
    hand_raise_frames = [make_hand_raise_pose(track_id=person_id) for _ in range(30)]
    scenarios.append(("hand_raised", "Hand Raised (30 frames)", hand_raise_frames))

    # Bending: 30 frames bent forward
    bending_frames = [make_bending_pose(track_id=person_id) for _ in range(30)]
    scenarios.append(("bending", "Bending (30 frames)", bending_frames))

    # Fall: 20 frame falling sequence
    fall_frames = make_falling_sequence(n_frames=20, track_id=person_id)
    scenarios.append(("fall", "Fall (20 frame sequence)", fall_frames))

    # Waving: 30 frame oscillating sequence
    waving_frames = make_waving_sequence(n_frames=30, track_id=person_id)
    scenarios.append(("waving", "Waving (30 frames)", waving_frames))

    print("\n" + "=" * 70)
    print("SYNTHETIC EVALUATION RESULTS")
    print("=" * 70)

    for expected_activity, description, frames in scenarios:
        # Reset activity manager for each scenario
        mgr = ActivityManager()

        if expected_activity not in result.per_activity:
            result.per_activity[expected_activity] = ActivityMetrics()
        metrics = result.per_activity[expected_activity]

        detected_at_end = False
        for frame_num, pk in enumerate(frames):
            t0 = time.perf_counter()
            pas = mgr.process_person(person_id, pk, frame_num)
            dt = (time.perf_counter() - t0) * 1000.0
            result.latency_ms.append(dt)
            result.total_frames += 1

            if pas.current_activity == expected_activity:
                result.correct_frames += 1

        # Check final activity state after all frames
        pas = mgr.get_person_state(person_id)
        if pas and pas.current_activity == expected_activity:
            metrics.true_positives += 1
            detected_at_end = True
        else:
            metrics.false_negatives += 1

        status = "PASS" if detected_at_end else "FAIL"
        final_act = pas.current_activity if pas else "N/A"
        print(f"  [{status}] {description:40s} -> detected: {final_act}")

    # Summary
    print("\n" + "-" * 70)
    print(f"{'Activity':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 70)
    for act, m in sorted(result.per_activity.items()):
        print(f"  {act:<18s} {m.precision:>9.1%} {m.recall:>9.1%} {m.f1:>9.1%}")

    print(f"\n  Overall accuracy:  {result.accuracy:.1%} ({result.correct_frames}/{result.total_frames} frames)")
    print(f"  Avg latency:       {result.avg_latency_ms:.2f} ms")
    print(f"  P95 latency:       {result.p95_latency_ms:.2f} ms")
    print(f"  P99 latency:       {result.p99_latency_ms:.2f} ms")
    print("=" * 70 + "\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel Pose Intel Evaluation Harness")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic evaluation (no video/GPU needed)")
    parser.add_argument("--video-dir", type=str, help="Directory of test clips with JSON sidecars")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if not args.synthetic and not args.video_dir:
        print("Specify --synthetic or --video-dir. Use --help for details.")
        sys.exit(1)

    if args.synthetic:
        result = evaluate_synthetic()
    else:
        print("Video evaluation mode not yet implemented.")
        print("Use --synthetic for now, or provide annotated clips in the future.")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_data = {
            "accuracy": result.accuracy,
            "total_frames": result.total_frames,
            "correct_frames": result.correct_frames,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "p99_latency_ms": result.p99_latency_ms,
            "per_activity": {
                act: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "tp": m.true_positives,
                    "fp": m.false_positives,
                    "fn": m.false_negatives,
                }
                for act, m in result.per_activity.items()
            },
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
