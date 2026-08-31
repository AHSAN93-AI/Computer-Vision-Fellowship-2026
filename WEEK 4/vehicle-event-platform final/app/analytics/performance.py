"""
app/analytics/performance.py — System performance monitoring.

Measures: detection inference time, tracking time, rule-processing time,
total frame time, FPS, CPU usage, memory usage.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

MAX_SAMPLES = 300  # Keep last 300 frames for rolling avg


@dataclass
class FrameTimings:
    """Timings for a single processed frame."""
    frame_id: int
    total_ms: float
    detection_ms: float
    tracking_ms: float
    analytics_ms: float
    rules_ms: float
    db_ms: float
    timestamp: float


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    fps: float = 0.0
    avg_total_ms: float = 0.0
    avg_detection_ms: float = 0.0
    avg_tracking_ms: float = 0.0
    avg_analytics_ms: float = 0.0
    avg_rules_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    bottleneck: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 1),
            "avg_total_ms": round(self.avg_total_ms, 1),
            "avg_detection_ms": round(self.avg_detection_ms, 1),
            "avg_tracking_ms": round(self.avg_tracking_ms, 1),
            "avg_analytics_ms": round(self.avg_analytics_ms, 1),
            "avg_rules_ms": round(self.avg_rules_ms, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "bottleneck": self.bottleneck,
        }


class PerformanceMonitor:
    """Collects and aggregates per-frame performance metrics."""

    def __init__(self, log_interval_seconds: int = 10):
        self._samples: deque = deque(maxlen=MAX_SAMPLES)
        self._frame_start: float = 0.0
        self._last_log_time: float = time.time()
        self._log_interval = log_interval_seconds
        self._process = psutil.Process()

    def start_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def record_frame(self, timings: FrameTimings) -> None:
        self._samples.append(timings)
        now = time.time()
        if now - self._last_log_time >= self._log_interval:
            stats = self.get_stats()
            logger.info(
                f"Perf | FPS={stats.fps:.1f} | "
                f"Total={stats.avg_total_ms:.0f}ms | "
                f"Det={stats.avg_detection_ms:.0f}ms | "
                f"Track={stats.avg_tracking_ms:.0f}ms | "
                f"CPU={stats.cpu_percent:.0f}% | "
                f"MEM={stats.memory_mb:.0f}MB | "
                f"Bottleneck={stats.bottleneck}"
            )
            self._last_log_time = now

    def get_stats(self) -> PerformanceStats:
        if not self._samples:
            return PerformanceStats()

        samples = list(self._samples)
        n = len(samples)

        avg_total = sum(s.total_ms for s in samples) / n
        avg_det = sum(s.detection_ms for s in samples) / n
        avg_track = sum(s.tracking_ms for s in samples) / n
        avg_analytics = sum(s.analytics_ms for s in samples) / n
        avg_rules = sum(s.rules_ms for s in samples) / n

        # FPS: frames in last second
        now = time.time()
        recent = [s for s in samples if (now - s.timestamp) <= 1.0]
        fps = float(len(recent))

        # Bottleneck: which stage takes the most time
        stage_times = {
            "detection": avg_det,
            "tracking": avg_track,
            "analytics": avg_analytics,
            "rules": avg_rules,
        }
        bottleneck = max(stage_times, key=stage_times.get)

        try:
            cpu = self._process.cpu_percent(interval=None)
            mem = self._process.memory_info().rss / 1_048_576  # bytes → MB
        except Exception:
            cpu = 0.0
            mem = 0.0

        return PerformanceStats(
            fps=fps,
            avg_total_ms=avg_total,
            avg_detection_ms=avg_det,
            avg_tracking_ms=avg_track,
            avg_analytics_ms=avg_analytics,
            avg_rules_ms=avg_rules,
            cpu_percent=cpu,
            memory_mb=mem,
            bottleneck=bottleneck,
        )
