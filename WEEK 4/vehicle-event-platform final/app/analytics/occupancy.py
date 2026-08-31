"""
app/analytics/occupancy.py — Real-time zone occupancy monitoring.

Tracks current, max, and average occupancy per zone.
Records time-series data in 1-minute buckets for charting.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import ZoneConfig

logger = logging.getLogger(__name__)

# Time-series bucket size in seconds
BUCKET_SECONDS = 60
# Max number of time-series buckets to retain (60 = 1 hour)
MAX_BUCKETS = 60


@dataclass
class OccupancyBucket:
    """One time-series bucket."""
    timestamp: float
    count: int


@dataclass
class ZoneOccupancyStats:
    """Occupancy statistics for one zone."""
    zone_id: str
    zone_name: str
    current_count: int = 0
    max_capacity: int = 0
    max_observed: int = 0
    total_entries: int = 0
    # Rolling time series: list of (timestamp, count) tuples
    time_series: deque = field(default_factory=lambda: deque(maxlen=MAX_BUCKETS))
    _bucket_start: float = field(default_factory=time.time)
    _bucket_sum: int = 0
    _bucket_samples: int = 0

    @property
    def average_occupancy(self) -> float:
        if not self.time_series:
            return float(self.current_count)
        total = sum(b.count for b in self.time_series)
        return total / len(self.time_series)

    @property
    def is_over_capacity(self) -> bool:
        return self.max_capacity > 0 and self.current_count > self.max_capacity

    @property
    def utilization_pct(self) -> float:
        if self.max_capacity <= 0:
            return 0.0
        return (self.current_count / self.max_capacity) * 100.0

    def time_series_json(self, minutes: int = 10) -> List[Dict]:
        cutoff = time.time() - (minutes * 60)
        return [
            {"timestamp": b.timestamp, "count": b.count}
            for b in self.time_series
            if b.timestamp >= cutoff
        ]

    def tick(self, current_count: int) -> None:
        """Called every frame to update rolling stats."""
        self.current_count = current_count
        if current_count > self.max_observed:
            self.max_observed = current_count

        now = time.time()
        self._bucket_sum += current_count
        self._bucket_samples += 1

        # Roll bucket
        if now - self._bucket_start >= BUCKET_SECONDS:
            avg = self._bucket_sum / max(1, self._bucket_samples)
            self.time_series.append(OccupancyBucket(
                timestamp=self._bucket_start,
                count=int(round(avg)),
            ))
            self._bucket_start = now
            self._bucket_sum = 0
            self._bucket_samples = 0

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "current_count": self.current_count,
            "max_capacity": self.max_capacity,
            "max_observed": self.max_observed,
            "average_occupancy": round(self.average_occupancy, 2),
            "utilization_pct": round(self.utilization_pct, 1),
            "is_over_capacity": self.is_over_capacity,
            "total_entries": self.total_entries,
        }


class OccupancyMonitor:
    """
    Monitors real-time occupancy for all polygon zones.

    Receives occupancy counts from ZoneManager and maintains stats.
    Also tracks hysteresis for over-capacity events to prevent flapping.
    """

    def __init__(self, zone_configs: List[ZoneConfig]):
        self._stats: Dict[str, ZoneOccupancyStats] = {}
        # Hysteresis state: zone_id → True if currently over capacity
        self._over_capacity_state: Dict[str, bool] = {}

        for zc in zone_configs:
            if zc.type == "polygon":
                self._stats[zc.id] = ZoneOccupancyStats(
                    zone_id=zc.id,
                    zone_name=zc.name,
                    max_capacity=zc.max_capacity,
                )
                self._over_capacity_state[zc.id] = False

        logger.info(f"OccupancyMonitor initialized for {len(self._stats)} zones")

    def update(self, zone_counts: Dict[str, int]) -> None:
        """
        Update occupancy counts for all zones.
        zone_counts: {zone_id: current_count}
        """
        for zone_id, count in zone_counts.items():
            if zone_id in self._stats:
                self._stats[zone_id].tick(count)

    def record_entry(self, zone_id: str) -> None:
        """Increment total entry count for a zone."""
        if zone_id in self._stats:
            self._stats[zone_id].total_entries += 1

    def check_over_capacity(
        self,
        zone_id: str,
        hysteresis: int = 2,
    ) -> Tuple[bool, bool]:
        """
        Check over-capacity with hysteresis to prevent flapping.

        Returns (is_now_over_capacity, state_changed)
        - Fires when: count > max_capacity
        - Resolves when: count <= max_capacity - hysteresis
        """
        stats = self._stats.get(zone_id)
        if stats is None or stats.max_capacity <= 0:
            return False, False

        was_over = self._over_capacity_state.get(zone_id, False)
        now_over = stats.is_over_capacity

        if not was_over and now_over:
            # Newly over capacity
            self._over_capacity_state[zone_id] = True
            return True, True
        elif was_over and stats.current_count <= (stats.max_capacity - hysteresis):
            # Returned below threshold with hysteresis
            self._over_capacity_state[zone_id] = False
            return False, True
        return now_over, False

    def get_stats(self, zone_id: str) -> Optional[ZoneOccupancyStats]:
        return self._stats.get(zone_id)

    def get_all_stats(self) -> Dict[str, ZoneOccupancyStats]:
        return dict(self._stats)

    def summary(self) -> List[dict]:
        return [s.to_dict() for s in self._stats.values()]
