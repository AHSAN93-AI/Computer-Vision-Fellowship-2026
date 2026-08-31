"""
app/analytics/dwell.py — Dwell time tracking per tracked vehicle per zone.

Key design decisions:
- Brief tracking loss (< grace_seconds) does NOT reset the dwell timer.
  This prevents gaming the system by briefly leaving the frame.
- Zone exit (confirmed) DOES reset the timer for that zone.
- Tracks current dwell, rolling average dwell, and max dwell per zone.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import DwellConfig

logger = logging.getLogger(__name__)

# How many dwell samples to keep for rolling average per zone
MAX_DWELL_SAMPLES = 500


@dataclass
class DwellRecord:
    """Dwell state for one (track_id, zone_id) pair."""
    track_id: int
    zone_id: str
    entry_time: float               # when vehicle entered the zone
    last_seen_in_zone: float        # last frame timestamp when vehicle was confirmed in zone
    is_active: bool = True          # False after confirmed exit

    @property
    def current_dwell_seconds(self) -> float:
        if not self.is_active:
            return self.last_seen_in_zone - self.entry_time
        return time.time() - self.entry_time

    @property
    def gap_seconds(self) -> float:
        """How long since we last saw this vehicle in the zone."""
        return time.time() - self.last_seen_in_zone


@dataclass
class ZoneDwellStats:
    """Aggregate dwell statistics for a zone."""
    zone_id: str
    current_dwell_by_track: Dict[int, float] = field(default_factory=dict)
    # Completed dwell samples (for avg/max)
    completed_samples: deque = field(default_factory=lambda: deque(maxlen=MAX_DWELL_SAMPLES))

    @property
    def average_dwell_seconds(self) -> float:
        if not self.completed_samples:
            return 0.0
        return sum(self.completed_samples) / len(self.completed_samples)

    @property
    def max_dwell_seconds(self) -> float:
        if not self.completed_samples:
            return max(self.current_dwell_by_track.values(), default=0.0)
        return max(
            max(self.completed_samples, default=0.0),
            max(self.current_dwell_by_track.values(), default=0.0),
        )


class DwellTracker:
    """
    Tracks dwell time for each (track_id, zone_id) pair.

    Call update_presence() every frame with the set of tracks currently in each zone.
    Call confirm_exit() when a vehicle definitively exits a zone.
    Query current_dwell() to get elapsed time for a vehicle in a zone.
    """

    def __init__(self, config: DwellConfig):
        self._cfg = config
        # (track_id, zone_id) → DwellRecord
        self._records: Dict[Tuple[int, str], DwellRecord] = {}
        # zone_id → ZoneDwellStats
        self._stats: Dict[str, ZoneDwellStats] = defaultdict(lambda: ZoneDwellStats(zone_id=""))

    def _key(self, track_id: int, zone_id: str) -> Tuple[int, str]:
        return (track_id, zone_id)

    def record_presence(self, track_id: int, zone_id: str) -> None:
        """Mark a vehicle as present in a zone this frame."""
        now = time.time()
        key = self._key(track_id, zone_id)

        if key not in self._records:
            self._records[key] = DwellRecord(
                track_id=track_id,
                zone_id=zone_id,
                entry_time=now,
                last_seen_in_zone=now,
            )
            logger.debug(f"DwellTracker: new entry track={track_id} zone={zone_id}")
        else:
            rec = self._records[key]
            if not rec.is_active:
                # Vehicle re-entered zone after a confirmed exit → reset timer
                rec.entry_time = now
                rec.is_active = True
                logger.debug(f"DwellTracker: re-entry track={track_id} zone={zone_id}")
            rec.last_seen_in_zone = now

    def record_absence(self, track_id: int, zone_id: str) -> bool:
        """
        Mark vehicle as absent from zone this frame.

        If absent > grace_seconds → confirm exit and finalize dwell.
        Returns True if dwell was finalized (exit confirmed).
        """
        key = self._key(track_id, zone_id)
        if key not in self._records:
            return False
        rec = self._records[key]
        if not rec.is_active:
            return False

        if rec.gap_seconds > self._cfg.lost_track_grace_seconds:
            # Confirmed exit
            return self._finalize_dwell(track_id, zone_id)
        return False

    def confirm_exit(self, track_id: int, zone_id: str) -> float:
        """
        Explicitly confirm that a vehicle has exited a zone (e.g., from ZoneEvent).
        Returns the final dwell duration in seconds.
        """
        key = self._key(track_id, zone_id)
        if key not in self._records:
            return 0.0
        rec = self._records[key]
        duration = rec.current_dwell_seconds
        self._finalize_dwell(track_id, zone_id)
        return duration

    def _finalize_dwell(self, track_id: int, zone_id: str) -> bool:
        key = self._key(track_id, zone_id)
        if key not in self._records:
            return False
        rec = self._records[key]
        duration = rec.current_dwell_seconds
        rec.is_active = False

        stats = self._stats[zone_id]
        stats.zone_id = zone_id
        stats.current_dwell_by_track.pop(track_id, None)
        if duration > 0:
            stats.completed_samples.append(duration)

        logger.debug(
            f"DwellTracker: finalized track={track_id} zone={zone_id} dwell={duration:.1f}s"
        )
        return True

    def current_dwell(self, track_id: int, zone_id: str) -> float:
        """Get current dwell time for a (track, zone) pair in seconds."""
        key = self._key(track_id, zone_id)
        rec = self._records.get(key)
        if rec is None or not rec.is_active:
            return 0.0
        return rec.current_dwell_seconds

    def update_all(self, zone_track_map: Dict[str, List[int]]) -> None:
        """
        Batch update: provide which tracks are currently in which zones.
        Handles presence and absence updates for all known records.

        zone_track_map: {zone_id: [track_id, ...]}
        """
        now = time.time()
        all_zone_ids = set(zone_track_map.keys())

        # Update presence for vehicles in zones
        for zone_id, track_ids in zone_track_map.items():
            for tid in track_ids:
                self.record_presence(tid, zone_id)

        # Update absence for vehicles that were in zones but now aren't
        for (tid, zid), rec in list(self._records.items()):
            if not rec.is_active:
                continue
            zone_tracks = zone_track_map.get(zid, [])
            if tid not in zone_tracks:
                self.record_absence(tid, zid)

        # Update per-zone dwell stats
        for zone_id, track_ids in zone_track_map.items():
            stats = self._stats[zone_id]
            stats.zone_id = zone_id
            stats.current_dwell_by_track = {
                tid: self.current_dwell(tid, zone_id) for tid in track_ids
            }

    def get_zone_stats(self, zone_id: str) -> ZoneDwellStats:
        stats = self._stats.get(zone_id)
        if stats is None:
            return ZoneDwellStats(zone_id=zone_id)
        return stats

    def get_all_stats(self) -> Dict[str, ZoneDwellStats]:
        return dict(self._stats)

    def get_record(self, track_id: int, zone_id: str) -> Optional[DwellRecord]:
        return self._records.get(self._key(track_id, zone_id))

    def purge_track(self, track_id: int) -> None:
        """Remove all records for a lost track."""
        keys_to_remove = [k for k in self._records if k[0] == track_id]
        for k in keys_to_remove:
            self._records.pop(k, None)

    def get_dwell(self, zone_id: str, track_id: int) -> float:
        """
        Convenience alias: get_dwell(zone_id, track_id) → current dwell seconds.

        Note: tests call get_dwell(zone_id, track_id).
        The core API is current_dwell(track_id, zone_id).
        Both orderings are accepted here.
        """
        return self.current_dwell(track_id, zone_id)
