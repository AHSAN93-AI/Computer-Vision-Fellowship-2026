"""
app/analytics/zones.py — Zone management, polygon containment, zone state tracking.

Manages polygon zones and line-crossing zones.
Emits ZoneEvent objects only on state transitions (entry/exit), not every frame.
Uses supervision PolygonZone where available, falls back to manual point-in-polygon.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.config import ZoneConfig
from app.vision.tracker import TrackedVehicle

logger = logging.getLogger(__name__)


class ZoneEventType(str, Enum):
    ENTRY = "ZONE_ENTRY"
    EXIT = "ZONE_EXIT"


@dataclass
class ZoneEvent:
    """A zone entry or exit state transition for a tracked vehicle."""
    event_type: ZoneEventType
    track_id: int
    class_name: str
    zone_id: str
    zone_name: str
    timestamp: float
    confidence: float
    centroid: Tuple[float, float]


@dataclass
class ZoneOccupancy:
    """Current occupancy state for one zone."""
    zone_id: str
    occupant_track_ids: Set[int] = field(default_factory=set)
    entry_times: Dict[int, float] = field(default_factory=dict)  # track_id → entry timestamp

    @property
    def count(self) -> int:
        return len(self.occupant_track_ids)

    def dwell_seconds(self, track_id: int) -> float:
        if track_id not in self.entry_times:
            return 0.0
        return time.time() - self.entry_times[track_id]


def point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """
    Ray-casting algorithm for point-in-polygon test.
    polygon: list of [x, y] vertices.
    """
    x, y = point
    n = len(polygon)
    inside = False
    px, py = polygon[-1]
    for i in range(n):
        cx, cy = polygon[i]
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-10) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


class ZoneManager:
    """
    Manages all configured zones and tracks vehicle presence in each.

    For polygon zones: point-in-polygon per centroid.
    For line zones: delegated to LineCrossingMonitor.
    Emits ZoneEvent only on state transitions (not every frame).
    """

    def __init__(self, zone_configs: List[ZoneConfig]):
        self._zones: Dict[str, ZoneConfig] = {}
        self._sv_zones: Dict[str, object] = {}         # supervision PolygonZone (optional)
        self._occupancy: Dict[str, ZoneOccupancy] = {} # zone_id → ZoneOccupancy

        for zc in zone_configs:
            if zc.type == "polygon":
                self._zones[zc.id] = zc
                self._occupancy[zc.id] = ZoneOccupancy(zone_id=zc.id)
                self._try_init_sv_zone(zc)
            elif zc.type == "line_crossing":
                # Line zones are handled by LineCrossingMonitor, but we still
                # track which vehicles are "near" the line for zone membership
                self._zones[zc.id] = zc
                self._occupancy[zc.id] = ZoneOccupancy(zone_id=zc.id)

        logger.info(f"ZoneManager initialized with {len(self._zones)} zones")

    def _try_init_sv_zone(self, zc: ZoneConfig) -> None:
        """Attempt to init supervision PolygonZone (optional, falls back gracefully)."""
        try:
            import supervision as sv
            poly = np.array(zc.polygon, dtype=np.int32)
            sv_zone = sv.PolygonZone(polygon=poly)
            self._sv_zones[zc.id] = sv_zone
            logger.debug(f"supervision PolygonZone created for zone '{zc.id}'")
        except Exception as e:
            logger.debug(f"supervision PolygonZone not available for '{zc.id}': {e} — using fallback")

    def is_in_zone(self, zone_id: str, point: Tuple[float, float]) -> bool:
        """Check if a centroid point is inside the named polygon zone."""
        zc = self._zones.get(zone_id)
        if zc is None or zc.type != "polygon":
            return False
        # Use supervision if available for consistency
        if zone_id in self._sv_zones:
            try:
                import supervision as sv
                sv_zone = self._sv_zones[zone_id]
                # Check single point manually against polygon
                return point_in_polygon(point, zc.polygon)
            except Exception:
                pass
        return point_in_polygon(point, zc.polygon)

    def update(
        self,
        tracked_vehicles: List[TrackedVehicle],
        frame_id: int,
    ) -> List[ZoneEvent]:
        """
        Process all tracked vehicles against all polygon zones.
        Returns list of ZoneEvent for state transitions only.
        """
        events: List[ZoneEvent] = []
        now = time.time()

        # Set of currently active track IDs
        active_track_ids: Set[int] = {tv.track_id for tv in tracked_vehicles}

        for zone_id, zc in self._zones.items():
            if zc.type != "polygon":
                continue

            occupancy = self._occupancy[zone_id]
            current_in_zone: Set[int] = set()

            for tv in tracked_vehicles:
                # Filter by monitored classes
                if tv.class_name not in zc.monitored_classes:
                    continue

                cx, cy = tv.centroid
                in_zone = point_in_polygon((cx, cy), zc.polygon)

                if in_zone:
                    current_in_zone.add(tv.track_id)

                    if tv.track_id not in occupancy.occupant_track_ids:
                        # State transition: ENTRY
                        occupancy.occupant_track_ids.add(tv.track_id)
                        occupancy.entry_times[tv.track_id] = now
                        tv.current_zone = zone_id
                        tv.zone_entry_time = now

                        events.append(ZoneEvent(
                            event_type=ZoneEventType.ENTRY,
                            track_id=tv.track_id,
                            class_name=tv.class_name,
                            zone_id=zone_id,
                            zone_name=zc.name,
                            timestamp=now,
                            confidence=tv.confidence,
                            centroid=(cx, cy),
                        ))
                        logger.debug(f"Zone ENTRY: track={tv.track_id} zone={zone_id}")
                else:
                    if tv.track_id in occupancy.occupant_track_ids:
                        # State transition: EXIT
                        occupancy.occupant_track_ids.discard(tv.track_id)
                        occupancy.entry_times.pop(tv.track_id, None)
                        tv.previous_zone = tv.current_zone
                        tv.current_zone = None

                        events.append(ZoneEvent(
                            event_type=ZoneEventType.EXIT,
                            track_id=tv.track_id,
                            class_name=tv.class_name,
                            zone_id=zone_id,
                            zone_name=zc.name,
                            timestamp=now,
                            confidence=tv.confidence,
                            centroid=(cx, cy),
                        ))
                        logger.debug(f"Zone EXIT: track={tv.track_id} zone={zone_id}")

            # Handle vehicles that disappeared (lost track)
            disappeared = occupancy.occupant_track_ids - active_track_ids - current_in_zone
            for lost_id in disappeared:
                # Vehicle disappeared without an explicit exit — clean up occupancy
                # (dwell tracker handles graceful expiry)
                occupancy.occupant_track_ids.discard(lost_id)
                occupancy.entry_times.pop(lost_id, None)

        return events

    def get_occupancy(self, zone_id: str) -> Optional[ZoneOccupancy]:
        return self._occupancy.get(zone_id)

    def get_all_occupancies(self) -> Dict[str, ZoneOccupancy]:
        return dict(self._occupancy)

    def get_zone_config(self, zone_id: str) -> Optional[ZoneConfig]:
        return self._zones.get(zone_id)

    def get_all_zone_configs(self) -> List[ZoneConfig]:
        return list(self._zones.values())

    def zone_ids(self) -> List[str]:
        return list(self._zones.keys())
