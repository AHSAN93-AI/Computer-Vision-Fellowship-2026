"""
app/analytics/lines.py — Virtual line crossing detection.

Tracks direction A→B and B→A separately.
Maintains IN/OUT counters with cooldown to prevent double-counting.
Detects wrong-direction crossings based on configured expected direction.
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

# Cooldown frames after a crossing to prevent double-count
CROSSING_COOLDOWN_FRAMES = 20
# Hysteresis: vehicle centroid must move N pixels past the line
HYSTERESIS_PX = 15


class CrossingDirection(str, Enum):
    A_TO_B = "A_to_B"
    B_TO_A = "B_to_A"
    NONE = "none"


@dataclass
class LineCrossingEvent:
    """A vehicle crossing a virtual line."""
    track_id: int
    class_name: str
    zone_id: str
    zone_name: str
    direction: CrossingDirection
    is_wrong_direction: bool
    timestamp: float
    confidence: float
    centroid: Tuple[float, float]


@dataclass
class LineState:
    """Per-line tracking state."""
    zone_id: str
    in_count: int = 0
    out_count: int = 0
    # Map track_id → last side (-1 = side A, +1 = side B, 0 = unknown)
    track_sides: Dict[int, int] = field(default_factory=dict)
    # Map track_id → frame_id of last crossing (cooldown)
    last_crossing_frame: Dict[int, int] = field(default_factory=dict)


def _signed_distance(
    point: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """
    Signed distance from point to line (p1→p2).
    Positive = left of line direction, negative = right.
    Uses cross-product of (p2-p1) and (point-p1).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Perpendicular component
    return (dx * (point[1] - p1[1])) - (dy * (point[0] - p1[0]))


class LineCrossingMonitor:
    """
    Monitors virtual lines for crossing events.

    Each "line zone" in config has a [[x1,y1],[x2,y2]] line definition.
    We define:
      - Side A = negative signed distance (right side of line vector)
      - Side B = positive signed distance (left side of line vector)
      - A→B = vehicle moves from side A to side B = configured IN direction
      - B→A = vehicle moves from side B to side A = configured OUT direction
    """

    def __init__(self, zone_configs: List[ZoneConfig]):
        self._line_zones: Dict[str, ZoneConfig] = {}
        self._line_states: Dict[str, LineState] = {}

        for zc in zone_configs:
            if zc.type == "line_crossing" and zc.line:
                if len(zc.line) == 2:
                    self._line_zones[zc.id] = zc
                    self._line_states[zc.id] = LineState(zone_id=zc.id)
                    logger.debug(f"Line zone registered: '{zc.id}'")
                else:
                    logger.warning(f"Line zone '{zc.id}' has invalid line definition")

        logger.info(f"LineCrossingMonitor initialized with {len(self._line_zones)} line zones")

    def update(
        self,
        tracked_vehicles: List[TrackedVehicle],
        frame_id: int,
    ) -> List[LineCrossingEvent]:
        """Process tracked vehicles against all line zones. Returns crossing events."""
        events: List[LineCrossingEvent] = []
        now = time.time()

        active_ids: Set[int] = {tv.track_id for tv in tracked_vehicles}

        for zone_id, zc in self._line_zones.items():
            state = self._line_states[zone_id]
            p1 = tuple(zc.line[0])
            p2 = tuple(zc.line[1])

            # Prune stale tracks
            stale = [tid for tid in state.track_sides if tid not in active_ids]
            for tid in stale:
                state.track_sides.pop(tid, None)
                state.last_crossing_frame.pop(tid, None)

            for tv in tracked_vehicles:
                if tv.class_name not in zc.monitored_classes:
                    continue

                cx, cy = tv.centroid
                dist = _signed_distance((cx, cy), p1, p2)

                # Current side: -1 if right of line (A), +1 if left (B)
                current_side = 1 if dist > HYSTERESIS_PX else (-1 if dist < -HYSTERESIS_PX else 0)

                if current_side == 0:
                    # In hysteresis band — don't update side
                    continue

                prev_side = state.track_sides.get(tv.track_id, 0)

                if prev_side == 0:
                    # First observation — record side, no crossing yet
                    state.track_sides[tv.track_id] = current_side
                    continue

                if current_side == prev_side:
                    # No side change
                    continue

                # Side changed → crossing detected
                # Check cooldown
                last_frame = state.last_crossing_frame.get(tv.track_id, -9999)
                if (frame_id - last_frame) < CROSSING_COOLDOWN_FRAMES:
                    state.track_sides[tv.track_id] = current_side
                    continue

                # Determine direction
                if prev_side == -1 and current_side == 1:
                    direction = CrossingDirection.A_TO_B
                elif prev_side == 1 and current_side == -1:
                    direction = CrossingDirection.B_TO_A
                else:
                    state.track_sides[tv.track_id] = current_side
                    continue

                # Update counters
                expected = CrossingDirection(zc.expected_direction)
                is_wrong = direction != expected

                if direction == CrossingDirection.A_TO_B:
                    state.in_count += 1
                    logger.debug(f"Line '{zone_id}': track {tv.track_id} IN (A→B), count={state.in_count}")
                else:
                    state.out_count += 1
                    logger.debug(f"Line '{zone_id}': track {tv.track_id} OUT (B→A), count={state.out_count}")

                state.track_sides[tv.track_id] = current_side
                state.last_crossing_frame[tv.track_id] = frame_id

                events.append(LineCrossingEvent(
                    track_id=tv.track_id,
                    class_name=tv.class_name,
                    zone_id=zone_id,
                    zone_name=zc.name,
                    direction=direction,
                    is_wrong_direction=is_wrong,
                    timestamp=now,
                    confidence=tv.confidence,
                    centroid=(cx, cy),
                ))

                if is_wrong:
                    logger.warning(
                        f"WRONG DIRECTION: track {tv.track_id} crossed '{zone_id}' "
                        f"going {direction.value} (expected {expected.value})"
                    )

        return events

    def get_counts(self, zone_id: str) -> Tuple[int, int]:
        """Returns (in_count, out_count) for a line zone."""
        state = self._line_states.get(zone_id)
        if state is None:
            return (0, 0)
        return (state.in_count, state.out_count)

    def get_all_counts(self) -> Dict[str, Tuple[int, int]]:
        return {
            zid: (s.in_count, s.out_count)
            for zid, s in self._line_states.items()
        }

    def reset_counts(self, zone_id: str) -> None:
        if zone_id in self._line_states:
            self._line_states[zone_id].in_count = 0
            self._line_states[zone_id].out_count = 0
