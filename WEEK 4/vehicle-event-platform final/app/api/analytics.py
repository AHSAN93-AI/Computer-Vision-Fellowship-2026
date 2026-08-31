"""
app/api/analytics.py — Analytics and metrics endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

from app.database import repository as db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_pipeline = None


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@router.get("/live")
async def get_live_metrics():
    """Current live metrics from the pipeline state."""
    if _pipeline is None:
        return {
            "running": False,
            "fps": 0,
            "active_tracks": 0,
            "active_events": 0,
            "zone_occupancies": {},
            "line_counts": {},
            "perf": {},
        }

    state = _pipeline.state
    perf = _pipeline.get_performance_stats()

    return {
        "running": state.running,
        "fps": round(state.fps, 1),
        "active_tracks": state.active_tracks,
        "active_events": state.active_events,
        "zone_occupancies": state.zone_occupancies,
        "line_counts": state.line_counts,
        "perf": perf,
        "frame_id": state.frame_id,
    }


@router.get("/summary")
async def get_summary():
    """Aggregated event analytics from the database."""
    return await db.get_analytics_summary()


@router.get("/occupancy/{zone_id}")
async def get_occupancy(zone_id: str, minutes: int = Query(10, ge=1, le=60)):
    """Occupancy time series for a zone."""
    if _pipeline is None:
        return {"zone_id": zone_id, "time_series": []}

    if _pipeline._occupancy_monitor is None:
        return {"zone_id": zone_id, "time_series": []}

    stats = _pipeline._occupancy_monitor.get_stats(zone_id)
    if stats is None:
        return {"zone_id": zone_id, "time_series": []}

    return {
        "zone_id": zone_id,
        "zone_name": stats.zone_name,
        "current_count": stats.current_count,
        "max_capacity": stats.max_capacity,
        "max_observed": stats.max_observed,
        "average_occupancy": round(stats.average_occupancy, 2),
        "utilization_pct": round(stats.utilization_pct, 1),
        "is_over_capacity": stats.is_over_capacity,
        "time_series": stats.time_series_json(minutes=minutes),
    }


@router.get("/dwell/{zone_id}")
async def get_dwell_stats(zone_id: str):
    """Dwell time statistics for a zone."""
    if _pipeline is None or _pipeline._dwell_tracker is None:
        return {"zone_id": zone_id}

    stats = _pipeline._dwell_tracker.get_zone_stats(zone_id)
    return {
        "zone_id": zone_id,
        "current_dwells": {
            str(k): round(v, 1)
            for k, v in stats.current_dwell_by_track.items()
        },
        "average_dwell_seconds": round(stats.average_dwell_seconds, 1),
        "max_dwell_seconds": round(stats.max_dwell_seconds, 1),
        "completed_samples": len(stats.completed_samples),
    }


@router.get("/occupancy")
async def get_all_occupancy():
    """Occupancy for all zones."""
    if _pipeline is None or _pipeline._occupancy_monitor is None:
        return {"zones": []}
    return {"zones": _pipeline._occupancy_monitor.summary()}


@router.get("/performance")
async def get_performance():
    """System performance metrics."""
    if _pipeline is None:
        return {}
    return _pipeline.get_performance_stats()
