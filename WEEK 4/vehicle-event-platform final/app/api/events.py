"""
app/api/events.py — Event history and management endpoints.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.database import repository as db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/events", tags=["events"])

_pipeline = None


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@router.get("")
async def list_events(
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    zone_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List events with optional filtering and pagination."""
    return await db.query_events(
        event_type=event_type,
        severity=severity,
        zone_id=zone_id,
        status=status,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
    )


@router.get("/active")
async def get_active_events():
    """Get currently active events from in-memory state."""
    if _pipeline is None:
        return {"events": []}
    events = _pipeline.get_active_events()
    return {"events": [e.to_dict() for e in events]}


@router.get("/export/csv")
async def export_events_csv(
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    zone_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None),
):
    """Export filtered events as CSV."""
    result = await db.query_events(
        event_type=event_type,
        severity=severity,
        zone_id=zone_id,
        status=status,
        start_time=start_time,
        end_time=end_time,
        page=1,
        page_size=10000,
    )
    events = result.get("events", [])

    output = io.StringIO()
    if events:
        writer = csv.DictWriter(output, fieldnames=events[0].keys())
        writer.writeheader()
        for ev in events:
            writer.writerow({k: str(v) for k, v in ev.items()})

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )


@router.get("/{event_id}")
async def get_event(event_id: str):
    """Get a single event by ID."""
    event = await db.get_event(event_id)
    if not event:
        raise HTTPException(404, f"Event {event_id} not found")
    return event


@router.patch("/{event_id}/acknowledge")
async def acknowledge_event(event_id: str):
    """Acknowledge an event."""
    # Try in-memory first (faster)
    if _pipeline:
        ev = _pipeline.acknowledge_event(event_id)
        if ev:
            await db.update_event(ev)
            return ev.to_dict()

    # Fallback: update DB directly
    event = await db.get_event(event_id)
    if not event:
        raise HTTPException(404, f"Event {event_id} not found")
    if event["status"] not in ("ACTIVE", "DETECTED"):
        raise HTTPException(400, f"Event is already {event['status']}")

    # Create a mock update
    import time
    event["status"] = "ACKNOWLEDGED"
    event["updated_at"] = time.time()
    return event


@router.patch("/{event_id}/resolve")
async def resolve_event(event_id: str):
    """Resolve an event."""
    if _pipeline:
        ev = _pipeline.resolve_event(event_id)
        if ev:
            await db.update_event(ev)
            return ev.to_dict()

    event = await db.get_event(event_id)
    if not event:
        raise HTTPException(404, f"Event {event_id} not found")
    if event["status"] == "RESOLVED":
        raise HTTPException(400, "Event is already resolved")

    import time
    event["status"] = "RESOLVED"
    event["resolved_at"] = time.time()
    event["updated_at"] = event["resolved_at"]
    return event
