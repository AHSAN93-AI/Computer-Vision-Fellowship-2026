"""
app/database/repository.py — Async database operations using aiosqlite + SQLAlchemy.

All operations are async and handle DB failures gracefully:
- Logs errors and returns safe defaults
- Never crashes the processing pipeline on DB failure
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, func, text

from app.database.models import Base, EventModel, OccupancyRecord

logger = logging.getLogger(__name__)

# ─── Database setup ───────────────────────────────────────────────────────────

_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/events.db")
_engine = None
_AsyncSessionLocal = None


def _get_db_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/events.db")
    # Ensure data directory exists for sqlite
    if "sqlite" in url:
        db_path = url.replace("sqlite+aiosqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return url


async def init_db() -> None:
    """Initialize the database, creating tables if they don't exist."""
    global _engine, _AsyncSessionLocal
    try:
        db_url = _get_db_url()
        _engine = create_async_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        )
        _AsyncSessionLocal = sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Database initialized: {db_url}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Set up in-memory fallback
        try:
            _engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
            _AsyncSessionLocal = sessionmaker(
                _engine, class_=AsyncSession, expire_on_commit=False
            )
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.warning("Using in-memory SQLite database (data will not persist)")
        except Exception as e2:
            logger.critical(f"Failed to initialize even in-memory DB: {e2}")


def _get_session():
    if _AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _AsyncSessionLocal()


# ─── Event CRUD ───────────────────────────────────────────────────────────────

async def create_event(event: object) -> Optional[EventModel]:
    """Persist a new event to the database."""
    try:
        async with _get_session() as session:
            model = EventModel(
                event_id=event.event_id,
                rule_id=event.rule_id,
                event_type=event.event_type.value,
                severity=event.severity.value,
                status=event.status.value,
                track_id=event.track_id,
                zone_id=event.zone_id,
                zone_name=event.zone_name,
                class_name=event.class_name,
                description=event.description,
                source_id=event.source_id,
                confidence=event.confidence,
                created_at=event.created_at,
                updated_at=event.updated_at,
                resolved_at=event.resolved_at,
                duration_seconds=event.duration_seconds,
                evidence_path=event.evidence_path,
                event_metadata=event.metadata,
            )
            session.add(model)
            await session.commit()
            return model
    except Exception as e:
        logger.error(f"DB create_event failed: {e}")
        return None


async def update_event(event: object) -> bool:
    """Update an existing event's status and duration."""
    try:
        async with _get_session() as session:
            stmt = (
                update(EventModel)
                .where(EventModel.event_id == event.event_id)
                .values(
                    status=event.status.value,
                    updated_at=event.updated_at,
                    resolved_at=event.resolved_at,
                    duration_seconds=event.duration_seconds,
                    description=event.description,
                    evidence_path=event.evidence_path,
                    event_metadata=event.metadata,
                )
            )
            await session.execute(stmt)
            await session.commit()
            return True
    except Exception as e:
        logger.error(f"DB update_event failed: {e}")
        return False


async def get_event(event_id: str) -> Optional[Dict]:
    """Fetch a single event by ID."""
    try:
        async with _get_session() as session:
            result = await session.execute(
                select(EventModel).where(EventModel.event_id == event_id)
            )
            row = result.scalar_one_or_none()
            return _event_to_dict(row) if row else None
    except Exception as e:
        logger.error(f"DB get_event failed: {e}")
        return None


async def query_events(
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    zone_id: Optional[str] = None,
    status: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    page: int = 1,
    page_size: int = 50,
) -> Dict[str, Any]:
    """Query events with filters and pagination."""
    try:
        async with _get_session() as session:
            query = select(EventModel)

            if event_type:
                query = query.where(EventModel.event_type == event_type)
            if severity:
                query = query.where(EventModel.severity == severity)
            if zone_id:
                query = query.where(EventModel.zone_id == zone_id)
            if status:
                query = query.where(EventModel.status == status)
            if start_time:
                query = query.where(EventModel.created_at >= start_time)
            if end_time:
                query = query.where(EventModel.created_at <= end_time)

            # Count
            count_query = select(func.count()).select_from(query.subquery())
            total = (await session.execute(count_query)).scalar_one()

            # Paginate
            offset = (page - 1) * page_size
            query = query.order_by(EventModel.created_at.desc()).offset(offset).limit(page_size)
            rows = (await session.execute(query)).scalars().all()

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "events": [_event_to_dict(r) for r in rows],
            }
    except Exception as e:
        logger.error(f"DB query_events failed: {e}")
        return {"total": 0, "page": page, "page_size": page_size, "events": []}


async def get_analytics_summary() -> Dict[str, Any]:
    """Aggregate event counts for dashboard."""
    try:
        async with _get_session() as session:
            # By type
            type_result = await session.execute(
                text("SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type")
            )
            by_type = {row[0]: row[1] for row in type_result.fetchall()}

            # By severity
            sev_result = await session.execute(
                text("SELECT severity, COUNT(*) as cnt FROM events GROUP BY severity")
            )
            by_severity = {row[0]: row[1] for row in sev_result.fetchall()}

            # By zone
            zone_result = await session.execute(
                text("SELECT zone_id, COUNT(*) as cnt FROM events GROUP BY zone_id")
            )
            by_zone = {row[0]: row[1] for row in zone_result.fetchall()}

            # Total + active
            total = (await session.execute(text("SELECT COUNT(*) FROM events"))).scalar_one()
            active = (await session.execute(
                text("SELECT COUNT(*) FROM events WHERE status IN ('ACTIVE','DETECTED')")
            )).scalar_one()

            # Events over last 24 hours in 1-hour buckets
            now = time.time()
            day_ago = now - 86400
            hourly_result = await session.execute(
                text(
                    "SELECT CAST((created_at - :day_ago) / 3600 AS INTEGER) as hour_bucket, COUNT(*) "
                    "FROM events WHERE created_at >= :day_ago "
                    "GROUP BY hour_bucket ORDER BY hour_bucket"
                ),
                {"day_ago": day_ago}
            )
            hourly = [{"hour": row[0], "count": row[1]} for row in hourly_result.fetchall()]

            return {
                "total": total,
                "active": active,
                "by_type": by_type,
                "by_severity": by_severity,
                "by_zone": by_zone,
                "hourly_last_24h": hourly,
            }
    except Exception as e:
        logger.error(f"DB get_analytics_summary failed: {e}")
        return {"total": 0, "active": 0, "by_type": {}, "by_severity": {}, "by_zone": {}, "hourly_last_24h": []}


async def record_occupancy(
    zone_id: str, zone_name: str, count: int, max_capacity: int
) -> None:
    """Store an occupancy snapshot."""
    try:
        async with _get_session() as session:
            rec = OccupancyRecord(
                zone_id=zone_id,
                zone_name=zone_name,
                timestamp=time.time(),
                count=count,
                max_capacity=max_capacity,
            )
            session.add(rec)
            await session.commit()
    except Exception as e:
        logger.error(f"DB record_occupancy failed: {e}")


def _event_to_dict(row: EventModel) -> Dict[str, Any]:
    return {
        "event_id": row.event_id,
        "rule_id": row.rule_id,
        "event_type": row.event_type,
        "severity": row.severity,
        "status": row.status,
        "track_id": row.track_id,
        "zone_id": row.zone_id,
        "zone_name": row.zone_name,
        "class_name": row.class_name,
        "description": row.description,
        "source_id": row.source_id,
        "confidence": row.confidence,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "resolved_at": row.resolved_at,
        "duration_seconds": row.duration_seconds,
        "evidence_path": row.evidence_path,
        "metadata": row.event_metadata or {},
    }
