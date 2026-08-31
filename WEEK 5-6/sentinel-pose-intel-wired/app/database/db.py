"""
app.database.db — SQLite Persistence Layer (§4.15)

Async SQLite access via ``aiosqlite`` for use with FastAPI.
Also provides synchronous wrappers for the pipeline thread.

Schema
------
``activity_events`` table:
  event_id, person_id, activity_type, start_time, end_time, duration,
  confidence, source_id, evidence_path, alert_status

``alert_events`` table:
  alert_id, event_id, person_id, alert_type, severity, timestamp,
  camera_source, confidence, evidence_path, duration, status,
  acknowledged_at, acknowledged_by
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

_CREATE_ACTIVITY_EVENTS = """
CREATE TABLE IF NOT EXISTS activity_events (
    event_id       TEXT PRIMARY KEY,
    person_id      INTEGER NOT NULL,
    activity_type  TEXT NOT NULL,
    start_time     REAL NOT NULL,
    end_time       REAL,
    duration       REAL,
    confidence     REAL,
    source_id      TEXT DEFAULT 'default',
    evidence_path  TEXT,
    alert_status   TEXT DEFAULT 'none',
    created_at     TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_ALERT_EVENTS = """
CREATE TABLE IF NOT EXISTS alert_events (
    alert_id        TEXT PRIMARY KEY,
    event_id        TEXT,
    person_id       INTEGER NOT NULL,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'WARNING',
    timestamp       REAL NOT NULL,
    camera_source   TEXT DEFAULT 'default',
    confidence      REAL,
    evidence_path   TEXT,
    duration        REAL,
    status          TEXT NOT NULL DEFAULT 'active',
    acknowledged_at TEXT,
    acknowledged_by TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (event_id) REFERENCES activity_events(event_id)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_ae_person     ON activity_events(person_id);
CREATE INDEX IF NOT EXISTS idx_ae_type       ON activity_events(activity_type);
CREATE INDEX IF NOT EXISTS idx_ae_start      ON activity_events(start_time);
CREATE INDEX IF NOT EXISTS idx_alert_status  ON alert_events(status);
CREATE INDEX IF NOT EXISTS idx_alert_person  ON alert_events(person_id);
"""


class Database:
    """Synchronous SQLite database wrapper.

    Used by the pipeline thread.  For async FastAPI routes, use
    ``AsyncDatabase`` (below) or run queries in a thread executor.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        settings = get_settings()
        self._path = Path(db_path) if db_path else settings.database_abs_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the database and create tables if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_CREATE_ACTIVITY_EVENTS)
            self._conn.executescript(_CREATE_ALERT_EVENTS)
            self._conn.executescript(_CREATE_INDEXES)
            self._conn.commit()
            logger.info("Database connected: %s", self._path)
        except Exception:
            logger.exception("Database connection failed: %s", self._path)
            raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database closed")

    @contextmanager
    def _cursor(self):
        """Context manager yielding a cursor with auto-commit."""
        if self._conn is None:
            self.connect()
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Activity events ────────────────────────────────

    def insert_activity_event(
        self,
        person_id: int,
        activity_type: str,
        start_time: float,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        confidence: float = 0.0,
        source_id: str = "default",
        evidence_path: Optional[str] = None,
        alert_status: str = "none",
    ) -> str:
        """Insert an activity event and return its event_id."""
        event_id = str(uuid.uuid4())[:12]
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO activity_events
                   (event_id, person_id, activity_type, start_time, end_time,
                    duration, confidence, source_id, evidence_path, alert_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, person_id, activity_type, start_time, end_time,
                 duration, confidence, source_id, evidence_path, alert_status),
            )
        logger.debug("Activity event inserted: %s (%s, person #%d)", event_id, activity_type, person_id)
        return event_id

    def query_activity_events(
        self,
        person_id: Optional[int] = None,
        activity_type: Optional[str] = None,
        start_after: Optional[float] = None,
        start_before: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query activity events with optional filters."""
        conditions = []
        params: list = []
        if person_id is not None:
            conditions.append("person_id = ?")
            params.append(person_id)
        if activity_type:
            conditions.append("activity_type = ?")
            params.append(activity_type)
        if start_after is not None:
            conditions.append("start_time >= ?")
            params.append(start_after)
        if start_before is not None:
            conditions.append("start_time <= ?")
            params.append(start_before)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM activity_events WHERE {where} ORDER BY start_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def count_activity_events(
        self,
        activity_type: Optional[str] = None,
    ) -> int:
        """Count activity events, optionally filtered by type."""
        if activity_type:
            sql = "SELECT COUNT(*) FROM activity_events WHERE activity_type = ?"
            params = [activity_type]
        else:
            sql = "SELECT COUNT(*) FROM activity_events"
            params = []
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    def get_activity_type_counts(self) -> Dict[str, int]:
        """Return {activity_type: count} for all types."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT activity_type, COUNT(*) as cnt FROM activity_events GROUP BY activity_type"
            )
            return {row["activity_type"]: row["cnt"] for row in cur.fetchall()}

    def get_average_duration_by_type(self) -> Dict[str, float]:
        """Return {activity_type: avg_duration} for completed events."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT activity_type, AVG(duration) as avg_dur FROM activity_events WHERE duration IS NOT NULL GROUP BY activity_type"
            )
            return {row["activity_type"]: row["avg_dur"] for row in cur.fetchall()}

    # ── Alert events ───────────────────────────────────

    def insert_alert(
        self,
        person_id: int,
        alert_type: str,
        severity: str,
        timestamp: float,
        confidence: float = 0.0,
        camera_source: str = "default",
        evidence_path: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Insert an alert and return its alert_id."""
        alert_id = "ALT-" + str(uuid.uuid4())[:8]
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO alert_events
                   (alert_id, event_id, person_id, alert_type, severity,
                    timestamp, camera_source, confidence, evidence_path, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
                (alert_id, event_id, person_id, alert_type, severity,
                 timestamp, camera_source, confidence, evidence_path),
            )
        logger.info("Alert inserted: %s (%s, severity=%s, person #%d)", alert_id, alert_type, severity, person_id)
        return alert_id

    def acknowledge_alert(self, alert_id: str, by: str = "operator") -> bool:
        """Mark an alert as acknowledged."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alert_events SET status='acknowledged', acknowledged_at=datetime('now'), acknowledged_by=? WHERE alert_id=?",
                (by, alert_id),
            )
            return cur.rowcount > 0

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alert_events SET status='resolved' WHERE alert_id=?",
                (alert_id,),
            )
            return cur.rowcount > 0

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Return all alerts with status 'active'."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM alert_events WHERE status='active' ORDER BY timestamp DESC")
            return [dict(row) for row in cur.fetchall()]

    def query_alerts(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query alerts with optional status filter."""
        if status:
            sql = "SELECT * FROM alert_events WHERE status=? ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params = [status, limit, offset]
        else:
            sql = "SELECT * FROM alert_events ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params = [limit, offset]
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def export_events_csv(self) -> str:
        """Export all activity events as a CSV string."""
        import csv
        import io
        with self._cursor() as cur:
            cur.execute("SELECT * FROM activity_events ORDER BY start_time DESC")
            rows = cur.fetchall()
            if not rows:
                return ""
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            return output.getvalue()
