"""
Requirement 21: Inspection Database (SQLite).
"""

import os
import sqlite3
from contextlib import contextmanager

from app.config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    timestamp TEXT,
    product_type TEXT,
    model_version TEXT,
    status TEXT,
    defect_count INTEGER,
    max_severity TEXT,
    classifier_confidence REAL,
    anomaly_score REAL,
    defect_area_ratio REAL,
    processing_time_ms REAL,
    evidence_path TEXT,
    invalid_reason TEXT
);

CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT,
    defect_class TEXT,
    confidence REAL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_inspection(result: dict, defect_confidences: dict | None = None):
    """
    result: InspectionResult.to_dict() style dict.
    defect_confidences: optional {defect_class: confidence} for the `defects` table.
    """
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO inspections (
                inspection_id, timestamp, product_type, model_version, status,
                defect_count, max_severity, classifier_confidence, anomaly_score,
                defect_area_ratio, processing_time_ms, evidence_path, invalid_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["inspection_id"],
                result["timestamp"],
                result["product_type"],
                result["model_version"],
                result["status"],
                result["defect_count"],
                result["max_severity"],
                result["classifier_confidence"],
                result["anomaly_score"],
                result["defect_area_ratio"],
                result["processing_time_ms"],
                result["evidence_path"],
                result.get("invalid_reason"),
            ),
        )
        if defect_confidences:
            for defect_class, confidence in defect_confidences.items():
                conn.execute(
                    "INSERT INTO defects (inspection_id, defect_class, confidence) VALUES (?, ?, ?)",
                    (result["inspection_id"], defect_class, confidence),
                )


def query_inspections(status=None, defect=None, date=None, limit=200, offset=0):
    query = "SELECT * FROM inspections WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if date:
        query += " AND substr(timestamp, 1, 10) = ?"
        params.append(date)
    if defect:
        query += """ AND inspection_id IN (
            SELECT inspection_id FROM defects WHERE defect_class = ?
        )"""
        params.append(defect)
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_inspection(inspection_id: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)
        ).fetchone()
        return dict(row) if row else None


def get_analytics():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM inspections").fetchone()["c"]
        by_status = conn.execute(
            "SELECT status, COUNT(*) c FROM inspections GROUP BY status"
        ).fetchall()
        by_defect = conn.execute(
            "SELECT defect_class, COUNT(*) c FROM defects GROUP BY defect_class ORDER BY c DESC"
        ).fetchall()
        by_severity = conn.execute(
            "SELECT max_severity, COUNT(*) c FROM inspections WHERE status='FAIL' GROUP BY max_severity"
        ).fetchall()
        over_time = conn.execute(
            """
            SELECT substr(timestamp, 1, 10) day,
                   SUM(CASE WHEN status='FAIL' THEN 1 ELSE 0 END) fails,
                   COUNT(*) total
            FROM inspections GROUP BY day ORDER BY day
            """
        ).fetchall()
        avg_latency = conn.execute(
            "SELECT AVG(processing_time_ms) a FROM inspections"
        ).fetchone()["a"]

        status_counts = {r["status"]: r["c"] for r in by_status}
        passed = status_counts.get("PASS", 0)
        failed = status_counts.get("FAIL", 0)
        invalid = status_counts.get("INVALID", 0)

        return {
            "total_inspections": total,
            "passed": passed,
            "failed": failed,
            "invalid": invalid,
            "pass_rate": (passed / total) if total else 0.0,
            "defect_rate": (failed / total) if total else 0.0,
            "invalid_rate": (invalid / total) if total else 0.0,
            "defects_by_category": {r["defect_class"]: r["c"] for r in by_defect},
            "severity_distribution": {r["max_severity"]: r["c"] for r in by_severity},
            "defect_rate_over_time": [dict(r) for r in over_time],
            "avg_processing_time_ms": avg_latency or 0.0,
        }


def save_settings(decision: dict, severity: dict):
    import json

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES ('decision', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (json.dumps(decision),),
        )
        conn.execute(
            "INSERT INTO settings (key, value) VALUES ('severity', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (json.dumps(severity),),
        )


def load_settings():
    import json

    with get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        out = {}
        for r in rows:
            out[r["key"]] = json.loads(r["value"])
        return out
