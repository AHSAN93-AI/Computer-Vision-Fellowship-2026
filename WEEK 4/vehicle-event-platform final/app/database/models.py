"""
app/database/models.py — SQLAlchemy ORM models (async / aiosqlite).

Tables:
  - events: all event records
  - zone_configs: DB-backed zone configuration
  - rule_configs: DB-backed rule configuration
  - occupancy_records: time-series occupancy data
"""

from __future__ import annotations

import time
from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class EventModel(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    rule_id = Column(String(100), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    track_id = Column(Integer, nullable=True, index=True)
    zone_id = Column(String(100), nullable=False, index=True)
    zone_name = Column(String(200), nullable=True)
    class_name = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    source_id = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(Float, nullable=False, default=time.time)
    updated_at = Column(Float, nullable=False, default=time.time)
    resolved_at = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True, default=0.0)
    evidence_path = Column(Text, nullable=True)
    event_metadata = Column(JSON, nullable=True)


class OccupancyRecord(Base):
    __tablename__ = "occupancy_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(100), nullable=False, index=True)
    zone_name = Column(String(200), nullable=True)
    timestamp = Column(Float, nullable=False, index=True)
    count = Column(Integer, nullable=False, default=0)
    max_capacity = Column(Integer, nullable=True)


class ZoneConfigModel(Base):
    __tablename__ = "zone_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    zone_type = Column(String(50), nullable=False)
    config_json = Column(JSON, nullable=True)
    enabled = Column(Boolean, default=True)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)


class RuleConfigModel(Base):
    __tablename__ = "rule_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    enabled = Column(Boolean, default=True)
    zone_id = Column(String(100), nullable=True)
    condition = Column(String(100), nullable=True)
    config_json = Column(JSON, nullable=True)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)
