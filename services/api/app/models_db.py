from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    model_alias: Mapped[str] = mapped_column(String(32), index=True)
    inputs_json: Mapped[dict] = mapped_column(JSON)
    outputs_json: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    true_label: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ModelMetricsAgg(Base):
    __tablename__ = "model_metrics_agg"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_alias: Mapped[str] = mapped_column(String(32), index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime, index=True)
    window_end: Mapped[datetime] = mapped_column(DateTime, index=True)
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    p50_ms: Mapped[Optional[float]] = mapped_column(Float)
    p95_ms: Mapped[Optional[float]] = mapped_column(Float)
    p99_ms: Mapped[Optional[float]] = mapped_column(Float)
    throughput_rps: Mapped[Optional[float]] = mapped_column(Float)
    error_rate: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class TrafficConfig(Base):
    __tablename__ = "traffic_config"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
