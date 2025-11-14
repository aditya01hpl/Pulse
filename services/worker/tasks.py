from __future__ import annotations

import os
from datetime import datetime, timedelta

import mlflow
from celery.schedules import crontab
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .celery_app import app

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://recsys:recsys@postgres:5432/recsys")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "recsys")

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Every 5 minutes aggregate metrics; every 10 minutes evaluate canary
    sender.add_periodic_task(300.0, aggregate_metrics.s(), name="aggregate_metrics_5m")
    sender.add_periodic_task(600.0, evaluate_canary_and_adjust.s(), name="evaluate_canary_10m")


@app.task
async def aggregate_metrics():
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
        # Reuse API repo logic via SQL; inline minimal here to avoid imports
        await session.execute(text("SELECT NOW()"))
        # Call the same SQL as API's aggregate; duplicate minimal for decoupling
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(minutes=60)
        await session.execute(text("DELETE FROM model_metrics_agg WHERE window_start = :ws AND window_end = :we"), {"ws": window_start, "we": window_end})
        await session.execute(
            text(
                """
                INSERT INTO model_metrics_agg (model_alias, window_start, window_end, p50_ms, p95_ms, p99_ms, throughput_rps, error_rate, created_at)
                SELECT
                  model_alias,
                  :ws as window_start,
                  :we as window_end,
                  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_ms,
                  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_ms,
                  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_ms,
                  COUNT(*) / (EXTRACT(EPOCH FROM (:we - :ws))) as throughput_rps,
                  AVG(CASE WHEN status <> 'ok' THEN 1.0 ELSE 0.0 END) as error_rate,
                  NOW()
                FROM predictions
                WHERE created_at BETWEEN :ws AND :we
                GROUP BY model_alias
                """
            ),
            {"ws": window_start, "we": window_end},
        )
        await session.execute(
            text(
                """
                WITH joined AS (
                  SELECT p.model_alias, p.request_id,
                         (p.outputs_json->>'pred')::int as pred,
                         l.true_label
                  FROM predictions p
                  JOIN labels l ON p.request_id = l.request_id
                  WHERE p.created_at BETWEEN :ws AND :we
                )
                UPDATE model_metrics_agg mma
                SET accuracy = sub.acc
                FROM (
                  SELECT model_alias, AVG(CASE WHEN pred = true_label THEN 1.0 ELSE 0.0 END) as acc
                  FROM joined
                  GROUP BY model_alias
                ) sub
                WHERE mma.model_alias = sub.model_alias
                  AND mma.window_start = :ws AND mma.window_end = :we
                """
            ),
            {"ws": window_start, "we": window_end},
        )
        await session.commit()


@app.task
async def evaluate_canary_and_adjust():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    async with AsyncSessionLocal() as session:
        window_end = datetime.utcnow()
        window_start = window_end - timedelta(minutes=60)
        # Read metrics for baseline and canary
        rows = (await session.execute(
            text(
                """
                SELECT model_alias, accuracy, p99_ms, error_rate
                FROM model_metrics_agg
                WHERE window_start = :ws AND window_end = :we
                  AND model_alias IN ('baseline','canary')
                """
            ), {"ws": window_start, "we": window_end}
        )).all()
        metrics = {r.model_alias: {"accuracy": r.accuracy or 0.0, "p99_ms": r.p99_ms or 1e9, "error_rate": r.error_rate or 1.0} for r in rows}
        base = metrics.get("baseline")
        can = metrics.get("canary")
        if not base or not can:
            return
        # Decision rules
        acc_gain = (can["accuracy"] - base["accuracy"]) * 100.0
        latency_ok = can["p99_ms"] <= base["p99_ms"] * 1.05
        error_ok = (can["error_rate"] - base["error_rate"]) <= 0.005

        if acc_gain > 1.0 and latency_ok and error_ok:
            # Promote canary: increase its traffic to 60%, baseline to 35%, control to 5%
            new_weights = {"baseline": 0.35, "canary": 0.60, "control": 0.05}
            await session.execute(
                text(
                    """
                    INSERT INTO traffic_config (key, value, updated_at)
                    VALUES ('weights', :v, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = NOW()
                    """
                ),
                {"v": '{"baseline": 0.35, "canary": 0.6, "control": 0.05}'},
            )
            await session.commit()
            # Optionally transition MLflow stages: canary->Production, baseline->Archived
            try:
                client = mlflow.tracking.MlflowClient()
                # Find latest versions by stage
                canary = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Staging"]) or []
                baseline = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"]) or []
                if canary:
                    client.transition_model_version_stage(REGISTERED_MODEL_NAME, canary[0].version, "Production", archive_existing_versions=False)
                if baseline:
                    client.transition_model_version_stage(REGISTERED_MODEL_NAME, baseline[0].version, "Archived", archive_existing_versions=False)
            except Exception:
                pass
        else:
            # Rollback to conservative split
            await session.execute(
                text(
                    """
                    INSERT INTO traffic_config (key, value, updated_at)
                    VALUES ('weights', :v, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = NOW()
                    """
                ),
                {"v": '{"baseline": 0.85, "canary": 0.10, "control": 0.05}'},
            )
            await session.commit()
