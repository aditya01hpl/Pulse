from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models_db import Label, ModelMetricsAgg, Prediction, TrafficConfig


async def upsert_traffic_weights(session: AsyncSession, weights: Dict[str, float]) -> None:
    payload = json.dumps(weights)
    await session.execute(
        text(
            """
            INSERT INTO traffic_config (key, value, updated_at)
            VALUES (:k, :v, NOW())
            ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = NOW()
            """
        ),
        {"k": "weights", "v": payload},
    )


async def get_traffic_weights(session: AsyncSession) -> Optional[Dict[str, float]]:
    res = await session.execute(select(TrafficConfig).where(TrafficConfig.key == "weights"))
    row = res.scalar_one_or_none()
    if not row:
        return None
    return json.loads(row.value)


async def insert_prediction(
    session: AsyncSession,
    request_id: str,
    model_alias: str,
    inputs: dict,
    outputs: dict,
    latency_ms: float,
    status: str = "ok",
) -> int:
    p = Prediction(
        request_id=request_id,
        model_alias=model_alias,
        inputs_json=inputs,
        outputs_json=outputs,
        latency_ms=latency_ms,
        status=status,
    )
    session.add(p)
    await session.flush()
    return p.id


async def insert_label(session: AsyncSession, request_id: str, true_label: int) -> int:
    l = Label(request_id=request_id, true_label=true_label)
    session.add(l)
    await session.flush()
    return l.id


async def aggregate_metrics(session: AsyncSession, window_minutes: int = 60) -> None:
    # Simplified aggregation using SQL for speed in demo
    window_end = datetime.utcnow()
    window_start = window_end - timedelta(minutes=window_minutes)

    # latency percentiles per model
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

    # accuracy join
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
