from .traffic import choose_model, get_weights
from .schemas import PredictBatchRequest, PredictBatchResponse, PredictRequest, PredictResponse
from .repository import insert_prediction, upsert_traffic_weights
from .mlflow_loader import registry
from .db import Base, engine, get_session
from .config import settings
import time
import uuid
from typing import List

import numpy as np
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import Counter, Histogram
from starlette.responses import Response

app = FastAPI(title="RecSys Gateway", version="0.1.0")

# Basic metrics (extended)
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "api_request_total", "Total API requests", ["endpoint", "method", "status"], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint", "method"], registry=REGISTRY
)
MODEL_TRAFFIC = Counter(
    "model_traffic_total", "Requests by model", ["model_alias"], registry=REGISTRY
)
MODEL_ERRORS = Counter(
    "model_errors_total", "Errors by model", ["model_alias"], registry=REGISTRY
)
MODEL_LATENCY = Histogram(
    "model_latency_seconds", "Latency by model", ["model_alias"], registry=REGISTRY
)


@app.on_event("startup")
async def on_startup():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Store initial weights if not present
    from sqlalchemy.ext.asyncio import AsyncSession

    async with get_session() as session:  # type: AsyncSession
        await upsert_traffic_weights(session, settings.traffic_weights)
        await session.commit()
    # Load models
    registry.load_all()


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
async def readyz() -> JSONResponse:
    # Simple readiness: models loaded
    ok = len(registry.models) >= 1
    return JSONResponse({"ready": ok})


@app.get("/metrics")
async def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return Response(data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, session=Depends(get_session)):
    start = time.perf_counter()
    request_id = req.request_id or uuid.uuid4().hex[:16]
    weights = await get_weights(session)
    model_alias = choose_model(weights)
    status_code = "200"
    pred_int = -1
    try:
        x = np.array(req.features, dtype=float)
        y_pred = registry.predict_single(model_alias, x)
        pred_int = int(y_pred[0]) if hasattr(
            y_pred, "__len__") else int(y_pred)
        elapsed = (time.perf_counter() - start)
        MODEL_TRAFFIC.labels(model_alias=model_alias).inc()
        MODEL_LATENCY.labels(model_alias=model_alias).observe(elapsed)
        await insert_prediction(
            session,
            request_id=request_id,
            model_alias=model_alias,
            inputs={"features": req.features},
            outputs={"pred": pred_int},
            latency_ms=elapsed * 1000.0,
            status="ok",
        )
        await session.commit()
        return PredictResponse(model_alias=model_alias, pred=pred_int)
    except Exception:
        MODEL_ERRORS.labels(model_alias=model_alias).inc()
        status_code = "500"
        await insert_prediction(
            session,
            request_id=request_id,
            model_alias=model_alias,
            inputs={"features": req.features},
            outputs={"error": True},
            latency_ms=(time.perf_counter() - start) * 1000.0,
            status="error",
        )
        await session.commit()
        raise
    finally:
        REQUEST_COUNT.labels(endpoint="/predict",
                             method="POST", status=status_code).inc()
        REQUEST_LATENCY.labels(
            endpoint="/predict", method="POST").observe(time.perf_counter() - start)


@app.post("/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(req: PredictBatchRequest, session=Depends(get_session)):
    start = time.perf_counter()
    request_id = req.request_id or uuid.uuid4().hex[:16]
    weights = await get_weights(session)
    model_alias = choose_model(weights)
    status_code = "200"
    try:
        X = np.array(req.features, dtype=float)
        y_pred = registry.predict_batch(model_alias, X)
        preds = [int(v) for v in y_pred]
        elapsed = (time.perf_counter() - start)
        MODEL_TRAFFIC.labels(model_alias=model_alias).inc()
        MODEL_LATENCY.labels(model_alias=model_alias).observe(elapsed)
        await insert_prediction(
            session,
            request_id=request_id,
            model_alias=model_alias,
            inputs={"features": req.features},
            outputs={"preds": preds},
            latency_ms=elapsed * 1000.0,
            status="ok",
        )
        await session.commit()
        return PredictBatchResponse(model_alias=model_alias, preds=preds)
    except Exception:
        MODEL_ERRORS.labels(model_alias=model_alias).inc()
        status_code = "500"
        await insert_prediction(
            session,
            request_id=request_id,
            model_alias=model_alias,
            inputs={"features": req.features},
            outputs={"error": True},
            latency_ms=(time.perf_counter() - start) * 1000.0,
            status="error",
        )
        await session.commit()
        raise
    finally:
        REQUEST_COUNT.labels(endpoint="/predict/batch",
                             method="POST", status=status_code).inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch",
                               method="POST").observe(time.perf_counter() - start)
