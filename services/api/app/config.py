from __future__ import annotations

import os
from typing import Dict

from pydantic_settings import BaseSettings


def parse_weights(raw: str) -> Dict[str, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    weights = {}
    for p in parts:
        k, v = p.split(":", 1)
        weights[k.strip()] = float(v)
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("Invalid TRAFFIC_WEIGHTS; sum must be > 0")
    # normalize
    return {k: v / s for k, v in weights.items()}


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://recsys:recsys@localhost:5432/recsys")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registered_model_name: str = os.getenv("REGISTERED_MODEL_NAME", "recsys")
    traffic_weights_raw: str = os.getenv("TRAFFIC_WEIGHTS", "baseline:0.85,canary:0.10,control:0.05")

    @property
    def traffic_weights(self) -> Dict[str, float]:
        return parse_weights(self.traffic_weights_raw)


settings = Settings()
