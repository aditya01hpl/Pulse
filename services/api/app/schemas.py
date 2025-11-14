from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: List[float]
    request_id: Optional[str] = None


class PredictBatchRequest(BaseModel):
    features: List[List[float]]
    request_id: Optional[str] = None


class PredictResponse(BaseModel):
    model_alias: str
    pred: int


class PredictBatchResponse(BaseModel):
    model_alias: str
    preds: List[int]
