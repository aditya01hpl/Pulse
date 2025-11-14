from __future__ import annotations

import os
from typing import Dict

import mlflow

from .config import settings


class ModelRegistry:
    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def load_all(self) -> None:
        # baseline -> Production, canary -> Staging, control -> Archived
        mapping = {
            "baseline": (settings.registered_model_name, "Production"),
            "canary": (settings.registered_model_name, "Staging"),
            "control": (settings.registered_model_name, "Archived"),
        }
        for alias, (name, stage) in mapping.items():
            self.models[alias] = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{stage}")

    def predict_single(self, alias: str, features):
        model = self.models[alias]
        # Ensure 2D for single row
        import numpy as np

        arr = features
        if not hasattr(features, "shape"):
            arr = np.array([features], dtype=float)
        return model.predict(arr)

    def predict_batch(self, alias: str, X):
        model = self.models[alias]
        return model.predict(X)


registry = ModelRegistry()
