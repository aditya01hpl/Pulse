from __future__ import annotations

import os
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from synthetic_data import RecSysDataConfig, generate_synthetic_recsys, split_features_labels


def export_onnx(model, input_dim: int) -> bytes:
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    return onx.SerializeToString()


def train_models(X_train: np.ndarray, y_train: np.ndarray):
    models = {}

    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=500, n_jobs=1)
    lr.fit(X_train, y_train)
    models["baseline_lr"] = lr

    # Canary: RandomForest
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    models["canary_rf"] = rf

    # Control: SGDClassifier (hinge -> linear SVM-like)
    sgd = SGDClassifier(loss="log_loss", max_iter=2000, random_state=0)
    sgd.fit(X_train, y_train)
    models["control_sgd"] = sgd

    return models


def log_and_register_models(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: List[str],
    registered_model_name: str = "recsys",
):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("recsys-local")

    results: List[Tuple[str, str, float, str]] = []  # (alias, run_id, acc, version)

    for alias, model in models.items():
        with mlflow.start_run(run_name=f"train_{alias}") as run:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_param("model_alias", alias)

            signature = infer_signature(pd.DataFrame(X_test, columns=feature_cols), y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=pd.DataFrame(X_test[:5], columns=feature_cols),
            )

            # Optional ONNX export as artifact
            try:
                onnx_bytes = export_onnx(model, input_dim=X_test.shape[1])
                onnx_path = "model-onnx"
                with open("model.onnx", "wb") as f:
                    f.write(onnx_bytes)
                mlflow.log_artifact("model.onnx", artifact_path=onnx_path)
                os.remove("model.onnx")
            except Exception as e:
                # ONNX may fail for some estimators; ignore
                print(f"ONNX export failed for {alias}: {e}")

            results.append((alias, run.info.run_id, acc, ""))

    # Fetch latest versions and assign stages
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    # Map alias by most recent creation time
    alias_to_version = {}
    for v in sorted(versions, key=lambda x: int(x.version)):
        run_id = v.run_id
        run = client.get_run(run_id)
        alias = run.data.params.get("model_alias")
        if alias:
            alias_to_version[alias] = v.version

    # Stage policy
    # baseline -> Production, canary -> Staging, control -> Archived
    stage_map = {"baseline_lr": "Production", "canary_rf": "Staging", "control_sgd": "Archived"}

    for alias, stage in stage_map.items():
        version = alias_to_version.get(alias)
        if version:
            try:
                client.transition_model_version_stage(
                    name=registered_model_name, version=version, stage=stage, archive_existing_versions=False
                )
                print(f"Set {registered_model_name} v{version} ({alias}) -> {stage}")
            except Exception as e:
                print(f"Stage transition failed for {alias}: {e}")


def main():
    cfg = RecSysDataConfig()
    df = generate_synthetic_recsys(cfg)
    X, y, feature_cols = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = train_models(X_train, y_train)
    log_and_register_models(models, X_test, y_test, feature_cols)
    print("Training and registration complete.")


if __name__ == "__main__":
    main()
