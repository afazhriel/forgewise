"""Reproducible ML baseline for ForgeWise.

Predicts ``Machine failure`` from the verified AI4I sensor feature set.
Deliberately simple and reproducible:

* fixed random seed and stratified 80/20 split
* leakage-safe features (failure-mode flags and IDs excluded)
* class-balanced RandomForest (handles the ~3.4% positive rate)
* model + metadata (version, hashes, split indices) persisted for audit

Run:  python -m src.train_model
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import data_loader

PROJECT_ROOT = data_loader.PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

HYPERPARAMETERS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("numeric", "passthrough", data_loader.NUMERIC_FEATURES),
        ]
    )
    model = RandomForestClassifier(**HYPERPARAMETERS)
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train(verbose: bool = True) -> tuple[Pipeline, dict]:
    df = data_loader.load_dataset()
    X, y = data_loader.split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    test_indices = X_test.index.tolist()
    trained_at = datetime.now(timezone.utc)
    dataset_hash = data_loader.dataset_sha256()
    version = f"fw-{trained_at:%Y%m%dT%H%M%SZ}-{dataset_hash[:8]}"

    metadata = {
        "model_version": version,
        "trained_at": trained_at.isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": HYPERPARAMETERS,
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "stratified": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_positives": int(y_train.sum()),
            "test_positives": int(y_test.sum()),
            "test_indices": test_indices,
        },
        "features": {
            "feature_columns": data_loader.FEATURE_COLUMNS,
            "numeric": data_loader.NUMERIC_FEATURES,
            "categorical": ["Type"],
            "excluded_columns": data_loader.EXCLUDED_COLUMNS,
        },
        "target": data_loader.TARGET,
        "dataset": {
            "source_page": data_loader.SOURCE_PAGE,
            "source_url": data_loader.SOURCE_URL,
            "license": data_loader.LICENSE,
            "sha256": dataset_hash,
            "n_rows": int(len(df)),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if verbose:
        print(f"Trained {metadata['model_type']} -> {MODEL_PATH}")
        print(f"Model version : {version}")
        print(f"Train rows    : {len(X_train)} (pos={int(y_train.sum())})")
        print(f"Test rows     : {len(X_test)} (pos={int(y_test.sum())})")

    return pipeline, metadata


def load_model(path: str = str(MODEL_PATH)) -> tuple[Pipeline, dict]:
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["metadata"]


if __name__ == "__main__":
    train()
