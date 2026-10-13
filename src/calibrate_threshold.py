"""Reproducible validation-set threshold analysis.

The 0.5 threshold used to report confusion-matrix metrics in
``src/evaluate_model.py`` is an arbitrary default. This module selects a
probability threshold on a **validation** split and reports its effect on the
**untouched held-out test set**.

Design — no test-set tuning:

* The held-out test indices come from ``models/model_metadata.json`` (produced by
  ``src/train_model.py``) and are **never** used to choose the threshold.
* A stratified validation split is carved out of the *training pool* (all rows
  that are not in the held-out test set) with a fixed seed.
* A calibration clone of the pipeline (same deterministic configuration) is fit
  on the remaining training rows and scored on the validation split.
* The threshold is selected to maximise F1 on validation; ties break toward
  higher recall.
* The selected threshold is then *reported* on the held-out test set for
  comparison only. ``artifacts/metrics.json`` is never overwritten.

The app's risk bands (LOW < 0.30, HIGH >= 0.70) are a separate, documented
policy; this analysis informs but does not automatically change them.

Run:  python -m src.calibrate_threshold
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src import data_loader, train_model

ARTIFACTS_DIR = data_loader.PROJECT_ROOT / "artifacts"
ANALYSIS_PATH = ARTIFACTS_DIR / "threshold_analysis.json"

CURRENT_THRESHOLD = 0.5
VAL_SIZE = 0.25
VAL_RANDOM_STATE = 123
SELECTION_METRIC = "f1"
THRESHOLD_GRID = [round(x, 3) for x in np.arange(0.05, 1.0, 0.05)]


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without the dataset)
# ---------------------------------------------------------------------------
def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """Precision / recall / F1 / confusion matrix at a single threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def sweep(y_true, y_score, thresholds=THRESHOLD_GRID) -> list[dict]:
    """Metrics for every threshold in the grid."""
    return [threshold_metrics(y_true, y_score, t) for t in thresholds]


def select_threshold(rows: list[dict], metric: str = SELECTION_METRIC) -> dict:
    """Pick the best row by ``metric``; tie-break toward recall, then threshold."""
    return max(rows, key=lambda r: (r[metric], r["recall"], r["threshold"]))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def run_analysis(verbose: bool = True, write: bool = True) -> dict:
    pipeline, metadata = train_model.load_model()
    df = data_loader.load_dataset()
    X, y = data_loader.split_features_target(df)

    test_idx = list(metadata["split"]["test_indices"])
    test_set = set(test_idx)
    pool_idx = [i for i in X.index if i not in test_set]

    fit_idx, val_idx = train_test_split(
        pool_idx,
        test_size=VAL_SIZE,
        random_state=VAL_RANDOM_STATE,
        stratify=y.loc[pool_idx],
    )

    clone = train_model.build_pipeline()
    clone.fit(X.loc[fit_idx], y.loc[fit_idx])

    val_score = clone.predict_proba(X.loc[val_idx])[:, 1]
    val_true = y.loc[val_idx].to_numpy()

    rows = sweep(val_true, val_score)
    best = select_threshold(rows)
    at_current = threshold_metrics(val_true, val_score, CURRENT_THRESHOLD)

    # Held-out reference — computed with the DEPLOYED model, never used to select.
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx].to_numpy()
    test_score = pipeline.predict_proba(X_test)[:, 1]
    test_at_current = threshold_metrics(y_test, test_score, CURRENT_THRESHOLD)
    test_at_selected = threshold_metrics(y_test, test_score, best["threshold"])

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "validation-set threshold selection (no test-set tuning)",
        "selection_metric": SELECTION_METRIC,
        "model_version": metadata["model_version"],
        "validation": {
            "random_state": VAL_RANDOM_STATE,
            "val_size": VAL_SIZE,
            "n_fit": len(fit_idx),
            "n_val": len(val_idx),
            "val_positives": int(val_true.sum()),
            "val_indices": [int(i) for i in val_idx],
            "current_threshold": CURRENT_THRESHOLD,
            "metrics_at_current": at_current,
            "selected_threshold": best["threshold"],
            "metrics_at_selected": best,
            "sweep": rows,
        },
        "held_out_test_reference": {
            "note": (
                "Reported for comparison only; NOT used to select the threshold. "
                "artifacts/metrics.json is preserved unchanged."
            ),
            "n_test": len(test_idx),
            "metrics_at_current_0.5": test_at_current,
            "metrics_at_selected": test_at_selected,
        },
        "policy_note": (
            "The app's risk bands (LOW < 0.30, HIGH >= 0.70) are a separate "
            "documented policy; this analysis informs, but does not automatically "
            "change, those bands."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if write:
        ANALYSIS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if verbose:
        print(json.dumps({
            "selected_threshold": result["validation"]["selected_threshold"],
            "validation_at_current": at_current,
            "validation_at_selected": best,
            "held_out_at_current": test_at_current,
            "held_out_at_selected": test_at_selected,
        }, indent=2))
        print(f"\nWrote {ANALYSIS_PATH}")

    return result


if __name__ == "__main__":
    run_analysis()
