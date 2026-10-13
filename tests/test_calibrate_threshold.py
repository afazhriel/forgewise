"""Tests for validation-set threshold calibration.

The pure helpers are tested directly. The integration test exercises the real
analysis but is skipped unless a local model artifact or dataset is available,
and it never mutates committed artifacts (``write=False``).
"""

from __future__ import annotations

import pytest

from src import calibrate_threshold as ct
from src import data_loader, train_model


def test_threshold_metrics_basic():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    m = ct.threshold_metrics(y_true, y_score, 0.5)
    assert m["threshold"] == 0.5
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 1,
        "true_positive": 1,
    }


def test_threshold_metrics_all_positive_at_zero():
    m = ct.threshold_metrics([0, 1], [0.2, 0.9], 0.0)
    assert m["recall"] == 1.0
    assert m["confusion_matrix"]["false_positive"] == 1


def test_sweep_covers_grid():
    rows = ct.sweep([0, 1], [0.1, 0.9])
    assert [r["threshold"] for r in rows] == ct.THRESHOLD_GRID
    assert len(rows) == len(ct.THRESHOLD_GRID)


def test_select_threshold_maximises_metric():
    rows = [
        {"threshold": 0.5, "precision": 0.9, "recall": 0.3, "f1": 0.45},
        {"threshold": 0.2, "precision": 0.6, "recall": 0.8, "f1": 0.70},
    ]
    assert ct.select_threshold(rows)["threshold"] == 0.2


def test_select_threshold_tie_breaks_on_recall_then_threshold():
    rows = [
        {"threshold": 0.4, "precision": 0.8, "recall": 0.5, "f1": 0.6},
        {"threshold": 0.6, "precision": 0.8, "recall": 0.5, "f1": 0.6},
        {"threshold": 0.3, "precision": 0.7, "recall": 0.7, "f1": 0.6},
    ]
    # Equal F1 -> higher recall wins; then higher threshold as final tie-break.
    assert ct.select_threshold(rows)["threshold"] == 0.3


@pytest.mark.skipif(
    not (train_model.MODEL_PATH.exists() or data_loader.CSV_PATH.exists()),
    reason="Threshold analysis needs a model artifact or the local dataset.",
)
def test_run_analysis_preserves_held_out_metrics():
    """The held-out test reference at 0.5 must match the committed metrics."""
    result = ct.run_analysis(verbose=False, write=False)

    selected = result["validation"]["selected_threshold"]
    assert selected in ct.THRESHOLD_GRID

    ref = result["held_out_test_reference"]
    at_current = ref["metrics_at_current_0.5"]
    # Held-out metrics computed here must equal the committed metrics.json.
    import json

    committed = json.loads(
        (data_loader.PROJECT_ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8")
    )
    assert at_current["precision"] == pytest.approx(committed["precision"], abs=1e-6)
    assert at_current["recall"] == pytest.approx(committed["recall"], abs=1e-6)
    assert at_current["confusion_matrix"] == committed["confusion_matrix"]
