"""Evaluate the ForgeWise baseline on the held-out test split.

Uses the exact test indices recorded at training time (in
``models/model_metadata.json``) so the evaluation is reproducible and never
tuned against the test set.

Writes metrics to ``artifacts/metrics.json`` and figures to ``artifacts/``.

Run:  python -m src.evaluate_model
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless / no GUI backend
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import data_loader, train_model  # noqa: E402

ARTIFACTS_DIR = data_loader.PROJECT_ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def evaluate(verbose: bool = True) -> dict:
    pipeline, metadata = train_model.load_model()
    df = data_loader.load_dataset()
    X, _ = data_loader.split_features_target(df)

    test_idx = metadata["split"]["test_indices"]
    X_test = X.loc[test_idx]
    y_test = df.loc[test_idx, data_loader.TARGET].astype(int)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model_version": metadata["model_version"],
        "evaluated_at": metadata["trained_at"],
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "note": (
            "Metrics on the held-out test split only. Class imbalance is heavy "
            "(~3.4% positives); recall and average_precision (PR-AUC) are the "
            "most informative here. No test-set tuning was performed."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _plot_confusion(metrics["confusion_matrix"])
    _plot_roc_pr(y_test, y_prob)

    if verbose:
        print(json.dumps(metrics, indent=2))
        print(f"\nWrote {METRICS_PATH}")
    return metrics


def _plot_confusion(cm: dict) -> None:
    matrix = [
        [cm["true_negative"], cm["false_positive"]],
        [cm["false_negative"], cm["true_positive"]],
    ]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _plot_roc_pr(y_test, y_prob) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (test)")
    axes[0].legend()

    axes[1].plot(rec, prec, label="PR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (test)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "roc_pr_curves.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
