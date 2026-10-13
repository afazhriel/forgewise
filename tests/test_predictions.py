"""Behavioural tests for model predictions and input sensitivity.

These lock in the observed behaviour: a typical "healthy" reading scores low, a
labelled failure row scores high, and materially changing a sensor input changes
the prediction (the score is recomputed, not cached).

Skipped unless a local model artifact or the dataset is available.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import data_loader, predict, train_model
from src.decision_engine import (
    STATUS_INCOMPLETE,
    SensorReading,
    make_decision,
)

pytestmark = pytest.mark.skipif(
    not (train_model.MODEL_PATH.exists() or data_loader.CSV_PATH.exists()),
    reason="Prediction tests need a model artifact or the local dataset.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reading(torque: float = 40.0, wear: float = 50.0, rpm: float = 1500.0) -> SensorReading:
    return SensorReading(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=rpm,
        torque=torque,
        tool_wear=wear,
        type="M",
        machine_id="test-machine",
        timestamp=_now(),
    )


def _sample_reading(target: int) -> SensorReading:
    df = data_loader.load_dataset()
    row = df[df["Machine failure"] == target].iloc[0]
    return SensorReading(
        air_temperature=float(row["Air temperature [K]"]),
        process_temperature=float(row["Process temperature [K]"]),
        rotational_speed=float(row["Rotational speed [rpm]"]),
        torque=float(row["Torque [Nm]"]),
        tool_wear=float(row["Tool wear [min]"]),
        type=str(row["Type"]),
        machine_id=f"dataset-example-{target}",
        timestamp=_now(),
    )


def test_typical_reading_scores_low():
    p = predict.predict_probability(_reading())
    assert 0.0 <= p < 0.5


def test_labelled_failure_row_scores_high():
    p = predict.predict_probability(_sample_reading(1))
    assert p >= 0.5


def test_labelled_healthy_row_scores_lower_than_failure_row():
    healthy = predict.predict_probability(_sample_reading(0))
    failure = predict.predict_probability(_sample_reading(1))
    assert healthy < failure


def test_changing_input_changes_prediction():
    """A materially higher torque must produce a fresh, higher prediction."""
    low = predict.predict_probability(_reading(torque=40.0))
    high = predict.predict_probability(_reading(torque=70.0))
    assert high > low


def test_missing_field_marks_evidence_incomplete():
    reading = _reading()
    reading.torque = None
    decision = make_decision(reading, model_probability=0.0)
    assert decision.evidence_status == STATUS_INCOMPLETE
    assert decision.decision == "REQUEST_DATA"
    assert any(gap.startswith("MISSING_FIELD:torque") for gap in decision.evidence_gaps)
