"""Inference helper: turn a SensorReading into a model failure probability.

This is the bridge between the trained model and the decision engine. The
model output is a *prediction*; the decision engine decides what to do with it.
"""

from __future__ import annotations

import pandas as pd

from src import data_loader, model_store
from src.decision_engine import SensorReading

# Maps SensorReading attribute -> dataset feature column name.
_FIELD_TO_COLUMN = {
    "type": "Type",
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}


def _load() -> tuple[object, dict]:
    # Delegates to the portable loader (loads an artifact, or trains + caches).
    return model_store.get_model()


def reading_to_frame(reading: SensorReading) -> pd.DataFrame:
    row = {col: getattr(reading, attr) for attr, col in _FIELD_TO_COLUMN.items()}
    return pd.DataFrame([row], columns=data_loader.FEATURE_COLUMNS)


def predict_probability(reading: SensorReading) -> float:
    """Return the model's failure probability for a single reading."""
    pipeline, _ = _load()
    frame = reading_to_frame(reading)
    return float(pipeline.predict_proba(frame)[0, 1])


def model_version() -> str:
    _, metadata = _load()
    return metadata["model_version"]


if __name__ == "__main__":
    example = SensorReading(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1380,
        torque=60.0,
        tool_wear=210,
        type="L",
        machine_id="demo-01",
    )
    print(f"model version    : {model_version()}")
    print(f"failure prob     : {predict_probability(example):.4f}")
