"""Tests for portable model acquisition (src/model_store.py)."""

from __future__ import annotations

import pytest

from src import data_loader, model_store, predict, train_model
from src.decision_engine import SensorReading


def _have_local_model_source() -> bool:
    return train_model.MODEL_PATH.exists() or data_loader.CSV_PATH.exists()


# --- artifact detection -----------------------------------------------------
def test_model_artifact_exists_reflects_files(tmp_path, monkeypatch):
    model = tmp_path / "model.joblib"
    meta = tmp_path / "model_metadata.json"
    monkeypatch.setattr(train_model, "MODEL_PATH", model)
    monkeypatch.setattr(train_model, "METADATA_PATH", meta)

    assert model_store.model_artifact_exists() is False
    model.write_bytes(b"x")
    assert model_store.model_artifact_exists() is False  # metadata still missing
    meta.write_text("{}", encoding="utf-8")
    assert model_store.model_artifact_exists() is True


# --- bootstrap behaviour (mocked, no network) -------------------------------
def test_bootstrap_loads_existing_artifact(monkeypatch):
    monkeypatch.setattr(model_store, "model_artifact_exists", lambda: True)
    monkeypatch.setattr(
        model_store.train_model, "load_model", lambda *a, **k: ("LOADED", {"model_version": "x"})
    )

    def _boom(*a, **k):
        raise AssertionError("train() must not run when an artifact exists")

    monkeypatch.setattr(model_store.train_model, "train", _boom)
    model_store.clear_cache()
    try:
        assert model_store.get_model() == ("LOADED", {"model_version": "x"})
    finally:
        model_store.clear_cache()


def test_bootstrap_trains_when_artifact_missing(monkeypatch):
    seen = {}
    monkeypatch.setattr(model_store, "model_artifact_exists", lambda: False)

    def _fake_train(verbose=True):
        seen["trained"] = True
        return ("TRAINED", {"model_version": "t"})

    monkeypatch.setattr(model_store.train_model, "train", _fake_train)
    model_store.clear_cache()
    try:
        assert model_store.get_model() == ("TRAINED", {"model_version": "t"})
    finally:
        model_store.clear_cache()
    assert seen.get("trained") is True


def test_dataset_unavailable_raises_model_unavailable(monkeypatch):
    monkeypatch.setattr(model_store, "model_artifact_exists", lambda: False)

    def _fail(verbose=True):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(model_store.train_model, "train", _fail)
    model_store.clear_cache()
    try:
        with pytest.raises(model_store.ModelUnavailable) as err:
            model_store.get_model()
    finally:
        model_store.clear_cache()
    assert "UCI" in str(err.value)
    assert "network failure" in str(err.value)


def test_corrupt_artifact_raises_model_unavailable(monkeypatch):
    monkeypatch.setattr(model_store, "model_artifact_exists", lambda: True)

    def _fail(*a, **k):
        raise ValueError("bad pickle")

    monkeypatch.setattr(model_store.train_model, "load_model", _fail)
    model_store.clear_cache()
    try:
        with pytest.raises(model_store.ModelUnavailable):
            model_store.get_model()
    finally:
        model_store.clear_cache()


# --- end-to-end (uses local artifact or dataset; no network) ----------------
@pytest.mark.skipif(not _have_local_model_source(), reason="needs model artifact or dataset")
def test_get_model_returns_usable_pipeline():
    model_store.clear_cache()
    try:
        pipeline, metadata = model_store.get_model()
    finally:
        model_store.clear_cache()
    assert hasattr(pipeline, "predict_proba")
    assert "model_version" in metadata


@pytest.mark.skipif(not _have_local_model_source(), reason="needs model artifact or dataset")
def test_normal_prediction_flow_probability_in_range():
    reading = SensorReading(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1500.0,
        torque=40.0,
        tool_wear=50.0,
        type="M",
    )
    probability = predict.predict_probability(reading)
    assert 0.0 <= probability <= 1.0


def test_invalid_input_still_builds_a_frame():
    # Nonsensical values and an unknown product type must not crash conversion.
    reading = SensorReading(
        air_temperature=-10.0,
        process_temperature=-20.0,
        rotational_speed=-5.0,
        torque=-3.0,
        tool_wear=-1.0,
        type="Z",
    )
    frame = predict.reading_to_frame(reading)
    assert frame.shape == (1, len(data_loader.FEATURE_COLUMNS))
    assert frame.iloc[0]["Type"] == "Z"
