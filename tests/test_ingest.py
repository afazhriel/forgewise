"""Tests for the shared equipment-ingestion normaliser (src/ingest.py)."""

from __future__ import annotations

import pytest

from src import ingest
from src.decision_engine import SensorReading


def test_aliases_and_numeric_coercion():
    payload = {
        "Asset": "cnc-07",
        "Product Type": "m",
        "Air temperature [K]": "298.1",
        "process temperature [k]": 308.6,
        "rpm": "1500",
        "Torque [Nm]": 40,
        "wear": 50,
    }
    result = ingest.normalize_payload(payload)
    assert result.ok
    assert result.fields["machine_id"] == "cnc-07"
    assert result.fields["type"] == "M"
    assert result.fields["air_temperature"] == pytest.approx(298.1)
    assert result.fields["rotational_speed"] == pytest.approx(1500.0)
    assert result.fields["torque"] == pytest.approx(40.0)


def test_default_type_adds_warning():
    result = ingest.normalize_payload({"torque": 40})
    assert result.ok
    assert result.fields["type"] == "M"
    assert any("defaulted to 'M'" in w for w in result.warnings)


def test_unknown_product_type_is_error():
    result = ingest.normalize_payload({"type": "X", "torque": 40})
    assert not result.ok
    assert any("Unknown product type" in e for e in result.errors)


def test_non_numeric_value_is_error():
    result = ingest.normalize_payload({"torque": "not-a-number"})
    assert not result.ok
    assert result.fields["torque"] is None
    assert any("torque" in e for e in result.errors)


def test_missing_field_becomes_none_and_is_reported():
    result = ingest.normalize_payload({"air_temperature": 298.1})
    assert result.ok
    assert result.fields["torque"] is None
    assert "torque" in result.missing_fields


def test_unknown_keys_are_reported_not_rejected():
    result = ingest.normalize_payload({"torque": 40, "wibble": 1, "wobble": 2})
    assert result.ok
    assert set(result.unknown_keys) == {"wibble", "wobble"}


def test_non_dict_payload_is_rejected():
    result = ingest.normalize_payload(["not", "a", "dict"])
    assert not result.ok


def test_to_reading_builds_sensor_reading():
    reading = ingest.to_reading(
        {"machine_id": "cnc-1", "type": "L", "air_temperature": 300, "torque": 40}
    )
    assert isinstance(reading, SensorReading)
    assert reading.machine_id == "cnc-1"
    assert reading.type == "L"
    assert reading.torque == pytest.approx(40.0)


def test_to_reading_raises_on_errors():
    with pytest.raises(ValueError):
        ingest.to_reading({"type": "Z"})


def test_payload_schema_lists_aliases():
    schema = ingest.payload_schema()
    assert "air_temperature" in schema
    assert "rpm" in schema
