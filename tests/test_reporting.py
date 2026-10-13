"""Tests for report generation and the Gemini-unavailable fallback."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from src import reporting
from src.decision_engine import SensorReading, make_decision


def _payload() -> dict:
    reading = SensorReading(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1380.0,
        torque=60.0,
        tool_wear=210.0,
        type="L",
        machine_id="demo-01",
    )
    decision = make_decision(reading, model_probability=0.82, model_version="fw-test")
    return {"reading": asdict(reading), "decision": asdict(decision)}


def _no_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_generate_report_without_key_uses_offline_template(monkeypatch):
    _no_keys(monkeypatch)
    text, source = reporting.generate_report(_payload())
    assert source == "template"
    assert "ForgeWise Inspection Report" in text
    # Report must separate prediction from evidence.
    assert "PREDICTION" in text
    assert "Verified evidence" in text


def test_generate_report_no_fallback_raises_without_key(monkeypatch):
    _no_keys(monkeypatch)
    with pytest.raises(RuntimeError):
        reporting.generate_report(_payload(), allow_fallback=False)


def test_template_report_includes_model_version_and_checks():
    text = reporting.render_template_report(_payload())
    assert "fw-test" in text
    assert "Recommended checks" in text


def test_build_prompt_is_grounded_in_payload():
    prompt = reporting.build_prompt(_payload())
    assert "PREDICTION" in prompt
    assert "decision" in prompt
    assert "reading" in prompt
