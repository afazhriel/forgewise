"""Tests for report generation, the Gemini adapter, and graceful fallback.

No live API call is ever made: the provider is injected with a fake client.
"""

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


_GOOD_TEXT = """
## Summary
Nominal assessment.

## Risk assessment
Prediction 0.82 (not verified).

## Verified evidence
Air temperature 298.1.

## Missing / stale evidence
None.

## Recommended checks
Inspect the tool.

## Limitations
Synthetic dataset; estimates only.

## Human verification required
Yes, verify before acting.
"""


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, text=None, error=None):
        self._text = text
        self._error = error
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self._error:
            raise self._error
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text=None, error=None):
        self.models = _FakeModels(text=text, error=error)


# --- offline template -------------------------------------------------------
def test_generate_report_without_key_uses_offline_template(monkeypatch):
    _no_keys(monkeypatch)
    text, source = reporting.generate_report(_payload())
    assert source == "template"
    assert "ForgeWise Inspection Report" in text
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
    assert "Limitations" in text
    assert "Human verification required" in text


def test_build_prompt_is_grounded_in_payload():
    prompt = reporting.build_prompt(_payload())
    assert "PREDICTION" in prompt
    assert "decision" in prompt
    assert "reading" in prompt


def test_generate_report_result_offline_has_no_error():
    result = reporting.generate_report_result(_payload())
    assert result.source == "template"
    assert result.is_fallback is True
    assert "key" in (result.fallback_reason or "").lower()


# --- structured-output validation ------------------------------------------
def test_validate_report_text_accepts_good_output():
    ok, missing = reporting.validate_report_text(_GOOD_TEXT)
    assert ok is True
    assert missing == []


def test_validate_report_text_rejects_empty():
    ok, missing = reporting.validate_report_text("")
    assert ok is False
    assert missing


def test_validate_report_text_rejects_truncated():
    ok, _ = reporting.validate_report_text("## Summary\nToo short.")
    assert ok is False


# --- Gemini adapter (injected client, no network) ---------------------------
def test_provider_success_marked_as_gemini():
    provider = reporting.GeminiProvider("fake-key", client=_FakeClient(text=_GOOD_TEXT))
    result = reporting.generate_report_result(_payload(), provider=provider)
    assert result.source == "gemini"
    assert result.is_fallback is False
    assert provider._client.models.calls == 1


def test_malformed_provider_response_falls_back_with_reason():
    provider = reporting.GeminiProvider("fake-key", client=_FakeClient(text="nonsense"))
    result = reporting.generate_report_result(_payload(), provider=provider)
    assert result.source == "template"
    assert result.error == "Malformed or incomplete Gemini response."
    assert result.missing_sections
    assert "ForgeWise Inspection Report" in result.text


def test_provider_api_error_falls_back_with_reason():
    provider = reporting.GeminiProvider(
        "fake-key", client=_FakeClient(error=RuntimeError("503 unavailable"))
    )
    result = reporting.generate_report_result(_payload(), provider=provider)
    assert result.source == "template"
    assert "503 unavailable" in (result.error or "")
    assert result.fallback_reason


def test_provider_error_no_fallback_raises():
    provider = reporting.GeminiProvider(
        "fake-key", client=_FakeClient(error=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        reporting.generate_report_result(_payload(), provider=provider, allow_fallback=False)


def test_gemini_available_false_without_key(monkeypatch):
    _no_keys(monkeypatch)
    assert reporting.gemini_available() is False


def test_gemini_available_true_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert reporting.gemini_available() is True
