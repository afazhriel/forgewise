"""Streamlit startup / smoke tests for the ForgeWise app.

Skipped only when neither a local model artifact nor the local dataset is
available (the app would then attempt a network download to train).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import data_loader, train_model

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

pytestmark = pytest.mark.skipif(
    not (train_model.MODEL_PATH.exists() or data_loader.CSV_PATH.exists()),
    reason="AppTest start-up needs a model artifact or the local dataset.",
)


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    return at


def test_app_starts_without_exception():
    at = _run_app()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]


def test_app_exposes_prediction_metrics():
    at = _run_app()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    # probability, risk level, evidence status, decision (plus sidebar metrics).
    assert len(at.metric) >= 4


def test_app_generate_report_button_uses_offline_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = _run_app()
    report_buttons = [b for b in at.button if "report" in b.label.lower()]
    assert report_buttons, "report button missing"
    report_buttons[0].click().run()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    rendered = "\n".join(m.value or "" for m in at.markdown)
    assert "Inspection Report" in rendered
