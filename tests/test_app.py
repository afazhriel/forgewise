"""Streamlit product-shell tests for the ForgeWise app.

Exercises navigation, the explicit Analyze action, sample loading, evidence
states, report generation (offline fallback), and decision history.

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


def _goto(at, page: str):
    at.radio[0].set_value(page).run()
    return at


def _button(at, text: str):
    matches = [b for b in at.button if text.lower() in b.label.lower()]
    assert matches, f"button {text!r} not found; have {[b.label for b in at.button]}"
    return matches[0]


def _metric_value(at, label: str) -> str:
    for metric in at.metric:
        if metric.label == label:
            return metric.value
    raise AssertionError(f"metric {label!r} not found; have {[m.label for m in at.metric]}")


def _analyze(at):
    _button(at, "Analyze reading").click().run()
    return at


# --- startup ----------------------------------------------------------------
def test_app_starts_without_exception():
    at = _run_app()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]


def test_app_exposes_prediction_metrics():
    at = _run_app()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    assert len(at.metric) >= 4


def test_app_shows_simulation_notice():
    at = _run_app()
    warnings = "\n".join(w.value for w in at.warning)
    assert "Demo data only" in warnings


def test_all_pages_render_without_exception():
    at = _run_app()
    for page in ("Machine Analysis", "AI Inspection Report", "Decision History", "Model Diagnostics"):
        _goto(at, page)
        assert len(at.exception) == 0, (page, [str(e.value) for e in at.exception])


# --- analysis flow ----------------------------------------------------------
def test_app_analyze_exposes_the_four_concepts():
    at = _run_app()
    _goto(at, "Machine Analysis")
    _analyze(at)
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    labels = {m.label for m in at.metric}
    assert "Model-estimated probability" in labels
    assert "Risk category" in labels
    assert "Evidence completeness" in labels
    assert "Decision" in labels


def test_app_changing_input_updates_prediction():
    at = _run_app()
    _goto(at, "Machine Analysis")
    _analyze(at)
    before = float(_metric_value(at, "Model-estimated probability"))

    torque = [w for w in at.number_input if w.label == "Torque [Nm]"]
    assert torque, "torque input missing"
    torque[0].set_value(70.0).run()
    _analyze(at)

    after = float(_metric_value(at, "Model-estimated probability"))
    assert after > before


def test_app_loading_failure_example_updates_inputs_and_scores_high():
    at = _run_app()
    _goto(at, "Machine Analysis")
    _button(at, "Load example — failure").click().run()
    _analyze(at)
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    assert float(_metric_value(at, "Model-estimated probability")) >= 0.5


def test_app_missing_field_triggers_request_data():
    at = _run_app()
    _goto(at, "Machine Analysis")
    omit = [s for s in at.selectbox if "missing field" in s.label.lower()]
    assert omit, "missing-field selector not found"
    omit[0].set_value("torque").run()
    _analyze(at)
    assert _metric_value(at, "Evidence completeness") == "INCOMPLETE"
    assert _metric_value(at, "Decision") == "REQUEST_DATA"


def test_app_stale_reading_is_flagged():
    at = _run_app()
    _goto(at, "Machine Analysis")
    slider = [s for s in at.slider if "age" in s.label.lower()]
    assert slider, "age slider not found"
    slider[0].set_value(120).run()
    _analyze(at)
    assert _metric_value(at, "Evidence completeness") == "STALE"


def test_app_healthy_vs_failure_comparison_runs():
    at = _run_app()
    _goto(at, "Machine Analysis")
    _button(at, "comparison").click().run()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    assert at.dataframe, "expected a comparison dataframe"


# --- report page ------------------------------------------------------------
def test_report_page_empty_before_analysis():
    at = _run_app()
    _goto(at, "AI Inspection Report")
    infos = "\n".join(i.value for i in at.info)
    assert "No assessment yet" in infos


def test_app_generate_report_button_uses_offline_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = _run_app()
    _goto(at, "Machine Analysis")
    _analyze(at)
    _goto(at, "AI Inspection Report")
    _button(at, "Generate AI Report").click().run()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    rendered = "\n".join(m.value or "" for m in at.markdown)
    assert "Inspection Report" in rendered
    warnings = "\n".join(w.value for w in at.warning)
    assert "offline deterministic template" in warnings


# --- decision history -------------------------------------------------------
def test_decision_history_lists_a_record_after_analysis():
    at = _run_app()
    _goto(at, "Machine Analysis")
    _analyze(at)
    _goto(at, "Decision History")
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    assert at.dataframe, "expected a decision-history table"
