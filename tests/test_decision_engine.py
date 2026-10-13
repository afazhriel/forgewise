"""Tests for the decision engine: evidence, risk, checks, audit trail."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src import decision_engine as de


def _fresh_reading(**overrides) -> de.SensorReading:
    base = dict(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1500.0,
        torque=40.0,
        tool_wear=50.0,
        type="M",
        machine_id="m-1",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    base.update(overrides)
    return de.SensorReading(**base)


# --- risk levels -----------------------------------------------------------
def test_risk_level_thresholds():
    assert de.risk_level(0.05) == "LOW"
    assert de.risk_level(0.50) == "MODERATE"
    assert de.risk_level(0.90) == "HIGH"
    assert de.risk_level(None) == "UNKNOWN"


# --- evidence completeness --------------------------------------------------
def test_missing_fields_trigger_request_data():
    reading = _fresh_reading(torque=None)
    decision = de.make_decision(reading, model_probability=0.1)
    assert decision.evidence_status == de.STATUS_INCOMPLETE
    assert decision.decision == "REQUEST_DATA"
    assert any("MISSING_FIELD:torque" in g for g in decision.evidence_gaps)
    assert any("torque" in c for c in decision.recommended_checks)


def test_stale_reading_is_flagged_and_rechecked():
    old = (datetime.now(timezone.utc) - timedelta(minutes=de.STALE_AFTER_MINUTES + 30))
    reading = _fresh_reading(timestamp=old.isoformat())
    decision = de.make_decision(reading, model_probability=0.1)
    assert decision.evidence_status == de.STATUS_STALE
    assert decision.decision == "RECAPTURE_READING"
    assert any(g.startswith("STALE_READING") for g in decision.evidence_gaps)


def test_unknown_freshness_is_a_gap():
    reading = _fresh_reading(timestamp=None)
    decision = de.make_decision(reading, model_probability=0.1)
    assert "UNKNOWN_FRESHNESS" in decision.evidence_gaps


def test_no_model_output_is_a_gap():
    reading = _fresh_reading()
    decision = de.make_decision(reading, model_probability=None)
    assert decision.evidence_status == de.STATUS_NO_MODEL
    assert "NO_MODEL_OUTPUT" in decision.evidence_gaps


def test_inconclusive_band_detected():
    reading = _fresh_reading()
    decision = de.make_decision(reading, model_probability=0.5)
    assert decision.evidence_status == de.STATUS_INCONCLUSIVE
    assert any("inconclusive band" in line for line in decision.rationale)


def test_range_violation_is_reported():
    reading = _fresh_reading(rotational_speed=99999.0)
    decision = de.make_decision(reading, model_probability=0.1)
    assert any(g.startswith("RANGE:") for g in decision.evidence_gaps)


# --- mechanism heuristics ---------------------------------------------------
def test_power_watts_formula():
    assert de.power_watts(100.0, 60.0) == __import__("pytest").approx(628.32, rel=1e-3)


def test_hdf_flag():
    flags = de.mechanism_flags(
        _fresh_reading(air_temperature=300.0, process_temperature=307.0, rotational_speed=1200.0)
    )
    assert flags.get("HDF_heat_dissipation") is True


def test_pwf_low_and_high_power():
    low = de.mechanism_flags(_fresh_reading(torque=5.0, rotational_speed=1000.0))
    high = de.mechanism_flags(_fresh_reading(torque=70.0, rotational_speed=2500.0))
    assert low.get("PWF_power_failure") is True
    assert high.get("PWF_power_failure") is True


def test_twf_and_osf_flags():
    flags = de.mechanism_flags(_fresh_reading(tool_wear=210.0, torque=70.0, type="L"))
    assert flags.get("TWF_tool_wear") is True
    assert flags.get("OSF_overstrain") is True


def test_no_flags_for_nominal_reading():
    flags = de.mechanism_flags(_fresh_reading())
    assert flags == {}


# --- decision mapping -------------------------------------------------------
def test_high_risk_escalates_with_checks():
    decision = de.make_decision(
        _fresh_reading(tool_wear=210.0, torque=70.0, type="L"),
        model_probability=0.9,
        model_version="fw-test",
    )
    assert decision.decision == "ESCALATE_INSPECTION"
    assert decision.risk_level == "HIGH"
    assert decision.recommended_checks
    assert decision.model_version == "fw-test"


def test_low_risk_nominal_monitors():
    decision = de.make_decision(_fresh_reading(), model_probability=0.05)
    assert decision.decision == "MONITOR"
    assert decision.evidence_status == de.STATUS_COMPLETE


def test_low_risk_but_mechanism_flag_schedules_check():
    reading = _fresh_reading(tool_wear=210.0, torque=70.0, type="L")
    decision = de.make_decision(reading, model_probability=0.05)
    assert decision.risk_level == "LOW"
    assert decision.mechanism_flags
    assert decision.decision == "SCHEDULE_CHECK"


# --- traceability -----------------------------------------------------------
def test_rationale_separates_model_evidence_missing():
    decision = de.make_decision(_fresh_reading(torque=None), model_probability=0.9)
    tags = {line.split("]")[0] + "]" for line in decision.rationale}
    assert "[MODEL]" in tags
    assert "[EVIDENCE]" in tags
    assert "[MISSING]" in tags


def test_log_decision_writes_jsonl(tmp_path):
    reading = _fresh_reading()
    decision = de.make_decision(reading, model_probability=0.9)
    log_path = tmp_path / "decision_log.jsonl"
    de.log_decision(decision, reading, log_path=log_path)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["decision"]["model_version"] == decision.model_version
    assert record["reading"]["reading_id"] == reading.reading_id
    assert "created_at" in record["decision"]


def test_log_decision_accepts_extra_provenance(tmp_path):
    reading = _fresh_reading()
    decision = de.make_decision(reading, model_probability=0.9)
    log_path = tmp_path / "decision_log.jsonl"
    de.log_decision(decision, reading, log_path=log_path, extra={"report_mode": "template"})
    record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["extra"] == {"report_mode": "template"}
