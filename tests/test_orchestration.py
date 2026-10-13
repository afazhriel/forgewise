"""Tests for the deterministic orchestration pipeline (src/orchestration.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src import decision_engine as de
from src import orchestration as orch


def _reading(**overrides) -> de.SensorReading:
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


# --- input validation -------------------------------------------------------
def test_validate_inputs_accepts_nominal():
    errors, warnings = orch.validate_inputs(_reading())
    assert errors == []
    assert warnings == []


def test_validate_inputs_flags_unknown_type():
    errors, _ = orch.validate_inputs(_reading(type="Z"))
    assert any("product type" in e.lower() for e in errors)


def test_validate_inputs_flags_non_finite():
    errors, _ = orch.validate_inputs(_reading(torque=float("inf")))
    assert any("finite" in e for e in errors)


def test_validate_inputs_warns_on_negative():
    errors, warnings = orch.validate_inputs(_reading(torque=-5.0))
    assert errors == []
    assert any("negative" in w for w in warnings)


def test_validate_inputs_treats_missing_as_not_an_error():
    errors, warnings = orch.validate_inputs(_reading(torque=None))
    assert errors == []
    assert warnings == []


# --- orchestration stages ---------------------------------------------------
def test_run_assessment_records_all_stages():
    assessment = orch.run_assessment(_reading(), model_version="fw-test", predict_fn=lambda r: 0.1)
    assert assessment.ok is True
    keys = [s.key for s in assessment.stages]
    assert keys == ["input_validation", "inference", "evidence", "decision_rules", "checks"]
    assert assessment.probability == 0.1
    assert assessment.decision.decision == "MONITOR"


def test_run_assessment_blocks_on_input_error():
    assessment = orch.run_assessment(_reading(type="Z"), model_version="fw", predict_fn=lambda r: 0.1)
    assert assessment.ok is False
    assert assessment.decision is None
    assert assessment.probability is None
    statuses = {s.key: s.status for s in assessment.stages}
    assert statuses["input_validation"] == orch.STAGE_ERROR
    assert statuses["inference"] == orch.STAGE_SKIPPED


def test_run_assessment_surfaces_inference_failure():
    def _boom(_reading):
        raise RuntimeError("model exploded")

    assessment = orch.run_assessment(_reading(), model_version="fw", predict_fn=_boom)
    # Inference failure is surfaced as NO_MODEL_OUTPUT, not swallowed.
    assert assessment.probability is None
    assert assessment.decision.evidence_status == de.STATUS_NO_MODEL
    inference = assessment.stage("inference")
    assert inference.status == orch.STAGE_ERROR
    assert "model exploded" in inference.detail


def test_run_assessment_uses_default_probability_for_missing_evidence():
    assessment = orch.run_assessment(_reading(torque=None), model_version="fw", predict_fn=lambda r: 0.9)
    assert assessment.decision.evidence_status == de.STATUS_INCOMPLETE
    assert assessment.decision.decision == "REQUEST_DATA"


# --- human review -----------------------------------------------------------
def test_human_review_required_for_incomplete_evidence():
    decision = de.make_decision(_reading(torque=None), model_probability=0.1)
    required, reasons = orch.human_review_required(decision)
    assert required is True
    assert any("missing" in r.lower() for r in reasons)


def test_human_review_required_for_high_risk():
    decision = de.make_decision(_reading(), model_probability=0.95)
    required, reasons = orch.human_review_required(decision)
    assert required is True
    assert any("HIGH" in r for r in reasons)


def test_human_review_not_required_for_clean_low_risk():
    decision = de.make_decision(_reading(), model_probability=0.05)
    required, reasons = orch.human_review_required(decision)
    assert required is False
    assert reasons == []


def test_human_review_required_for_stale():
    old = datetime.now(timezone.utc) - timedelta(minutes=de.STALE_AFTER_MINUTES + 10)
    decision = de.make_decision(_reading(timestamp=old.isoformat()), model_probability=0.05)
    required, reasons = orch.human_review_required(decision)
    assert required is True
    assert any("older" in r.lower() for r in reasons)


def test_assessment_to_payload_shape():
    assessment = orch.run_assessment(_reading(), model_version="fw", predict_fn=lambda r: 0.2)
    payload = assessment.to_payload()
    assert set(payload) == {"reading", "decision"}
    assert payload["decision"]["model_version"] == "fw"
