"""Deterministic assessment orchestration for ForgeWise.

This module is the transparent, testable pipeline that sits between the UI and
the underlying ML / evidence logic. It does not replace either; it sequences
them and records each stage's status explicitly so nothing is hidden.

Stages (in order):

1. ``input_validation`` — are the supplied values structurally usable?
2. ``inference``        — run the existing model (``src.predict``).
3. ``evidence``         — completeness & freshness (``src.decision_engine``).
4. ``decision_rules``   — explicit, documented decision mapping.
5. ``checks``           — permitted rule-based follow-up checks.

Report generation (Gemini or offline template) is a *separate*, optional step
(``src.reporting``) and is never required for an assessment.

The layer is deliberately plain Python. There is no agent framework: the flow
is linear, deterministic, and every stage can be unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from src import predict
from src.data_loader import VALID_TYPES
from src.decision_engine import (
    STATUS_COMPLETE,
    STATUS_INCONCLUSIVE,
    STATUS_INCOMPLETE,
    STATUS_NO_MODEL,
    STATUS_STALE,
    Decision,
    SensorReading,
    make_decision,
)

# Stage status codes (kept as plain strings for easy serialisation).
STAGE_OK = "ok"
STAGE_WARNING = "warning"
STAGE_ERROR = "error"
STAGE_SKIPPED = "skipped"

# Numeric fields that must be finite when present (``None`` means "missing",
# which the evidence engine handles as INCOMPLETE — not an input error).
_NUMERIC_FIELDS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]


@dataclass
class Stage:
    key: str
    title: str
    status: str
    detail: str = ""


@dataclass
class Assessment:
    reading: SensorReading
    model_version: str
    probability: Optional[float]
    decision: Optional[Decision]
    stages: list[Stage]
    input_errors: list[str]
    input_warnings: list[str]
    ok: bool

    def stage(self, key: str) -> Optional[Stage]:
        return next((s for s in self.stages if s.key == key), None)

    def to_payload(self) -> dict:
        return {
            "reading": asdict(self.reading),
            "decision": asdict(self.decision) if self.decision is not None else None,
        }


def validate_inputs(reading: SensorReading) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for the supplied reading.

    * errors — block assessment: unknown product type, non-finite values.
    * warnings — assessment proceeds: negative physical values (the evidence
      engine will also flag these as out-of-range).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if str(reading.type).upper() not in VALID_TYPES:
        errors.append(
            f"Unknown product type {reading.type!r}; expected one of {sorted(VALID_TYPES)}."
        )

    for name in _NUMERIC_FIELDS:
        value = getattr(reading, name)
        if value is None:
            continue  # handled as missing evidence, not an input error
        if not math.isfinite(value):
            errors.append(f"{name} must be a finite number (got {value!r}).")
        elif value < 0:
            warnings.append(f"{name} is negative ({value}); expected a non-negative value.")

    return errors, warnings


def human_review_required(decision: Decision) -> tuple[bool, list[str]]:
    """Whether a human must verify before any action, and why."""
    reasons: list[str] = []
    if decision.evidence_status == STATUS_INCOMPLETE:
        reasons.append("Required sensor fields are missing.")
    if decision.evidence_status == STATUS_STALE:
        reasons.append("The reading is older than the freshness limit.")
    if decision.evidence_status == STATUS_NO_MODEL:
        reasons.append("No model output is available.")
    if decision.evidence_status == STATUS_INCONCLUSIVE:
        reasons.append("The model output is inconclusive.")
    if decision.risk_level == "HIGH":
        reasons.append("The model estimates HIGH failure risk.")
    return bool(reasons), reasons


def run_assessment(
    reading: SensorReading,
    model_version: str,
    predict_fn: Optional[Callable[[SensorReading], float]] = None,
    now=None,
) -> Assessment:
    """Execute the deterministic pipeline and return a stage-annotated result."""
    predict_fn = predict_fn or predict.predict_probability
    stages: list[Stage] = []

    errors, warnings = validate_inputs(reading)
    stages.append(
        Stage(
            "input_validation",
            "Input validation",
            STAGE_ERROR if errors else (STAGE_WARNING if warnings else STAGE_OK),
            "; ".join(errors or warnings) or "All supplied values are structurally valid.",
        )
    )

    if errors:
        stages.append(
            Stage("inference", "Model inference", STAGE_SKIPPED, "Blocked by input errors.")
        )
        stages.append(
            Stage("evidence", "Evidence completeness & freshness", STAGE_SKIPPED, "Not reached.")
        )
        stages.append(Stage("decision_rules", "Decision rules", STAGE_SKIPPED, "Not reached."))
        stages.append(Stage("checks", "Follow-up checks", STAGE_SKIPPED, "Not reached."))
        return Assessment(
            reading=reading,
            model_version=model_version,
            probability=None,
            decision=None,
            stages=stages,
            input_errors=errors,
            input_warnings=warnings,
            ok=False,
        )

    # 2. Inference — failures are surfaced, not swallowed. A missing model
    # output is a valid state (NO_MODEL_OUTPUT), handled by the evidence engine.
    probability: Optional[float]
    try:
        probability = float(predict_fn(reading))
        stages.append(
            Stage(
                "inference",
                "Model inference",
                STAGE_OK,
                f"Predicted failure probability = {probability:.4f} (model {model_version}).",
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        probability = None
        stages.append(Stage("inference", "Model inference", STAGE_ERROR, str(exc)))

    # 3–5. Evidence + decision rules + checks (single deterministic call; the
    # decision engine owns the precedence rules).
    decision = make_decision(
        reading, model_probability=probability, model_version=model_version, now=now
    )
    stages.append(
        Stage(
            "evidence",
            "Evidence completeness & freshness",
            STAGE_WARNING if decision.evidence_gaps else STAGE_OK,
            f"{decision.evidence_status} — {len(decision.evidence_gaps)} gap(s) detected.",
        )
    )
    stages.append(
        Stage(
            "decision_rules",
            "Decision rules",
            STAGE_OK,
            f"Mapped to {decision.decision} (risk {decision.risk_level}, confidence {decision.confidence}).",
        )
    )
    stages.append(
        Stage(
            "checks",
            "Follow-up checks",
            STAGE_OK if decision.recommended_checks else STAGE_WARNING,
            f"{len(decision.recommended_checks)} permitted check(s) selected.",
        )
    )

    return Assessment(
        reading=reading,
        model_version=model_version,
        probability=probability,
        decision=decision,
        stages=stages,
        input_errors=errors,
        input_warnings=warnings,
        ok=True,
    )


__all__ = [
    "Assessment",
    "Stage",
    "STAGE_OK",
    "STAGE_WARNING",
    "STAGE_ERROR",
    "STAGE_SKIPPED",
    "validate_inputs",
    "human_review_required",
    "run_assessment",
    "STATUS_COMPLETE",
    "STATUS_INCOMPLETE",
    "STATUS_STALE",
    "STATUS_NO_MODEL",
    "STATUS_INCONCLUSIVE",
]
