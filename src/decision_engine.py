"""ForgeWise decision engine — the "Missing Evidence Engine".

This module deliberately keeps *model predictions* separate from *verified
evidence* and from *missing evidence*:

* ``[MODEL]``      -> a statistical prediction (not a fact)
* ``[EVIDENCE]``   -> an observed, measured value from the sensor payload
* ``[MISSING]``    -> information the engine needs but does not have

It is pure Python (no pandas / sklearn import) so it can be unit-tested in
isolation and reused as the trust layer on top of any model.

Mechanism heuristics are derived from the *public* AI4I 2020 dataset
documentation (UCI, CC BY 4.0). They are used ONLY to recommend follow-up
checks — never presented as ground truth or as the model's output.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable policy constants (documented design choices, not learned facts).
# ---------------------------------------------------------------------------
RISK_LOW_MAX = 0.30          # probability < 0.30  -> low risk
RISK_HIGH_MIN = 0.70         # probability >= 0.70 -> high risk
INCONCLUSIVE_LOW = 0.30      # model probability band where the model itself
INCONCLUSIVE_HIGH = 0.70     #   is not decisive (evidence is inconclusive)
STALE_AFTER_MINUTES = 60     # a reading older than this is stale evidence

# Plausible operating ranges (sanity envelope observed in the AI4I dataset).
PLAUSIBLE_RANGES = {
    "air_temperature": (250.0, 350.0),
    "process_temperature": (250.0, 380.0),
    "rotational_speed": (0.0, 5000.0),
    "torque": (0.0, 100.0),
    "tool_wear": (0.0, 300.0),
}

# Overstrain (OSF) tool-wear x torque product thresholds by product type,
# per the AI4I dataset description.
OSF_THRESHOLDS = {"L": 11000.0, "M": 12000.0, "H": 13000.0}

REQUIRED_FIELDS = [
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

# Evidence status codes.
STATUS_COMPLETE = "COMPLETE"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_STALE = "STALE"
STATUS_NO_MODEL = "NO_MODEL_OUTPUT"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class SensorReading:
    """A single machine sensor snapshot supplied to the engine."""

    air_temperature: Optional[float]
    process_temperature: Optional[float]
    rotational_speed: Optional[float]
    torque: Optional[float]
    tool_wear: Optional[float]
    type: str = "M"
    machine_id: str = "unknown"
    reading_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: Optional[str] = None  # ISO-8601 UTC string, if known

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if _is_missing(getattr(self, f))]

    def age_minutes(self, now: Optional[datetime] = None) -> Optional[float]:
        if not self.timestamp:
            return None
        now = now or datetime.now(timezone.utc)
        try:
            ts = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() / 60.0


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


# ---------------------------------------------------------------------------
# Mechanism heuristics (recommend checks; NOT the model's prediction)
# ---------------------------------------------------------------------------
def power_watts(torque: float, rotational_speed: float) -> float:
    """Mechanical power in watts from torque (Nm) and speed (rpm)."""
    return torque * (2.0 * math.pi * rotational_speed / 60.0)


def mechanism_flags(reading: SensorReading) -> dict:
    """Return boolean heuristic flags with cited thresholds."""
    flags: dict = {}
    if _is_missing(reading.air_temperature) or _is_missing(reading.process_temperature):
        pass
    else:
        temp_diff = reading.process_temperature - reading.air_temperature
        flags["HDF_heat_dissipation"] = bool(
            (not _is_missing(reading.rotational_speed))
            and temp_diff < 8.6
            and reading.rotational_speed < 1380
        )
    if not _is_missing(reading.torque) and not _is_missing(reading.rotational_speed):
        power = power_watts(reading.torque, reading.rotational_speed)
        flags["PWF_power_failure"] = bool(power < 3500 or power > 9000)
    if not _is_missing(reading.tool_wear):
        flags["TWF_tool_wear"] = bool(200 <= reading.tool_wear <= 240)
        if not _is_missing(reading.torque):
            limit = OSF_THRESHOLDS.get(str(reading.type).upper(), OSF_THRESHOLDS["M"])
            flags["OSF_overstrain"] = bool(reading.tool_wear * reading.torque > limit)
    return {k: v for k, v in flags.items() if v}


def range_violations(reading: SensorReading) -> list[str]:
    """Flag values outside the plausible operating envelope (sanity checks)."""
    out: list[str] = []
    for field_name, (lo, hi) in PLAUSIBLE_RANGES.items():
        value = getattr(reading, field_name)
        if _is_missing(value):
            continue
        if not (lo <= value <= hi):
            out.append(f"{field_name}={value} outside plausible range [{lo}, {hi}]")
    return out


def risk_level(probability: Optional[float]) -> str:
    if probability is None:
        return "UNKNOWN"
    if probability >= RISK_HIGH_MIN:
        return "HIGH"
    if probability < RISK_LOW_MAX:
        return "LOW"
    return "MODERATE"


# ---------------------------------------------------------------------------
# Evidence assessment + decision
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    machine_id: str
    reading_id: str
    model_version: str
    model_probability: Optional[float]
    risk_level: str
    evidence_status: str
    evidence_gaps: list[str]
    mechanism_flags: dict
    range_warnings: list[str]
    decision: str
    recommended_checks: list[str]
    confidence: str
    rationale: list[str]
    created_at: str


def assess_evidence(
    reading: SensorReading,
    model_probability: Optional[float],
    now: Optional[datetime] = None,
) -> tuple[str, list[str], Optional[float]]:
    """Return (status, evidence_gaps, age_minutes).

    Status precedence reflects what blocks a trustworthy decision first:
    missing data > stale data > missing model output > inconclusive model.
    """
    gaps: list[str] = []
    missing = reading.missing_fields()
    age = reading.age_minutes(now)

    for name in missing:
        gaps.append(f"MISSING_FIELD:{name}")
    if age is not None and age > STALE_AFTER_MINUTES:
        gaps.append(f"STALE_READING:{round(age, 1)}min")
    if reading.timestamp is None:
        gaps.append("UNKNOWN_FRESHNESS")
    for violation in range_violations(reading):
        gaps.append(f"RANGE:{violation}")
    if model_probability is None:
        gaps.append("NO_MODEL_OUTPUT")

    if missing:
        status = STATUS_INCOMPLETE
    elif age is not None and age > STALE_AFTER_MINUTES:
        status = STATUS_STALE
    elif model_probability is None:
        status = STATUS_NO_MODEL
    elif INCONCLUSIVE_LOW <= model_probability <= INCONCLUSIVE_HIGH:
        status = STATUS_INCONCLUSIVE
    else:
        status = STATUS_COMPLETE

    return status, gaps, age


def _recommended_checks(
    status: str,
    gaps: list[str],
    flags: dict,
    risk: str,
) -> list[str]:
    checks: list[str] = []

    if status == STATUS_INCOMPLETE:
        for gap in gaps:
            if gap.startswith("MISSING_FIELD:"):
                checks.append(f"Capture missing sensor field: {gap.split(':', 1)[1]}")
    if status == STATUS_STALE:
        checks.append(
            f"Re-capture the sensor snapshot (reading older than {STALE_AFTER_MINUTES} min)."
        )
    if "UNKNOWN_FRESHNESS" in gaps:
        checks.append("Attach a timestamp / data-source tag to the reading.")
    for gap in gaps:
        if gap.startswith("RANGE:"):
            checks.append(f"Verify suspect sensor value — {gap.split(':', 1)[1]}.")

    if "HDF_heat_dissipation" in flags:
        checks.append(
            "Verify heat-dissipation path (airflow, heat sink, coolant) and "
            "confirm the 8.6 K air/process temperature differential."
        )
    if "PWF_power_failure" in flags:
        checks.append(
            "Verify drive power supply and load; measure motor current "
            "(power outside the 3500-9000 W band)."
        )
    if "OSF_overstrain" in flags:
        checks.append(
            "Inspect for overstrain; review torque load and tool-wear limits "
            "for this product type."
        )
    if "TWF_tool_wear" in flags:
        checks.append(
            "Inspect tool condition and schedule a tool change "
            "(tool wear in the 200-240 min band)."
        )

    if status in (STATUS_COMPLETE, STATUS_INCONCLUSIVE) and risk == "HIGH":
        checks.append("Perform a manual inspection before the next production run.")
    if not checks and risk == "LOW":
        checks.append("No action required; continue routine monitoring.")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for check in checks:
        if check not in seen:
            seen.add(check)
            unique.append(check)
    return unique


def make_decision(
    reading: SensorReading,
    model_probability: Optional[float],
    model_version: str = "unknown",
    now: Optional[datetime] = None,
) -> Decision:
    """Combine model output and evidence into a traced decision record."""
    status, gaps, age = assess_evidence(reading, model_probability, now)
    risk = risk_level(model_probability)
    flags = mechanism_flags(reading)

    if reading.missing_fields():
        decision = "REQUEST_DATA"
    elif status == STATUS_STALE:
        decision = "RECAPTURE_READING"
    elif risk == "HIGH":
        decision = "ESCALATE_INSPECTION"
    elif risk == "MODERATE" or status == STATUS_INCONCLUSIVE:
        decision = "SCHEDULE_CHECK"
    elif flags:
        # Model says low risk but a documented mechanism heuristic disagrees.
        decision = "SCHEDULE_CHECK"
    else:
        decision = "MONITOR"

    checks = _recommended_checks(status, gaps, flags, risk)

    if status in (STATUS_COMPLETE, STATUS_INCONCLUSIVE) and risk in ("HIGH", "LOW"):
        confidence = "HIGH"
    elif status == STATUS_COMPLETE:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    rationale: list[str] = []
    if model_probability is not None:
        rationale.append(
            f"[MODEL] Predicted failure probability = {model_probability:.3f} "
            f"(version {model_version}) -> {risk} risk. Prediction, not verified fact."
        )
    else:
        rationale.append("[MODEL] No model output available for this reading.")

    missing = reading.missing_fields()
    if missing:
        rationale.append(f"[MISSING] Required fields absent: {', '.join(missing)}.")
    else:
        rationale.append("[EVIDENCE] All required sensor fields are present.")

    if age is None:
        rationale.append("[MISSING] Reading has no timestamp; freshness unknown.")
    elif age > STALE_AFTER_MINUTES:
        rationale.append(
            f"[EVIDENCE] Reading is {age:.1f} min old (> {STALE_AFTER_MINUTES} min) -> stale."
        )
    else:
        rationale.append(f"[EVIDENCE] Reading age {age:.1f} min -> fresh.")

    if flags:
        for name in flags:
            rationale.append(f"[EVIDENCE] Mechanism heuristic triggered: {name}.")
    else:
        rationale.append("[EVIDENCE] No documented failure-mechanism heuristic triggered.")

    if model_probability is not None and INCONCLUSIVE_LOW <= model_probability <= INCONCLUSIVE_HIGH:
        rationale.append(
            "[MODEL] Probability lies in the inconclusive band "
            f"[{INCONCLUSIVE_LOW}, {INCONCLUSIVE_HIGH}] -> model is not decisive."
        )

    return Decision(
        machine_id=reading.machine_id,
        reading_id=reading.reading_id,
        model_version=model_version,
        model_probability=model_probability,
        risk_level=risk,
        evidence_status=status,
        evidence_gaps=gaps,
        mechanism_flags=flags,
        range_warnings=range_violations(reading),
        decision=decision,
        recommended_checks=checks,
        confidence=confidence,
        rationale=rationale,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
def log_decision(
    decision: Decision,
    reading: SensorReading,
    log_path: str | Path = "artifacts/decision_log.jsonl",
    extra: dict | None = None,
) -> Path:
    """Append a full audit record (inputs, output, version, timestamp).

    ``extra`` lets the orchestration layer attach provenance such as the report
    mode (``gemini`` / ``template``) without changing the decision schema.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": decision.created_at,
        "reading": asdict(reading),
        "decision": asdict(decision),
    }
    if extra:
        record["extra"] = extra
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


if __name__ == "__main__":
    demo = SensorReading(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1380,
        torque=60.0,
        tool_wear=210,
        type="L",
        machine_id="demo-01",
    )
    result = make_decision(demo, model_probability=0.82)
    print(json.dumps(asdict(result), indent=2))
