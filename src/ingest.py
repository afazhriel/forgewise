"""Equipment ingestion: normalise external payloads into sensor readings.

Physical manufacturing equipment (via a gateway, SCADA export, PLC poller, or an
HTTP/MQTT bridge) sends a JSON object. This module maps its keys onto ForgeWise's
canonical sensor fields, coerces types, and reports problems — the single shared
validator used by every integration template in ``integrations/`` and by the app's
**Equipment Integration** page.

It is deliberately pure (no I/O, no framework), so it is fully unit-testable and
safe to reuse from any transport. A missing field is *not* an error: it becomes
``None`` and the Missing Evidence Engine treats it as missing evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.data_loader import VALID_TYPES
from src.decision_engine import SensorReading

NUMERIC_FIELDS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

# Canonical field -> accepted key aliases (compared case-insensitively).
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "type": ("type", "product_type", "product type", "product"),
    "air_temperature": (
        "air_temperature",
        "air temperature [k]",
        "air temperature",
        "air_temp",
        "airtemp",
    ),
    "process_temperature": (
        "process_temperature",
        "process temperature [k]",
        "process temperature",
        "process_temp",
        "processtemp",
    ),
    "rotational_speed": (
        "rotational_speed",
        "rotational speed [rpm]",
        "rotational speed",
        "rpm",
        "speed",
    ),
    "torque": ("torque", "torque [nm]", "torque_nm", "torque (nm)"),
    "tool_wear": ("tool_wear", "tool wear [min]", "tool wear", "wear"),
    "machine_id": ("machine_id", "machine id", "asset", "asset_id", "device_id"),
}


@dataclass
class NormalizedPayload:
    fields: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_keys: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def missing_fields(self) -> list[str]:
        return [name for name in NUMERIC_FIELDS if self.fields.get(name) is None]


def normalize_payload(payload: dict) -> NormalizedPayload:
    """Map an external payload onto canonical sensor fields.

    Returns field values plus any ``errors`` (blocking) and ``warnings``
    (non-blocking). Unknown keys are reported, not rejected.
    """
    if not isinstance(payload, dict):
        return NormalizedPayload({}, errors=["Payload must be a JSON object."])

    lower_map: dict[str, Any] = {}
    for key, value in payload.items():
        lower_map[str(key).strip().lower()] = value

    fields: dict[str, Any] = {}
    matched: set[str] = set()
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                fields[canonical] = lower_map[alias]
                matched.add(alias)
                break
        else:
            fields.setdefault(canonical, None)

    unknown = [key for key in lower_map if key not in matched]

    errors: list[str] = []
    warnings: list[str] = []

    raw_type = fields.get("type")
    if raw_type is None or str(raw_type).strip() == "":
        fields["type"] = "M"
        warnings.append("No product type supplied; defaulted to 'M'.")
    else:
        normalized_type = str(raw_type).strip().upper()
        if normalized_type not in VALID_TYPES:
            errors.append(
                f"Unknown product type {raw_type!r}; expected one of {sorted(VALID_TYPES)}."
            )
        fields["type"] = normalized_type

    for name in NUMERIC_FIELDS:
        value = fields.get(name)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            fields[name] = None
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} is not numeric: {value!r}.")
            fields[name] = None
            continue
        if not math.isfinite(numeric):
            errors.append(f"{name} must be finite (got {value!r}).")
            fields[name] = None
            continue
        if numeric < 0:
            warnings.append(f"{name} is negative ({numeric}); expected non-negative.")
        fields[name] = numeric

    machine_id = fields.get("machine_id")
    fields["machine_id"] = str(machine_id).strip() if machine_id not in (None, "") else None

    return NormalizedPayload(fields=fields, errors=errors, warnings=warnings, unknown_keys=unknown)


def to_reading(
    payload: dict,
    machine_id: str | None = None,
    timestamp: str | None = None,
) -> SensorReading:
    """Build a :class:`SensorReading` from an external payload.

    Raises ``ValueError`` if the payload has blocking errors.
    """
    normalized = normalize_payload(payload)
    if not normalized.ok:
        raise ValueError("; ".join(normalized.errors))

    resolved_id = machine_id or normalized.fields.get("machine_id") or "unknown"
    kwargs = {
        name: normalized.fields.get(name) for name in NUMERIC_FIELDS
    }
    return SensorReading(
        type=normalized.fields.get("type", "M"),
        machine_id=resolved_id,
        timestamp=timestamp,
        **kwargs,
    )


def payload_schema() -> str:
    """A human-readable description of the accepted payload for docs/UI."""
    lines = ["Accepted JSON keys (case-insensitive; aliases shown):"]
    for canonical, aliases in FIELD_ALIASES.items():
        lines.append(f"- {canonical}: {', '.join(aliases)}")
    lines.append("")
    lines.append("A missing sensor field is treated as missing evidence, not an error.")
    return "\n".join(lines)


__all__ = [
    "FIELD_ALIASES",
    "NUMERIC_FIELDS",
    "NormalizedPayload",
    "normalize_payload",
    "to_reading",
    "payload_schema",
]
