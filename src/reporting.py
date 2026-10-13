"""Traceable inspection report generation.

Two paths:

* ``render_template_report`` — deterministic, offline, no API key required.
  Always available and used as the fallback.
* ``generate_report`` — optional Google Gemini path. Requires the
  ``google-genai`` SDK and an API key supplied via environment variable
  (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``). The key is read from the
  environment and never hardcoded or logged.

Both paths are grounded strictly in the supplied evidence + decision payload;
the model is instructed not to invent sensor values or claim verified facts.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are a maintenance decision-support assistant. You write short, "
    "traceable inspection reports for factory maintenance personnel. STRICT "
    "RULES: (1) Use ONLY the data supplied in the payload; never invent sensor "
    "values, measurements, part numbers, or history. (2) Clearly separate "
    "PREDICTION (model output) from EVIDENCE (measured values) from MISSING "
    "information. (3) If a value is absent, say it is missing; do not guess. "
    "(4) Do not make unsupported claims about real-world accuracy. (5) Always "
    "state the model version and the timestamp. Use these sections: Summary, "
    "Risk assessment, Verified evidence, Missing / stale evidence, Recommended "
    "checks, Traceability."
)


def _api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def build_prompt(payload: dict) -> str:
    """Deterministic, grounded prompt describing the evidence + decision."""
    lines = [
        "Produce an inspection report from the following machine-assessment payload.",
        "Treat model_probability strictly as a PREDICTION, not a verified fact.",
        "",
        "PAYLOAD (JSON-like):",
    ]
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(
        "Cite which observations are EVIDENCE vs PREDICTION vs MISSING, then list "
        "the recommended checks. Keep it concise and operational."
    )
    return "\n".join(lines)


def render_template_report(payload: dict) -> str:
    """Deterministic offline report — no LLM, fully traceable."""
    decision = payload.get("decision", {})
    reading = payload.get("reading", {})
    checks = decision.get("recommended_checks", []) or ["None specified."]

    lines = [
        "# ForgeWise Inspection Report (offline template)",
        "",
        f"Machine           : {decision.get('machine_id')}",
        f"Reading ID        : {decision.get('reading_id')}",
        f"Model version     : {decision.get('model_version')}",
        f"Generated at      : {decision.get('created_at')}",
        "",
        "## Summary",
        f"Decision: **{decision.get('decision')}** "
        f"(confidence: {decision.get('confidence')}).",
        "",
        "## Risk assessment (PREDICTION)",
        f"- Model failure probability: {decision.get('model_probability')} "
        f"(a prediction, not a verified fact)",
        f"- Risk level: {decision.get('risk_level')}",
        f"- Evidence status: {decision.get('evidence_status')}",
        "",
        "## Verified evidence (MEASURED)",
    ]
    for key, value in reading.items():
        if value is not None:
            lines.append(f"- {key}: {value}")

    lines += ["", "## Missing / stale evidence"]
    gaps = decision.get("evidence_gaps", [])
    lines += [f"- {gap}" for gap in gaps] or ["- None."]

    lines += ["", "## Recommended checks"]
    lines += [f"- {check}" for check in checks]

    lines += ["", "## Traceability"]
    for item in decision.get("rationale", []):
        lines.append(f"- {item}")

    return "\n".join(lines)


def generate_report(
    payload: dict,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    allow_fallback: bool = True,
) -> tuple[str, str]:
    """Return ``(report_text, source)`` where source is ``gemini`` or ``template``.

    Falls back to the deterministic template when no key/SDK is available and
    ``allow_fallback`` is True.
    """
    key = _api_key(api_key)
    if not key:
        if allow_fallback:
            return render_template_report(payload), "template"
        raise RuntimeError("No Gemini API key found (set GEMINI_API_KEY).")

    try:
        from google import genai  # lazy import: SDK is optional
    except ImportError:
        if allow_fallback:
            return render_template_report(payload), "template"
        raise RuntimeError(
            "google-genai is not installed. Run: pip install google-genai"
        )

    client = genai.Client(api_key=key)
    prompt = build_prompt(payload)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.2,
        },
    )
    return (response.text or "").strip(), "gemini"


if __name__ == "__main__":
    from dataclasses import asdict

    from src.decision_engine import SensorReading, make_decision

    reading = SensorReading(
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1380,
        torque=60.0,
        tool_wear=210,
        type="L",
        machine_id="demo-01",
    )
    decision = make_decision(reading, model_probability=0.82)
    payload = {"reading": asdict(reading), "decision": asdict(decision)}
    text, source = generate_report(payload)
    print(f"source: {source}\n")
    print(text)
