"""Traceable inspection report generation.

Two paths, always clearly labelled:

* ``render_template_report`` — deterministic, offline, no API key required.
  Always available and used as the fallback.
* ``GeminiProvider`` — optional Google Gemini path. Requires the ``google-genai``
  SDK and a key supplied via environment variable (``GEMINI_API_KEY`` or
  ``GOOGLE_API_KEY``) or the app's session field. The key is read from the
  caller and never hardcoded, logged, or written to the decision history.

Both paths are grounded strictly in the supplied evidence + decision payload.
The provider is instructed **not** to invent failure causes, measurements,
inspection history, or maintenance actions, and the output is validated for the
expected section structure before it is accepted. Malformed or empty output
falls back to the deterministic template.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MODEL = "gemini-2.5-flash"

# Section headings we require the LLM output to contain (case-insensitive).
REQUIRED_SECTIONS = (
    "summary",
    "risk",
    "evidence",
    "missing",
    "recommended",
    "limitations",
    "human verification",
)

SYSTEM_INSTRUCTION = (
    "You are a maintenance decision-support assistant. You write short, "
    "traceable inspection reports for factory maintenance personnel. STRICT "
    "RULES: (1) Use ONLY the data supplied in the payload; never invent sensor "
    "values, measurements, part numbers, failure causes, inspection history, or "
    "maintenance actions. (2) Clearly separate PREDICTION (model output) from "
    "EVIDENCE (measured values) from MISSING information. (3) If a value is "
    "absent, say it is missing; do not guess. (4) Do not make unsupported claims "
    "about real-world accuracy; this is a prototype using a synthetic public "
    "dataset. (5) Always state the model version and the timestamp. Produce these "
    "sections as markdown headings: Summary, Risk assessment, Verified evidence, "
    "Missing / stale evidence, Recommended checks, Limitations, Human "
    "verification required, Traceability."
)


@dataclass
class ReportResult:
    """A generated report plus provenance the UI must surface."""

    text: str
    source: str  # "gemini" | "template"
    model_name: Optional[str] = None
    error: Optional[str] = None
    fallback_reason: Optional[str] = None
    missing_sections: list[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return self.source != "gemini"


# ---------------------------------------------------------------------------
# Grounding / prompt
# ---------------------------------------------------------------------------
def _api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def build_prompt(payload: dict) -> str:
    """Deterministic, grounded prompt describing the evidence + decision."""
    lines = [
        "Produce an inspection report from the following machine-assessment payload.",
        "Treat model_probability strictly as a PREDICTION, not a verified fact.",
        "Do not invent failure causes, measurements, inspection history, or actions.",
        "",
        "PAYLOAD (JSON-like):",
    ]
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(
        "Cite which observations are EVIDENCE vs PREDICTION vs MISSING, then list "
        "the recommended checks and any limitations. Keep it concise and operational."
    )
    return "\n".join(lines)


def validate_report_text(text: str) -> tuple[bool, list[str]]:
    """Check the LLM output has the expected, grounded section structure.

    Returns ``(ok, missing_sections)``. Validation is intentionally lenient
    (keyword-based) so it does not reject reasonable provider formatting, but it
    does reject empty or clearly truncated output.
    """
    if not text or len(text.strip()) < 40:
        return False, ["(output too short)", *REQUIRED_SECTIONS]
    lowered = text.lower()
    missing = [section for section in REQUIRED_SECTIONS if section not in lowered]
    # Require the core structure; a couple of incidental headings may vary.
    ok = len(missing) <= 2 and "summary" not in missing and "recommended" not in missing
    return ok, missing


# ---------------------------------------------------------------------------
# Offline template
# ---------------------------------------------------------------------------
def render_template_report(payload: dict) -> str:
    """Deterministic offline report — no LLM, fully traceable."""
    decision = payload.get("decision") or {}
    reading = payload.get("reading") or {}
    checks = decision.get("recommended_checks") or ["None specified."]

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

    lines += [
        "",
        "## Limitations",
        "- Synthetic public dataset (UCI AI4I 2020); not real plant telemetry.",
        "- Model probability != verified physical machine condition.",
        "- Prototype decision support; not approved for operational machine control.",
    ]

    lines += ["", "## Human verification required"]
    status = decision.get("evidence_status")
    risk = decision.get("risk_level")
    if status != "COMPLETE" or risk == "HIGH":
        lines.append(
            "- Yes: evidence is incomplete/stale or risk is high. A human must "
            "verify before acting."
        )
    else:
        lines.append(
            "- No blocking condition detected, but all outputs are estimates and "
            "remain subject to human judgement."
        )

    lines += ["", "## Traceability"]
    for item in decision.get("rationale", []):
        lines.append(f"- {item}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini provider adapter (injectable for tests; never called in tests)
# ---------------------------------------------------------------------------
class GeminiProvider:
    """Thin adapter over the Google Gen AI SDK.

    ``client`` may be injected for testing so no live API call is ever required.
    """

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL, client=None):
        self.api_key = api_key
        self.model_name = model_name
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai  # lazy import: SDK is optional

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_INSTRUCTION, "temperature": 0.2},
        )
        return (getattr(response, "text", "") or "").strip()


def gemini_available(api_key: Optional[str] = None) -> bool:
    """True when a key is present and the SDK is importable."""
    if not _api_key(api_key):
        return False
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Orchestrated report generation
# ---------------------------------------------------------------------------
def generate_report_result(
    payload: dict,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    allow_fallback: bool = True,
    provider: Optional[GeminiProvider] = None,
) -> ReportResult:
    """Return a :class:`ReportResult` with explicit provenance.

    Order: use an injected ``provider`` if given, else build a Gemini provider
    from the key. Any missing credential, SDK error, timeout, malformed output,
    or empty response falls back to the deterministic template (when allowed)
    with the reason recorded — never silently mislabelled as Gemini.
    """
    key = _api_key(api_key)

    if provider is None:
        if not key:
            if not allow_fallback:
                raise RuntimeError("No Gemini API key found (set GEMINI_API_KEY).")
            return ReportResult(
                text=render_template_report(payload),
                source="template",
                fallback_reason="No Gemini API key provided; used the offline template.",
            )
        try:
            provider = GeminiProvider(key, model_name=model_name)
        except Exception as exc:  # pragma: no cover - defensive
            if not allow_fallback:
                raise
            return ReportResult(
                text=render_template_report(payload),
                source="template",
                fallback_reason=f"Could not initialise Gemini: {exc}",
            )

    try:
        text = provider.generate(build_prompt(payload))
    except Exception as exc:  # API error, timeout, network, etc.
        if not allow_fallback:
            raise
        return ReportResult(
            text=render_template_report(payload),
            source="template",
            model_name=getattr(provider, "model_name", None),
            error=str(exc),
            fallback_reason=f"Gemini request failed ({type(exc).__name__}); used the offline template.",
        )

    ok, missing = validate_report_text(text)
    if not ok:
        if not allow_fallback:
            raise ValueError(
                "Gemini response failed structural validation; missing sections: "
                + ", ".join(missing)
            )
        return ReportResult(
            text=render_template_report(payload),
            source="template",
            model_name=getattr(provider, "model_name", None),
            error="Malformed or incomplete Gemini response.",
            fallback_reason="Gemini response failed structural validation; used the offline template.",
            missing_sections=missing,
        )

    return ReportResult(text=text, source="gemini", model_name=getattr(provider, "model_name", None))


def generate_report(
    payload: dict,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    allow_fallback: bool = True,
) -> tuple[str, str]:
    """Backward-compatible wrapper returning ``(report_text, source)``."""
    result = generate_report_result(
        payload, api_key=api_key, model_name=model_name, allow_fallback=allow_fallback
    )
    return result.text, result.source


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
    result = generate_report_result(payload)
    print(f"source: {result.source}\n")
    print(result.text)
