"""ForgeWise — The Missing Evidence Engine (Streamlit UI).

Run:
    .venv\\Scripts\\streamlit.exe run app.py

A maintenance decision-support product over a deterministic pipeline. The primary
workflow is deliberately linear and explicit:

    1. Machine inputs   — enter a reading or load a synthetic dataset example
    2. Analyze machine   — one explicit action; nothing is assessed before it
    3. Assessment results — model estimate, policy band, evidence, decision
    4. Optional AI report — Gemini when configured, otherwise offline template

Supporting reference views (kept out of the main flow): **Decision History** and
**Model Diagnostics**.

The UI separates the *model prediction* from *verified evidence* and from
*missing evidence*, and never presents a model estimate as a physical fact. This
is a decision-support prototype over the synthetic public UCI AI4I dataset — not
a real-time industrial monitoring system.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src import data_loader, ingest, model_store, orchestration, predict, reporting, vision
from src.decision_engine import SensorReading, log_decision, make_decision

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS / "metrics.json"
LOG_PATH = ARTIFACTS / "decision_log.jsonl"
INGEST_QUEUE_PATH = ARTIFACTS / "ingest_queue.jsonl"

# Primary workflow pages first, reference pages after.
PAGES = [
    "Overview",
    "Machine Analysis",
    "AI Inspection Report",
    "Visual Inspection",
    "Equipment Integration",
    "Decision History",
    "Model Diagnostics",
]
REFERENCE_PAGES = {"Equipment Integration", "Decision History", "Model Diagnostics"}

DEFAULT_READING = {
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1500.0,
    "torque": 40.0,
    "tool_wear": 50.0,
}

# Sensor field -> Streamlit widget key (so sample loading can update the widgets).
WIDGET_KEYS = {
    "type": "in_type",
    "air_temperature": "in_air",
    "process_temperature": "in_process",
    "rotational_speed": "in_rpm",
    "torque": "in_torque",
    "tool_wear": "in_wear",
}

OMIT_CHOICES = [
    "(none)",
    "torque",
    "tool_wear",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
]

FIELD_LABELS = {
    "type": "Product type",
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

ANALYZE_LABEL = "Analyze machine"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    return data_loader.load_dataset()


@st.cache_data(show_spinner=False)
def load_dataset_audit():
    try:
        return data_loader.audit_dataset(load_dataset())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def load_calibration():
    path = ARTIFACTS / "threshold_analysis.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Load the model, training from the official UCI dataset on a fresh deploy."""
    return model_store.get_model()


def recent_decisions(limit: int = 25) -> list[dict]:
    """Return the most recent decision-log records (newest first)."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))


def recent_ingest(limit: int = 8) -> list[dict]:
    """Return the most recent equipment-ingestion records (newest first)."""
    if not INGEST_QUEUE_PATH.exists():
        return []
    lines = INGEST_QUEUE_PATH.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))


def _sample_row(target: int) -> dict:
    df = load_dataset()
    row = df[df["Machine failure"] == target].iloc[0]
    return {
        "type": str(row["Type"]),
        "air_temperature": float(row["Air temperature [K]"]),
        "process_temperature": float(row["Process temperature [K]"]),
        "rotational_speed": float(row["Rotational speed [rpm]"]),
        "torque": float(row["Torque [Nm]"]),
        "tool_wear": float(row["Tool wear [min]"]),
    }


def _secrets_api_key() -> str | None:
    """Read a Gemini key from Streamlit Secrets, tolerating the no-secrets case."""
    try:
        secrets = st.secrets
    except Exception:
        return None
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if name in secrets:
                value = str(secrets[name]).strip()
                if value:
                    return value
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1180px; }
          h1, h2, h3, h4 { letter-spacing: -0.01em; }
          .fw-muted { color: #64748b; font-size: 0.85rem; }
          .fw-step {
            display: inline-block; background: #e2e8f0; color: #334155;
            border-radius: 6px; padding: 1px 8px; font-size: 0.72rem;
            font-weight: 800; letter-spacing: 0.03em; margin-right: 8px;
          }
          .fw-section { margin: 0.2rem 0 0.35rem; font-size: 1.06rem; font-weight: 700; }

          .fw-hero {
            background: linear-gradient(100deg, #0f172a 0%, #1e293b 55%, #1d4ed8 100%);
            color: #f8fafc; border-radius: 16px; padding: 1.1rem 1.4rem;
            margin-bottom: 1rem;
          }
          .fw-hero-title { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; }
          .fw-hero-title span { color: #93c5fd; }
          .fw-hero-sub { color: #cbd5e1; font-size: 0.9rem; margin-top: 0.15rem; }
          .fw-hero-meta { margin-top: 0.6rem; }

          .fw-pill {
            display: inline-block; padding: 3px 11px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
            margin: 2px 6px 2px 0;
          }
          .fw-ok   { background: #dcfce7; color: #166534; }
          .fw-warn { background: #fef3c7; color: #92400e; }
          .fw-err  { background: #fee2e2; color: #991b1b; }
          .fw-note { background: #dbeafe; color: #1e40af; }
          .fw-dark { background: rgba(255,255,255,0.14); color: #e2e8f0; }

          .fw-band {
            position: relative; display: flex; height: 30px; border-radius: 8px;
            overflow: hidden; border: 1px solid #e2e8f0; margin: 0.5rem 0 0.25rem;
          }
          .fw-band-seg {
            display: flex; align-items: center; justify-content: center;
            font-size: 11px; font-weight: 700; color: #475569;
          }
          .fw-band-low  { flex: 3; background: #dcfce7; }
          .fw-band-mod  { flex: 4; background: #fef3c7; }
          .fw-band-high { flex: 3; background: #fee2e2; }
          .fw-band-marker {
            position: absolute; top: -5px; bottom: -5px; width: 3px;
            background: #0f172a; border-radius: 2px; transform: translateX(-1.5px);
          }
          @media (max-width: 640px) {
            .fw-hero-title { font-size: 1.2rem; }
            .fw-band-seg { font-size: 9px; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pill(label: str, kind: str) -> str:
    return f'<span class="fw-pill fw-{kind}">{label}</span>'


def _risk_kind(risk: str) -> str:
    return {"LOW": "ok", "MODERATE": "warn", "HIGH": "err"}.get(risk, "note")


def _evidence_kind(status: str) -> str:
    return {
        "COMPLETE": "ok",
        "INCONCLUSIVE": "warn",
        "INCOMPLETE": "err",
        "STALE": "err",
        "NO_MODEL_OUTPUT": "err",
    }.get(status, "note")


def _band_bar(probability: float | None) -> str:
    """Visualise the model score against the documented risk bands (real data)."""
    pct = 0.0 if probability is None else max(0.0, min(1.0, probability)) * 100
    marker = (
        f'<div class="fw-band-marker" style="left:{pct:.1f}%"></div>'
        if probability is not None
        else ""
    )
    return (
        '<div class="fw-band">'
        '<div class="fw-band-seg fw-band-low">LOW &lt; 0.30</div>'
        '<div class="fw-band-seg fw-band-mod">MODERATE 0.30–0.70</div>'
        '<div class="fw-band-seg fw-band-high">HIGH ≥ 0.70</div>'
        f"{marker}"
        "</div>"
    )


def _step(number: str, title: str) -> None:
    st.markdown(
        f'<div class="fw-section"><span class="fw-step">STEP {number}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def _goto(target: str) -> None:
    """on_click callback so CTAs can change the navigation radio safely."""
    st.session_state["nav"] = target


def _reading_signature(fields: dict, machine_id: str, age_minutes: int) -> tuple:
    def _r(value):
        return None if value is None else round(float(value), 4)

    return (
        fields.get("type"),
        _r(fields.get("air_temperature")),
        _r(fields.get("process_temperature")),
        _r(fields.get("rotational_speed")),
        _r(fields.get("torque")),
        _r(fields.get("tool_wear")),
        machine_id,
        age_minutes,
    )


def _report_payload(assessment: orchestration.Assessment) -> dict:
    """Grounding payload: model output + evidence + missing evidence + applied rules."""
    payload = assessment.to_payload()
    decision = assessment.decision
    payload["policy"] = {
        "risk_bands": {"LOW": "< 0.30", "MODERATE": "0.30-0.70", "HIGH": ">= 0.70"},
        "stale_after_minutes": 60,
        "evidence_status": decision.evidence_status,
        "applied_decision": decision.decision,
        "data_provenance": "manual or simulated input; synthetic public AI4I dataset",
    }
    return payload


def _render_report_status(report: reporting.ReportResult | None) -> None:
    """Show an honest, unambiguous AI-report state (never assumes Gemini works)."""
    if report is None:
        return
    if report.source == "gemini":
        st.success(
            f"AI inspection report generated by **Gemini** (`{report.model_name}`), "
            "grounded in the assessment payload.",
            icon="✅",
        )
    elif report.error:
        st.error(
            "AI report request **failed** — showing the offline deterministic "
            "template instead. Reason: "
            + (report.fallback_reason or report.error),
            icon="⚠️",
        )
    else:
        st.warning(
            "AI report: **offline deterministic template** (Gemini not configured). "
            + (report.fallback_reason or ""),
            icon="ℹ️",
        )


# ---------------------------------------------------------------------------
# Startup: model bootstrap (must happen before navigation renders content)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ForgeWise — Missing Evidence Engine",
    page_icon="🛠️",
    layout="wide",
)
_inject_css()

try:
    with st.spinner(
        "Preparing model… on a fresh deployment this trains the baseline from "
        "the official UCI AI4I dataset (one-time, then cached)."
    ):
        _pipeline, metadata = load_model_bundle()
except model_store.ModelUnavailable as exc:
    st.error("ForgeWise could not start: no trained model is available.")
    st.write(str(exc))
    st.info(
        "Train locally with `python -m src.train_model`, or confirm the app can "
        "reach the official UCI dataset URL and redeploy."
    )
    st.stop()
except Exception as exc:  # unexpected startup failure
    st.error(f"Unexpected startup error while preparing the model: {exc}")
    st.stop()

metrics = load_metrics()


# ---------------------------------------------------------------------------
# Sidebar: navigation + model status + AI provider configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ ForgeWise")
    st.caption("The Missing Evidence Engine")

    st.caption("**Workflow**")
    page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    if page in REFERENCE_PAGES:
        st.caption("Reference view — separate from the main assessment flow.")

    st.divider()
    st.markdown("**Model status**")
    st.caption(f"Version `{metadata['model_version']}`")
    st.caption(f"{metadata['model_type']} · {metadata['dataset']['n_rows']:,} dataset rows")
    if metrics:
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        m2.metric("PR-AUC", f"{metrics['average_precision']:.3f}")
        m3, m4 = st.columns(2)
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("Precision", f"{metrics['precision']:.3f}")
        st.caption("Held-out test split (committed reference).")

    st.divider()
    st.markdown("**AI report provider**")
    session_key = st.text_input(
        "Gemini API key (optional override)",
        type="password",
        value="",
        key="gemini_key",
        help="Prefer Streamlit Secrets or the GEMINI_API_KEY environment variable.",
    )
    secret_key = _secrets_api_key()
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    resolved_key = (session_key or "").strip() or secret_key or env_key
    if resolved_key and reporting.gemini_available(resolved_key):
        source = "session" if (session_key or "").strip() else ("secrets" if secret_key else "environment")
        st.info(
            f"A Gemini key was detected ({source}). Generation is attempted **only** "
            "when you request it, and is labelled by its real source afterwards.",
            icon="🔑",
        )
    else:
        st.info(
            "No Gemini key detected — AI reports use the offline deterministic "
            "template.",
            icon="ℹ️",
        )

    st.divider()
    st.caption("Synthetic/public data — UCI AI4I 2020 (CC BY 4.0). Not live telemetry.")


def _render_hero() -> None:
    pills = _pill("Model " + metadata["model_version"][-8:], "dark")
    pills += _pill("Synthetic AI4I dataset", "dark")
    pills += _pill("Manual / simulated input", "dark")
    pills += _pill(
        "AI provider: key detected"
        if (resolved_key and reporting.gemini_available(resolved_key))
        else "AI provider: not configured",
        "dark",
    )
    st.markdown(
        f"""
        <div class="fw-hero">
          <div class="fw-hero-title">ForgeWise <span>· Missing Evidence Engine</span></div>
          <div class="fw-hero-sub">
            Maintenance decision support that separates model estimates from verified
            evidence — and makes missing or stale evidence explicit.
          </div>
          <div class="fw-hero-meta">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_overview() -> None:
    st.markdown("#### Overview")
    st.warning(
        "**Demo data only.** Every reading is **manually entered** or a "
        "**simulated dataset example**. This is **not** real-time telemetry, and no "
        "physical machine has been inspected.",
        icon="🧪",
    )

    assessment = st.session_state.get("assessment")
    report = st.session_state.get("report")

    if assessment is None or assessment.decision is None:
        st.info(
            "No assessment yet. Open **Machine Analysis** to enter or load a "
            f"reading, then press **{ANALYZE_LABEL}**."
        )
        st.button(
            "Start an assessment",
            type="primary",
            on_click=_goto,
            args=("Machine Analysis",),
        )
        return

    decision = assessment.decision
    prob_display = "—" if assessment.probability is None else f"{assessment.probability:.3f}"

    st.markdown("**Latest assessment**")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model-estimated probability", prob_display)
        c2.metric("Risk category", decision.risk_level)
        c3.metric("Evidence completeness", decision.evidence_status)
        c4.metric("Decision", decision.decision)
        st.markdown(_band_bar(assessment.probability), unsafe_allow_html=True)
        st.caption(
            "The marker shows the model score against the documented policy bands. "
            "This is a statistical estimate, not a verified physical condition."
        )

    next_action = (
        decision.recommended_checks[0]
        if decision.recommended_checks
        else "No action required; continue routine monitoring."
    )
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Next recommended action**")
        st.write(next_action)
        if decision.evidence_gaps:
            st.markdown("**Evidence gaps**")
            for gap in decision.evidence_gaps:
                st.warning(gap, icon="⚠️")
    with right:
        st.markdown("**Status**")
        tag = _pill(f"Risk {decision.risk_level}", _risk_kind(decision.risk_level))
        tag += _pill(
            f"Evidence {decision.evidence_status}", _evidence_kind(decision.evidence_status)
        )
        if report is not None:
            tag += _pill(
                "Report: Gemini" if report.source == "gemini" else "Report: offline",
                "note" if report.source == "gemini" else "warn",
            )
        st.markdown(tag, unsafe_allow_html=True)
        if st.button("Open AI Report", on_click=_goto, args=("AI Inspection Report",)):
            pass

    st.info(
        "Verified physical machine condition: **NOT VERIFIED**. ForgeWise has no "
        "physical inspection and no ground-truth telemetry, and is not approved for "
        "operational machine control.",
        icon="🔎",
    )


def render_analysis() -> None:
    st.markdown("#### Machine Analysis")
    st.caption(
        "Workflow: **1** enter machine inputs → **2** analyze machine → "
        "**3** review the assessment → **4** optionally generate an AI report."
    )

    # Initialise widget state so sample loading can update the inputs.
    for field, key in WIDGET_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_READING[field]
    for key, value in (("in_machine", "machine-01"), ("in_age", 0), ("in_omit", "(none)")):
        if key not in st.session_state:
            st.session_state[key] = value

    # --- STEP 1: inputs -----------------------------------------------------
    _step("1", "Machine inputs")
    with st.container(border=True):
        load_cols = st.columns([1, 1, 2])
        with load_cols[0]:
            load_normal = st.button("Load example — healthy (label 0)", width="stretch")
        with load_cols[1]:
            load_failure = st.button("Load example — failure (label 1)", width="stretch")
        with load_cols[2]:
            st.caption(
                "Optional: inserts one synthetic AI4I dataset row. These are "
                "**dataset examples**, not real machine measurements."
            )

        if load_normal or load_failure:
            try:
                sample = _sample_row(0 if load_normal else 1)
                for field, key in WIDGET_KEYS.items():
                    st.session_state[key] = sample[field]
            except Exception as exc:
                st.warning(f"Could not load a sample from the dataset: {exc}")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            r_type = st.selectbox("Product type", ["L", "M", "H"], key="in_type")
            air = st.number_input("Air temperature [K]", step=0.1, format="%.1f", key="in_air")
            process = st.number_input(
                "Process temperature [K]", step=0.1, format="%.1f", key="in_process"
            )
        with c2:
            rpm = st.number_input("Rotational speed [rpm]", step=1.0, format="%.0f", key="in_rpm")
            torque = st.number_input("Torque [Nm]", step=0.1, format="%.1f", key="in_torque")
            wear = st.number_input("Tool wear [min]", step=1.0, format="%.0f", key="in_wear")
        with c3:
            machine_id = st.text_input("Machine ID", key="in_machine")

        with st.expander("Advanced: simulate evidence problems"):
            st.caption(
                "Used to demonstrate the Missing Evidence Engine. Leave at defaults "
                "for a normal reading."
            )
            e1, e2 = st.columns(2)
            with e1:
                age_minutes = st.slider(
                    "Simulated reading age (minutes)",
                    min_value=0,
                    max_value=240,
                    step=5,
                    key="in_age",
                    help="Demonstrates stale-evidence detection.",
                )
            with e2:
                omit_field = st.selectbox(
                    "Simulate missing field (optional)", OMIT_CHOICES, key="in_omit"
                )

        with st.expander("Compare a healthy vs a failure dataset example"):
            st.caption(
                "Runs the current model on one labelled healthy row and one labelled "
                "failure row, side by side. Dataset examples only."
            )
            if st.button("Run healthy vs failure comparison"):
                try:
                    comparison = []
                    for label, name in ((0, "healthy (label 0)"), (1, "failure (label 1)")):
                        sample = SensorReading(
                            machine_id=name,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            **_sample_row(label),
                        )
                        p = predict.predict_probability(sample)
                        d = make_decision(
                            sample,
                            model_probability=p,
                            model_version=metadata["model_version"],
                        )
                        comparison.append(
                            {
                                "example": name,
                                "model_probability": round(p, 4),
                                "risk_level": d.risk_level,
                                "evidence_status": d.evidence_status,
                                "decision": d.decision,
                                "mechanism_flags": ", ".join(d.mechanism_flags) or "none",
                            }
                        )
                    st.dataframe(comparison, width="stretch", hide_index=True)
                    st.caption(
                        "A controlled comparison of two labelled dataset rows; the score "
                        "is a model estimate for simulated inputs, not verified fact."
                    )
                except Exception as exc:
                    st.warning(f"Comparison unavailable: {exc}")

    # --- STEP 2: analyze ----------------------------------------------------
    _step("2", "Analyze machine")
    st.caption(
        "Nothing is assessed until you run the analysis. Checking the box below "
        "**requests an AI inspection report as part of this analysis**; it is "
        "optional and never required."
    )
    gen_report = st.checkbox(
        "Request an AI inspection report during this analysis",
        key="in_genreport",
        help=(
            "Uses Gemini when a key is configured, otherwise the offline "
            "deterministic template. The report is an explanation layer, not the "
            "predictor."
        ),
    )

    fields = {
        "type": r_type,
        "air_temperature": float(air),
        "process_temperature": float(process),
        "rotational_speed": float(rpm),
        "torque": float(torque),
        "tool_wear": float(wear),
    }
    if omit_field != "(none)":
        fields[omit_field] = None

    signature = _reading_signature(fields, machine_id, age_minutes)
    analyze = st.button(ANALYZE_LABEL, type="primary", key="btn_analyze")

    if analyze:
        timestamp = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
        reading = SensorReading(machine_id=machine_id, timestamp=timestamp, **fields)
        with st.spinner("Running the assessment…"):
            assessment = orchestration.run_assessment(reading, metadata["model_version"])

        if not assessment.ok:
            st.session_state["assessment"] = None
            st.session_state["report"] = None
            st.session_state["assessment_signature"] = signature
            st.session_state["last_input_errors"] = assessment.input_errors
        else:
            result = None
            if gen_report:
                with st.spinner("Requesting the AI inspection report…"):
                    result = reporting.generate_report_result(
                        _report_payload(assessment), api_key=resolved_key or None
                    )
            st.session_state["assessment"] = assessment
            st.session_state["report"] = result
            st.session_state["assessment_signature"] = signature
            st.session_state["last_input_errors"] = []
            log_decision(
                assessment.decision,
                reading,
                log_path=LOG_PATH,
                extra={"report_mode": result.source if result else None},
            )

    assessment = st.session_state.get("assessment")
    if assessment is None:
        errors = st.session_state.get("last_input_errors")
        if errors:
            st.error("Input validation failed:\n\n" + "\n".join(f"- {e}" for e in errors))
        return

    if not assessment.ok:
        st.error(
            "Input validation failed:\n\n"
            + "\n".join(f"- {e}" for e in assessment.input_errors)
        )
        return

    if st.session_state.get("assessment_signature") != signature:
        st.info(f"Inputs have changed since the last analysis — press **{ANALYZE_LABEL}** to refresh.")

    _render_results(assessment)


def _render_results(assessment: orchestration.Assessment) -> None:
    decision = assessment.decision
    probability = assessment.probability
    prob_display = "—" if probability is None else f"{probability:.3f}"

    # --- STEP 3: results ----------------------------------------------------
    st.divider()
    _step("3", "Assessment results")
    st.caption(
        "These four numbers are related but **not interchangeable**. The model "
        "estimate is a statistical output, not a verified physical condition."
    )

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Model-estimated probability",
            prob_display,
            help="The classifier's statistical output. Not a measurement, not a guarantee.",
        )
        m2.metric(
            "Risk category",
            decision.risk_level,
            help="A policy band applied to the probability (LOW < 0.30, HIGH >= 0.70). A routing label, not evidence.",
        )
        m3.metric(
            "Evidence completeness",
            decision.evidence_status,
            help="Data quality: whether required fields are present and fresh. Independent of the score.",
        )
        m4.metric("Decision", decision.decision)
        st.markdown(_band_bar(probability), unsafe_allow_html=True)
        st.caption(
            "The marker is the model score within the documented policy bands. A "
            "probability of 0.000 means every tree voted \"no failure\" — a coarse "
            "estimate, not a certainty."
        )

    st.markdown("**How to read these four numbers**")
    read_left, read_right = st.columns(2)
    with read_left:
        st.markdown(
            f"**1. Model-estimated probability — `{prob_display}`**  \n"
            "The classifier's statistical output. It is *not* a physical measurement "
            "and *not* a guarantee.\n\n"
            f"**2. Risk category — `{decision.risk_level}`**  \n"
            "A documented *policy band* derived from that probability "
            "(LOW < 0.30, MODERATE 0.30–0.70, HIGH ≥ 0.70). It routes attention; it "
            "is not independent evidence."
        )
    with read_right:
        st.markdown(
            f"**3. Evidence completeness — `{decision.evidence_status}`**  \n"
            "A *data-quality* status: are the required fields present and fresh? This "
            "is independent of the model score.\n\n"
            "**4. Operational decision — "
            f"`{decision.decision}`**  \n"
            "The recommended action from the documented rules. A model estimate is "
            "**not** a verified physical condition — here it is **NOT VERIFIED**."
        )

    ev_col, pred_col = st.columns(2)
    with pred_col:
        st.markdown("**Model prediction (estimate)**")
        st.write(f"- Failure probability: **{prob_display}**")
        st.write(f"- Risk category: **{decision.risk_level}**")
        st.write(f"- Model version: `{decision.model_version}`")
        st.write(f"- Confidence label: {decision.confidence}")
        st.caption("Statistical output — not a verified fact.")
    with ev_col:
        st.markdown("**Verified evidence (measured inputs)**")
        evidence = asdict(assessment.reading)
        for key in FIELD_LABELS:
            value = evidence.get(key)
            shown = "MISSING" if value is None else value
            st.write(f"- {FIELD_LABELS[key]}: {shown}")

    st.markdown("**Decision & next action**")
    with st.container(border=True):
        st.markdown(f"- **Operational decision:** {decision.decision}")
        if decision.evidence_gaps:
            for gap in decision.evidence_gaps:
                st.warning(gap, icon="⚠️")
        else:
            st.success("All required evidence present and fresh.", icon="✅")
        st.markdown("**Recommended next inspection (rule-based)**")
        for check in decision.recommended_checks:
            st.markdown(f"- {check}")

    required, reasons = orchestration.human_review_required(decision)
    if required:
        st.error(
            "**Human verification required.** " + " ".join(reasons)
            + " Do not act on the model estimate alone.",
            icon="🔧",
        )
    else:
        st.success(
            "No blocking condition detected. All outputs remain estimates and are "
            "subject to human judgement.",
            icon="🔧",
        )

    # --- STEP 4: optional report -------------------------------------------
    st.divider()
    _step("4", "Optional AI inspection report")
    report = st.session_state.get("report")
    if report is not None:
        _render_report_status(report)
        st.button(
            "Open the AI report",
            type="primary",
            on_click=_goto,
            args=("AI Inspection Report",),
        )
    else:
        st.caption(
            "No AI report was requested for this assessment. Generate one from the "
            "**AI Inspection Report** page, or tick the request box in Step 2 before "
            "analyzing."
        )
        st.button(
            "Generate AI Report",
            on_click=_goto,
            args=("AI Inspection Report",),
        )

    with st.expander("Documented mechanism heuristics (support only, not the prediction)"):
        if decision.mechanism_flags:
            for name in decision.mechanism_flags:
                st.info(name, icon="🔎")
            st.caption(
                "Heuristics from the public AI4I dataset documentation; used to "
                "recommend checks, never as the model's prediction."
            )
        else:
            st.write("None triggered.")

    with st.expander("Pipeline stages"):
        for stage in assessment.stages:
            marker = {
                "ok": "✅",
                "warning": "🟡",
                "error": "⛔",
                "skipped": "⏭️",
            }.get(stage.status, "•")
            st.markdown(f"{marker} **{stage.title}** — {stage.detail}")

    with st.expander("Traceability / rationale"):
        for line in decision.rationale:
            st.markdown(f"- {line}")


def render_report() -> None:
    st.markdown("#### AI Inspection Report")
    st.caption(
        "An optional explanation layer produced by Gemini (when configured) or the "
        "offline deterministic template. It never replaces the decision rules, and "
        "the real source is always shown."
    )

    assessment = st.session_state.get("assessment")
    report = st.session_state.get("report")

    if assessment is None or assessment.decision is None:
        st.info("No assessment yet. Run one from **Machine Analysis** first.")
        st.button(
            "Start an assessment",
            type="primary",
            on_click=_goto,
            args=("Machine Analysis",),
        )
        return

    if report is None:
        if resolved_key and reporting.gemini_available(resolved_key):
            st.info(
                "A Gemini key is configured. Pressing the button below will **attempt** "
                "a Gemini request; if it fails, the offline template is shown and the "
                "failure is labelled.",
                icon="🔑",
            )
        else:
            st.info(
                "No Gemini key is configured — the report will use the offline "
                "deterministic template.",
                icon="ℹ️",
            )
        if st.button("Generate AI Report", type="primary"):
            with st.spinner("Requesting the AI inspection report…"):
                result = reporting.generate_report_result(
                    _report_payload(assessment), api_key=resolved_key or None
                )
            st.session_state["report"] = result
            st.rerun()
        return

    _render_report_status(report)
    st.markdown(report.text)
    if report.error:
        st.caption(f"Provider detail: {report.error}")

    with st.expander("Report provenance"):
        st.write(f"- Source: `{report.source}`")
        st.write(f"- Model: `{report.model_name or 'n/a'}`")
        st.write(f"- Fallback: {report.is_fallback}")
        if report.missing_sections:
            st.write(f"- Missing sections: {', '.join(report.missing_sections)}")


def render_vision() -> None:
    st.markdown("#### Visual Inspection")
    st.caption(
        "Two separate computer-vision tools. Both run fully offline with Pillow + "
        "NumPy and are deterministic."
    )
    st.warning(
        "These tools analyse **image usability** and **visual differences** only. "
        "They do **not** detect defects or diagnose machine condition, and every "
        "result requires human review.",
        icon="🧪",
    )

    tool = st.radio(
        "Tool",
        ["Image quality check", "Defect detection"],
        horizontal=True,
        key="vision_tool",
        label_visibility="collapsed",
    )
    if tool == "Image quality check":
        _render_quality_tool()
    else:
        _render_detection_tool()


def _render_quality_tool() -> None:
    st.markdown("**Tool A · Image quality check** — is this photo usable as visual evidence?")
    uploaded = st.file_uploader(
        "Upload a machine photo (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="vision_quality_upload",
    )
    if uploaded is None:
        st.caption(
            "Upload an image to measure resolution, brightness, contrast, sharpness "
            "and colour balance."
        )
        return
    try:
        image = vision.load_image(uploaded.getvalue())
        metrics = vision.analyze_quality(image)
    except Exception as exc:
        st.error(f"Could not read the image: {exc}")
        return

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Uploaded image", width="stretch")
    with right:
        if metrics.usable:
            st.success("**USABLE** as visual evidence.", icon="✅")
        else:
            st.warning("**RECAPTURE recommended.**", icon="⚠️")
        m1, m2 = st.columns(2)
        m1.metric("Resolution", f"{metrics.width}×{metrics.height}")
        m2.metric("Megapixels", metrics.megapixels)
        m3, m4 = st.columns(2)
        m3.metric("Brightness", f"{metrics.brightness:.2f}")
        m4.metric("Contrast", f"{metrics.contrast:.2f}")
        m5, m6 = st.columns(2)
        m5.metric("Sharpness", f"{metrics.sharpness:.1f}")
        m6.metric("Colour balance", f"{metrics.color_balance:.2f}")

    if metrics.issues:
        for issue in metrics.issues:
            st.warning(issue, icon="📷")
    st.caption(
        "Objective image-quality measurements (an evidence-usability gate), not "
        "defect detection. Thresholds are documented in `src/vision.py`."
    )


def _render_detection_tool() -> None:
    st.markdown("**Tool B · Defect detection** — reference comparison + detector interface")
    detector, message = vision.load_detector()
    if detector is None:
        st.info(message, icon="ℹ️")

    st.markdown("**Reference comparison (visual difference)**")
    st.caption(
        "Compares a known-good reference photo with a current photo of the same view "
        "and reports the changed fraction, mean difference, a global SSIM "
        "approximation, and the bounding box of changes."
    )
    c1, c2 = st.columns(2)
    reference = c1.file_uploader(
        "Reference (known-good) image", type=["jpg", "jpeg", "png"], key="vision_ref"
    )
    current = c2.file_uploader(
        "Current image", type=["jpg", "jpeg", "png"], key="vision_cur"
    )
    if reference is None or current is None:
        st.caption("Upload both images to compare them.")
        return
    try:
        result = vision.compare_images(reference.getvalue(), current.getvalue())
    except Exception as exc:
        st.error(f"Could not compare the images: {exc}")
        return

    verdict_kind = {
        "NO_MATERIAL_CHANGE": ("success", "No material visual change detected."),
        "REVIEW": ("warning", "Visual change detected — review recommended."),
        "SIGNIFICANT_CHANGE": (
            "error",
            "Significant visual change detected — human review required.",
        ),
    }[result.verdict]
    getattr(st, verdict_kind[0])(verdict_kind[1])

    cols = st.columns([1, 1])
    cols[0].image(reference, caption="Reference", width="stretch")
    cols[1].image(current, caption="Current", width="stretch")

    r1, r2, r3 = st.columns(3)
    r1.metric("Changed area", f"{result.changed_fraction * 100:.2f}%")
    r2.metric("Mean abs difference", f"{result.mean_abs_diff:.1f}")
    r3.metric("SSIM (global)", f"{result.structural_similarity:.3f}")
    if result.bounding_box:
        st.caption(f"Changed region (x0, y0, x1, y1): {result.bounding_box}")
    st.caption(result.note)


def render_integration() -> None:
    st.markdown("#### Equipment Integration")
    st.caption(
        "Reference view. Four interchangeable templates let physical manufacturing "
        "equipment feed readings into ForgeWise through one shared, unit-tested "
        "validator."
    )
    st.markdown(
        "| # | Template | Transport | Optional dependency |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | `integrations/rest_ingest.py` | HTTP POST `/ingest` | none |\n"
        "| 2 | `integrations/mqtt_subscriber.py` | MQTT subscribe | `paho-mqtt` |\n"
        "| 3 | `integrations/modbus_poller.py` | Modbus/TCP poll | `pymodbus` |\n"
        "| 4 | `integrations/csv_scada_import.py` | CSV batch (SCADA) | none |"
    )
    with st.expander("Accepted payload schema"):
        st.code(ingest.payload_schema())
    st.warning(
        "These templates are examples. Add authentication, TLS, and durable storage "
        "before any real deployment; the queue below is ephemeral on a hosted app.",
        icon="🧪",
    )

    st.markdown("**Try a payload**")
    st.caption(
        "Paste a JSON object; it is validated with the same normaliser the templates "
        "use."
    )
    default_payload = json.dumps(
        {
            "machine_id": "cnc-07",
            "type": "M",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 50,
        },
        indent=2,
    )
    text = st.text_area(
        "Payload (JSON)", value=default_payload, height=180, key="ingest_payload"
    )
    c1, c2 = st.columns(2)
    validate_clicked = c1.button("Validate payload", type="primary", key="btn_validate_payload")
    run_clicked = c2.button("Run assessment from payload", key="btn_run_payload")

    if validate_clicked or run_clicked:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return
        normalized = ingest.normalize_payload(payload)
        if normalized.errors:
            st.error(
                "Payload rejected:\n\n" + "\n".join(f"- {e}" for e in normalized.errors)
            )
            return
        st.success("Payload valid.", icon="✅")
        st.json(normalized.fields)
        for warning in normalized.warnings:
            st.warning(warning)
        if normalized.unknown_keys:
            st.caption(f"Ignored unknown keys: {', '.join(normalized.unknown_keys)}")
        if normalized.missing_fields:
            st.info(
                "Missing (treated as missing evidence): "
                + ", ".join(normalized.missing_fields)
            )

        if run_clicked:
            reading = ingest.to_reading(payload)
            assessment = orchestration.run_assessment(reading, metadata["model_version"])
            st.session_state["assessment"] = assessment
            st.session_state["report"] = None
            st.session_state["assessment_signature"] = None
            if assessment.ok:
                log_decision(
                    assessment.decision, reading, log_path=LOG_PATH,
                    extra={"report_mode": None},
                )
                st.success(
                    "Assessment created from payload. Open **Overview** or "
                    "**Machine Analysis** to review it."
                )
            else:
                st.error("Assessment failed input validation.")

    st.divider()
    st.markdown("**Recent queued readings**")
    queue_records = recent_ingest(8)
    if queue_records:
        st.dataframe(
            [
                {
                    "Received (UTC)": (record.get("received_at") or "")[:19] + "Z",
                    "Source": record.get("source", "—"),
                    "Machine": (record.get("fields") or {}).get("machine_id", "—"),
                    "Errors": len(record.get("errors") or []),
                }
                for record in queue_records
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(
            "No queued readings yet. A queued file appears here after you run one of "
            "the templates (`artifacts/ingest_queue.jsonl`)."
        )


def render_history() -> None:
    st.markdown("#### Decision History")
    st.caption(
        "Reference view. Records are written to the app container's local "
        "filesystem. On a hosted deployment this storage is **ephemeral** — records "
        "are lost on restart or redeploy, and this is **not** durable audit storage."
    )

    records = recent_decisions(25)
    if not records:
        st.info("No decisions recorded yet. Run an assessment from **Machine Analysis**.")
        st.button(
            "Start an assessment",
            type="primary",
            on_click=_goto,
            args=("Machine Analysis",),
        )
        return

    high = sum(1 for r in records if r.get("decision", {}).get("risk_level") == "HIGH")
    assets = {r.get("reading", {}).get("machine_id") for r in records}
    latest = (records[0].get("decision", {}).get("created_at") or "")[:19]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Records (last 25)", len(records))
    s2.metric("HIGH-risk events", high)
    s3.metric("Assets seen", len(assets))
    s4.metric("Latest (UTC)", latest + "Z" if latest else "—")

    rows = []
    for record in records:
        d = record.get("decision", {})
        reading = record.get("reading", {})
        extra = record.get("extra", {}) or {}
        created = (d.get("created_at") or "")[:19]
        score = d.get("model_probability")
        rows.append(
            {
                "Time (UTC)": f"{created}Z",
                "Asset": reading.get("machine_id", "—"),
                "Model": d.get("model_version", "—"),
                "Model score": "—" if score is None else round(float(score), 4),
                "Risk": d.get("risk_level", "—"),
                "Evidence": d.get("evidence_status", "—"),
                "Recommendation": d.get("decision", "—"),
                "Report mode": extra.get("report_mode") or "—",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_diagnostics() -> None:
    st.markdown("#### Model Diagnostics")
    st.caption(
        "Reference view. Metrics are computed on the held-out test split only — no "
        "test-set tuning. Numbers are read from committed artifacts; none are "
        "hand-entered."
    )

    audit = load_dataset_audit()

    if metrics:
        st.markdown("**Held-out test metrics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c2.metric("PR-AUC", f"{metrics['average_precision']:.3f}")
        c3.metric("Recall @ 0.5", f"{metrics['recall']:.3f}")
        c4.metric("Precision @ 0.5", f"{metrics['precision']:.3f}")
        st.caption(
            f"n_test = {metrics['n_test']} rows · positive rate "
            f"{metrics['positive_rate_test']*100:.1f}% · threshold {metrics['threshold']}. "
            "Class imbalance is heavy; recall and PR-AUC are the informative metrics. "
            "At 0.5 the model misses ~59% of failures."
        )

        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            [
                [cm["true_negative"], cm["false_positive"]],
                [cm["false_negative"], cm["true_positive"]],
            ],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.markdown("**Confusion matrix (held-out test)**")
        st.dataframe(cm_df, width="stretch")
    else:
        st.info("No committed metrics found (`artifacts/metrics.json`).")

    for image in ("confusion_matrix.png", "roc_pr_curves.png"):
        path = ARTIFACTS / image
        if path.exists():
            st.image(str(path))

    st.divider()
    st.markdown("**Threshold calibration (validation-selected, no test-set tuning)**")
    calibration = load_calibration()
    if calibration:
        val = calibration["validation"]
        ref = calibration["held_out_test_reference"]
        st.caption(
            f"Selected threshold **{val['selected_threshold']}** by maximising "
            f"{calibration['selection_metric']} on a validation split carved from the "
            f"training pool (seed {val['random_state']}). The held-out test set was "
            "**not** used to select it; `artifacts/metrics.json` is unchanged."
        )

        sweep = val.get("sweep")
        if sweep:
            curve = pd.DataFrame(sweep).set_index("threshold")[["precision", "recall", "f1"]]
            st.markdown("**Validation precision / recall / F1 vs threshold**")
            st.line_chart(curve, height=280)

        def _row(label: str, m: dict) -> dict:
            return {
                "evaluation": label,
                "threshold": m["threshold"],
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "f1": round(m["f1"], 3),
                "TN": m["confusion_matrix"]["true_negative"],
                "FP": m["confusion_matrix"]["false_positive"],
                "FN": m["confusion_matrix"]["false_negative"],
                "TP": m["confusion_matrix"]["true_positive"],
            }

        st.markdown("**Validation split**")
        st.dataframe(
            [
                _row("current (0.5)", val["metrics_at_current"]),
                _row(f"selected ({val['selected_threshold']})", val["metrics_at_selected"]),
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Held-out test (reference only)**")
        st.dataframe(
            [
                _row("current (0.5)", ref["metrics_at_current_0.5"]),
                _row(f"selected ({val['selected_threshold']})", ref["metrics_at_selected"]),
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(ref["note"])
        st.caption(calibration["policy_note"])
    else:
        st.caption("No threshold analysis found. Run `python -m src.calibrate_threshold`.")

    if audit:
        st.divider()
        st.markdown("**Dataset audit (synthetic AI4I 2020)**")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Rows", f"{audit['row_count']:,}")
        a2.metric("Columns", audit["column_count"])
        a3.metric("Missing values", audit["missing_values_total"])
        a4.metric("Duplicate rows", audit["duplicate_rows"])

    st.divider()
    st.markdown("**Limitations**")
    st.markdown(
        "- Synthetic public dataset (UCI AI4I 2020); not real plant telemetry.\n"
        "- Model probability is an estimate, not a verified physical condition.\n"
        "- Recall at the default threshold is limited, so some failures are missed.\n"
        "- Feature importances are not shown: a RandomForest's individual predictions "
        "are not causally attributable.\n"
        "- Prototype decision support — **not** approved for operational machine control."
    )


PAGE_RENDERERS = {
    "Overview": render_overview,
    "Machine Analysis": render_analysis,
    "AI Inspection Report": render_report,
    "Visual Inspection": render_vision,
    "Equipment Integration": render_integration,
    "Decision History": render_history,
    "Model Diagnostics": render_diagnostics,
}

_render_hero()
PAGE_RENDERERS[page]()

st.divider()
st.caption(
    "ForgeWise — decision-support prototype. Synthetic/public AI4I data; human "
    "verification required for any action."
)
