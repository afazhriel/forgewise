"""ForgeWise — The Missing Evidence Engine (Streamlit UI).

Run:
    .venv\\Scripts\\streamlit.exe run app.py

A maintenance decision-support product over a deterministic pipeline:

* **Overview**            — last assessment dashboard, evidence status, next action
* **Machine Analysis**    — sensor inputs, sample loading, explicit Analyze action
* **AI Inspection Report** — Gemini or offline template, provenance always shown
* **Decision History**    — traceable decision events (session/container log)
* **Model Diagnostics**   — real held-out metrics, charts, calibration, limitations

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

from src import data_loader, model_store, orchestration, predict, reporting
from src.decision_engine import SensorReading, log_decision, make_decision

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS / "metrics.json"
LOG_PATH = ARTIFACTS / "decision_log.jsonl"

PAGES = [
    "Overview",
    "Machine Analysis",
    "AI Inspection Report",
    "Decision History",
    "Model Diagnostics",
]

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
          .block-container { padding-top: 1.6rem; max-width: 1200px; }
          h1, h2, h3, h4 { letter-spacing: -0.01em; }
          .fw-muted { color: #64748b; font-size: 0.85rem; }

          .fw-hero {
            background: linear-gradient(100deg, #0f172a 0%, #1e293b 55%, #1d4ed8 100%);
            color: #f8fafc; border-radius: 16px; padding: 1.15rem 1.4rem;
            margin-bottom: 1.1rem;
          }
          .fw-hero-title { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
          .fw-hero-title span { color: #93c5fd; }
          .fw-hero-sub { color: #cbd5e1; font-size: 0.9rem; margin-top: 0.15rem; }
          .fw-hero-meta { margin-top: 0.6rem; }

          .fw-pill {
            display: inline-block; padding: 3px 11px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
            margin-right: 6px;
          }
          .fw-ok   { background: #dcfce7; color: #166534; }
          .fw-warn { background: #fef3c7; color: #92400e; }
          .fw-err  { background: #fee2e2; color: #991b1b; }
          .fw-note { background: #dbeafe; color: #1e40af; }
          .fw-dark { background: rgba(255,255,255,0.14); color: #e2e8f0; }

          .fw-band {
            position: relative; display: flex; height: 30px; border-radius: 8px;
            overflow: hidden; border: 1px solid #e2e8f0; margin: 0.4rem 0 0.2rem;
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
          .fw-kpi-label { color: #64748b; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.04em; font-weight: 600; }
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


# ---------------------------------------------------------------------------
# Startup: model bootstrap (must happen before navigation renders content)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ForgeWise — Missing Evidence Engine", page_icon="🛠️", layout="wide")
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
# Sidebar: navigation + model status + Gemini configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ ForgeWise")
    st.caption("The Missing Evidence Engine")
    page = st.radio("Navigation", PAGES, key="nav")

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
        "GEMINI_API_KEY (optional override)",
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
        st.success(f"Gemini configured ({source}).", icon="✅")
    else:
        st.info("Gemini not configured — AI reports use the offline template.", icon="ℹ️")

    st.divider()
    st.caption("Synthetic/public data — UCI AI4I 2020 (CC BY 4.0). Not live telemetry.")


def _render_hero() -> None:
    pills = _pill("Model " + metadata["model_version"][-8:], "dark")
    pills += _pill("Synthetic AI4I dataset", "dark")
    pills += _pill("Manual / simulated input", "dark")
    pills += _pill(
        "AI report: Gemini" if (resolved_key and reporting.gemini_available(resolved_key)) else "AI report: offline",
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
            "reading, then press **Analyze reading**."
        )
        cta = st.columns([1, 3])
        with cta[0]:
            st.button(
                "Open Machine Analysis",
                type="primary",
                on_click=_goto,
                args=("Machine Analysis",),
                use_container_width=True,
            )
        _render_system_status()
        return

    decision = assessment.decision
    prob_display = "—" if assessment.probability is None else f"{assessment.probability:.3f}"

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model-estimated probability", prob_display)
        c2.metric("Risk category", decision.risk_level)
        c3.metric("Evidence completeness", decision.evidence_status)
        c4.metric("Decision", decision.decision)

        st.markdown(_band_bar(assessment.probability), unsafe_allow_html=True)
        st.caption(
            "The marker shows the model score against the documented policy bands. "
            "It is a statistical estimate, not a verified physical condition."
        )

    next_action = (
        decision.recommended_checks[0]
        if decision.recommended_checks
        else "No action required; continue routine monitoring."
    )
    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            st.markdown("**Next recommended action**")
            st.write(next_action)
            st.markdown("**Evidence gaps**")
            if decision.evidence_gaps:
                for gap in decision.evidence_gaps:
                    st.warning(gap, icon="⚠️")
            else:
                st.success("All required evidence present and fresh.", icon="✅")
    with right:
        with st.container(border=True):
            st.markdown("**Status**")
            tag = _pill(f"Risk {decision.risk_level}", _risk_kind(decision.risk_level))
            tag += _pill(f"Evidence {decision.evidence_status}", _evidence_kind(decision.evidence_status))
            if report is not None:
                tag += _pill(
                    "Report: Gemini" if report.source == "gemini" else "Report: offline",
                    "note" if report.source == "gemini" else "warn",
                )
            st.markdown(tag, unsafe_allow_html=True)
            st.caption(
                "Model-estimated probability is a statistical output, not a verified "
                "physical condition."
            )
            st.button(
                "Open AI Inspection Report",
                on_click=_goto,
                args=("AI Inspection Report",),
                use_container_width=True,
            )

    _render_system_status()


def _render_system_status() -> None:
    with st.container(border=True):
        st.markdown("**System status**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Model version", metadata["model_version"][-13:])
        d2.metric("Dataset rows", f"{metadata['dataset']['n_rows']:,}")
        d3.metric("Positive rate", "3.4%")
        d4.metric("Physical condition", "Not verified")
        st.caption(
            "Verified physical machine condition: **NOT VERIFIED**. ForgeWise has no "
            "physical inspection and no ground-truth telemetry. Not approved for "
            "operational machine control."
        )


def render_analysis() -> None:
    st.markdown("#### Machine Analysis")
    st.caption(
        "Enter a reading manually or load a synthetic dataset example, then run an "
        "explicit assessment. Nothing is assessed until you press **Analyze reading**."
    )

    # Initialise widget state so sample loading can update the inputs.
    for field, key in WIDGET_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_READING[field]
    for key, value in (("in_machine", "machine-01"), ("in_age", 0), ("in_omit", "(none)")):
        if key not in st.session_state:
            st.session_state[key] = value

    with st.container(border=True):
        st.markdown("**Dataset examples**")
        act = st.columns([1, 1, 2])
        with act[0]:
            load_normal = st.button("Load example — healthy (label 0)", use_container_width=True)
        with act[1]:
            load_failure = st.button("Load example — failure (label 1)", use_container_width=True)
        st.caption(
            "Inserts a row from the AI4I dataset (synthetic). These are **dataset "
            "examples**, not real machine measurements."
        )

        if load_normal or load_failure:
            try:
                sample = _sample_row(0 if load_normal else 1)
                for field, key in WIDGET_KEYS.items():
                    st.session_state[key] = sample[field]
            except Exception as exc:
                st.warning(f"Could not load a sample from the dataset: {exc}")

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
                        d = make_decision(sample, model_probability=p, model_version=metadata["model_version"])
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
                    st.dataframe(comparison, use_container_width=True, hide_index=True)
                    st.caption(
                        "A controlled comparison of two labelled dataset rows; the score "
                        "is a model estimate for simulated inputs, not verified fact."
                    )
                except Exception as exc:
                    st.warning(f"Comparison unavailable: {exc}")

    with st.container(border=True):
        st.markdown("**Sensor reading**")
        c1, c2, c3 = st.columns(3)
        with c1:
            r_type = st.selectbox("Product type", ["L", "M", "H"], key="in_type")
            air = st.number_input("Air temperature [K]", step=0.1, format="%.1f", key="in_air")
            process = st.number_input("Process temperature [K]", step=0.1, format="%.1f", key="in_process")
        with c2:
            rpm = st.number_input("Rotational speed [rpm]", step=1.0, format="%.0f", key="in_rpm")
            torque = st.number_input("Torque [Nm]", step=0.1, format="%.1f", key="in_torque")
            wear = st.number_input("Tool wear [min]", step=1.0, format="%.0f", key="in_wear")
        with c3:
            machine_id = st.text_input("Machine ID", key="in_machine")
            age_minutes = st.slider(
                "Simulated reading age (minutes)", min_value=0, max_value=240, step=5,
                key="in_age", help="Demonstrates stale-evidence detection.",
            )
            omit_field = st.selectbox("Simulate missing field (optional)", OMIT_CHOICES, key="in_omit")

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

    gen_report = st.checkbox(
        "Also generate an AI report for this assessment",
        key="in_genreport",
        help=(
            "Uses Gemini when configured, otherwise the offline deterministic "
            "template. Never required for an assessment."
        ),
    )

    signature = _reading_signature(fields, machine_id, age_minutes)
    analyze = st.button("Analyze reading", type="primary", key="btn_analyze")

    if analyze:
        timestamp = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
        reading = SensorReading(machine_id=machine_id, timestamp=timestamp, **fields)
        assessment = orchestration.run_assessment(reading, metadata["model_version"])

        if not assessment.ok:
            st.session_state["assessment"] = None
            st.session_state["report"] = None
            st.session_state["assessment_signature"] = signature
            st.session_state["last_input_errors"] = assessment.input_errors
        else:
            result = None
            if gen_report:
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
        st.error("Input validation failed:\n\n" + "\n".join(f"- {e}" for e in assessment.input_errors))
        return

    if st.session_state.get("assessment_signature") != signature:
        st.info("Inputs have changed since the last analysis — press **Analyze reading** to refresh.")

    _render_results(assessment)


def _render_results(assessment: orchestration.Assessment) -> None:
    decision = assessment.decision
    probability = assessment.probability
    prob_display = "—" if probability is None else f"{probability:.3f}"

    st.divider()
    st.markdown("### Assessment results")
    st.caption("Recomputed from the inputs above. Model output is an estimate, not a fact.")

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Model-estimated probability", prob_display,
            help="A statistical estimate, not a physical measurement or a guarantee.",
        )
        m2.metric(
            "Risk category", decision.risk_level,
            help="A policy band on the probability (LOW < 0.30, HIGH >= 0.70). A routing label, not evidence.",
        )
        m3.metric(
            "Evidence completeness", decision.evidence_status,
            help="Data-quality status: whether required fields are present and fresh.",
        )
        m4.metric("Decision", decision.decision)
        st.markdown(_band_bar(probability), unsafe_allow_html=True)
        st.caption(
            "A probability of 0.000 means every tree in the forest voted \"no failure\" "
            "— a coarse estimate, not a certainty."
        )

    with st.expander("What these numbers mean — they are NOT the same"):
        st.markdown(
            "- **Model-estimated probability** — the classifier's statistical output. "
            "It is *not* a physical measurement and *not* a guarantee.\n"
            "- **Risk category** — a documented *policy band* applied to that "
            "probability (LOW < 0.30, MODERATE 0.30–0.70, HIGH ≥ 0.70). It routes "
            "attention; it is derived from the score, not independent evidence.\n"
            "- **Evidence completeness** — a *data-quality* status: are required "
            "fields present and fresh? Independent of the score.\n"
            "- **Verified physical machine condition** — whether the machine's true "
            "state is actually known. Here it is **NOT VERIFIED**."
        )

    col_pred, col_evi = st.columns(2)
    with col_pred:
        with st.container(border=True):
            st.markdown("**Model prediction (estimate)**")
            st.write(f"- Failure probability: **{prob_display}**")
            st.write(f"- Risk category: **{decision.risk_level}**")
            st.write(f"- Model version: `{decision.model_version}`")
            st.write(f"- Confidence label: {decision.confidence}")
            st.caption("Statistical output — not a verified fact.")
    with col_evi:
        with st.container(border=True):
            st.markdown("**Verified evidence (measured inputs)**")
            evidence = asdict(assessment.reading)
            for key in FIELD_LABELS:
                value = evidence.get(key)
                label = "MISSING" if value is None else value
                st.write(f"- {FIELD_LABELS[key]}: {label}")

    with st.container(border=True):
        st.markdown("**Evidence gaps and conflicts**")
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
            marker = {"ok": "✅", "warning": "🟡", "error": "⛔", "skipped": "⏭️"}.get(stage.status, "•")
            st.markdown(f"{marker} **{stage.title}** — {stage.detail}")

    with st.expander("Traceability / rationale"):
        for line in decision.rationale:
            st.markdown(f"- {line}")


def render_report() -> None:
    st.markdown("#### AI Inspection Report")
    st.caption(
        "An optional explanation layer produced by Gemini (when configured) or the "
        "offline deterministic template. It never replaces the decision rules."
    )

    assessment = st.session_state.get("assessment")
    report = st.session_state.get("report")

    if assessment is None or assessment.decision is None:
        st.info("No assessment yet. Run one from **Machine Analysis** first.")
        st.button(
            "Open Machine Analysis",
            type="primary",
            on_click=_goto,
            args=("Machine Analysis",),
        )
        return

    if report is None:
        st.info("No AI report generated for the latest assessment yet.")
        if st.button("Generate AI Report", type="primary"):
            result = reporting.generate_report_result(
                _report_payload(assessment), api_key=resolved_key or None
            )
            st.session_state["report"] = result
            st.rerun()
        return

    if report.source == "gemini":
        st.success(f"**Gemini-generated** (`{report.model_name}`), grounded in the assessment payload.")
    else:
        st.warning(
            "**Offline fallback** — an offline deterministic template generated this "
            "report; Gemini was not used."
            + (f" Reason: {report.fallback_reason}" if report.fallback_reason else "")
        )
    if report.error:
        st.caption(f"Provider detail: {report.error}")

    st.markdown(report.text)

    with st.expander("Report provenance"):
        st.write(f"- Source: `{report.source}`")
        st.write(f"- Model: `{report.model_name or 'n/a'}`")
        st.write(f"- Fallback: {report.is_fallback}")
        if report.missing_sections:
            st.write(f"- Missing sections: {', '.join(report.missing_sections)}")


def render_history() -> None:
    st.markdown("#### Decision History")
    st.caption(
        "Records are written to the app container's local filesystem. On a hosted "
        "deployment this storage is **ephemeral** — records are lost on restart or "
        "redeploy, and this is **not** durable audit storage."
    )

    records = recent_decisions(25)
    if not records:
        st.info("No decisions recorded yet. Run an assessment from **Machine Analysis**.")
        st.button(
            "Open Machine Analysis", type="primary",
            on_click=_goto, args=("Machine Analysis",),
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
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_diagnostics() -> None:
    st.markdown("#### Model Diagnostics")
    st.caption(
        "Technical detail, separated from the operator workflow. Metrics are computed "
        "on the held-out test split only — no test-set tuning. Numbers are read from "
        "committed artifacts; none are hand-entered."
    )

    audit = load_dataset_audit()

    if metrics:
        with st.container(border=True):
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
                [[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            )
            st.markdown("**Confusion matrix (held-out test)**")
            st.dataframe(cm_df, use_container_width=True)
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
            use_container_width=True, hide_index=True,
        )
        st.markdown("**Held-out test (reference only)**")
        st.dataframe(
            [
                _row("current (0.5)", ref["metrics_at_current_0.5"]),
                _row(f"selected ({val['selected_threshold']})", ref["metrics_at_selected"]),
            ],
            use_container_width=True, hide_index=True,
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
