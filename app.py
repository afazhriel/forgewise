"""ForgeWise — The Missing Evidence Engine (Streamlit UI).

Run:
    .venv\\Scripts\\streamlit.exe run app.py

The UI makes the trust layer visible: it shows the model *prediction*
separately from *verified evidence*, surfaces *missing/stale* evidence, and
lists *rule-based* follow-up checks. An optional Gemini report is grounded in
exactly the payload shown on screen.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from src import data_loader, predict, reporting
from src.decision_engine import SensorReading, log_decision, make_decision

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS / "metrics.json"
LOG_PATH = ARTIFACTS / "decision_log.jsonl"

st.set_page_config(page_title="ForgeWise — Missing Evidence Engine", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    return data_loader.load_dataset()


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def model_ready() -> bool:
    from src.train_model import METADATA_PATH, MODEL_PATH

    return MODEL_PATH.exists() and METADATA_PATH.exists()


def load_metadata():
    from src.train_model import METADATA_PATH

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def recent_decisions(limit: int = 8) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Header + model status
# ---------------------------------------------------------------------------
st.title("ForgeWise — The Missing Evidence Engine")
st.caption(
    "Evidence-based maintenance decision support. Model predictions are kept "
    "separate from verified facts, and missing/stale evidence is made explicit."
)

if not model_ready():
    st.error(
        "No trained model found. Run training first:\n\n"
        "`.\\.venv\\Scripts\\python.exe -m src.train_model`"
    )
    st.stop()

metadata = load_metadata()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Sidebar: model card + Gemini key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model card")
    st.markdown(f"**Version:** `{metadata['model_version']}`")
    st.markdown(f"**Type:** {metadata['model_type']}")
    st.markdown(f"**Target:** `{metadata['target']}`")
    st.markdown(f"**Trained:** {metadata['trained_at'][:19]}Z")
    st.markdown(f"**Dataset rows:** {metadata['dataset']['n_rows']}")
    st.markdown(f"**Dataset SHA-256:** `{metadata['dataset']['sha256'][:16]}...`")

    if metrics:
        st.subheader("Held-out test metrics")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c2.metric("PR-AUC", f"{metrics['average_precision']:.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Recall", f"{metrics['recall']:.3f}")
        c4.metric("Precision", f"{metrics['precision']:.3f}")
        st.caption(metrics["note"])

    st.divider()
    st.header("Gemini report (optional)")
    st.caption(
        "Leave blank to use the offline, deterministic template. The key is held "
        "only in this session and never written to disk."
    )
    api_key = st.text_input("GEMINI_API_KEY", type="password", value="")

    st.divider()
    st.caption(
        "Sources: UCI AI4I 2020 Predictive Maintenance Dataset "
        f"({data_loader.LICENSE})."
    )

# ---------------------------------------------------------------------------
# Sensor input
# ---------------------------------------------------------------------------
st.header("1. Sensor reading")

col_actions = st.columns([1, 1, 2])
with col_actions[0]:
    load_normal = st.button("Load healthy sample", use_container_width=True)
with col_actions[1]:
    load_failure = st.button("Load failure sample", use_container_width=True)

if "reading" not in st.session_state:
    st.session_state.reading = {
        "type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 50.0,
    }


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


if load_normal:
    st.session_state.reading = _sample_row(0)
if load_failure:
    st.session_state.reading = _sample_row(1)

current = st.session_state.reading

c1, c2, c3 = st.columns(3)
with c1:
    r_type = st.selectbox("Product type", ["L", "M", "H"], index=["L", "M", "H"].index(current["type"]))
    air = st.number_input("Air temperature [K]", value=float(current["air_temperature"]), step=0.1)
    process = st.number_input("Process temperature [K]", value=float(current["process_temperature"]), step=0.1)
with c2:
    rpm = st.number_input("Rotational speed [rpm]", value=float(current["rotational_speed"]), step=1.0)
    torque = st.number_input("Torque [Nm]", value=float(current["torque"]), step=0.1)
    wear = st.number_input("Tool wear [min]", value=float(current["tool_wear"]), step=1.0)
with c3:
    machine_id = st.text_input("Machine ID", value="machine-01")
    age_minutes = st.slider(
        "Simulated reading age (minutes)", min_value=0, max_value=240, value=0, step=5,
        help="Demonstrates stale-evidence detection.",
    )
    omit_field = st.selectbox(
        "Simulate missing field (optional)",
        ["(none)", "torque", "tool_wear", "air_temperature", "process_temperature", "rotational_speed"],
    )

fields = {
    "type": r_type,
    "air_temperature": air,
    "process_temperature": process,
    "rotational_speed": rpm,
    "torque": torque,
    "tool_wear": wear,
}
if omit_field != "(none)":
    fields[omit_field] = None

timestamp = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
reading = SensorReading(
    machine_id=machine_id,
    timestamp=timestamp,
    **fields,
)

# ---------------------------------------------------------------------------
# 2. Prediction vs evidence
# ---------------------------------------------------------------------------
st.header("2. Assessment")

probability = predict.predict_probability(reading)
decision = make_decision(
    reading,
    model_probability=probability,
    model_version=metadata["model_version"],
)
log_decision(decision, reading)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model failure probability", f"{probability:.3f}", help="A prediction, not a verified fact.")
m2.metric("Risk level", decision.risk_level)
m3.metric("Evidence status", decision.evidence_status)
m4.metric("Decision", decision.decision)

st.progress(min(max(probability, 0.0), 1.0))

# --- evidence vs prediction ---
col_pred, col_evi = st.columns(2)
with col_pred:
    st.subheader("Prediction (model output)")
    st.markdown("_This is a statistical output, not a verified fact._")
    st.write(f"- Failure probability: **{probability:.3f}**")
    st.write(f"- Model version: `{decision.model_version}`")
    st.write(f"- Risk level: **{decision.risk_level}**")

with col_evi:
    st.subheader("Verified evidence")
    for key, value in asdict(reading).items():
        if key in ("reading_id", "machine_id", "timestamp"):
            continue
        if value is None:
            st.write(f"- {key}: :red[MISSING]")
        else:
            st.write(f"- {key}: {value}")

# --- gaps + flags ---
g_col, f_col = st.columns(2)
with g_col:
    st.subheader("Missing / stale evidence")
    if decision.evidence_gaps:
        for gap in decision.evidence_gaps:
            st.warning(gap, icon="⚠️")
    else:
        st.success("All required evidence present and fresh.", icon="✅")

with f_col:
    st.subheader("Documented mechanism heuristics")
    if decision.mechanism_flags:
        for name in decision.mechanism_flags:
            st.info(name, icon="🔎")
        st.caption(
            "Heuristics derived from the public AI4I dataset documentation; "
            "used to recommend checks, not as the model's prediction."
        )
    else:
        st.write("None triggered.")

# --- recommended checks ---
st.subheader("Recommended follow-up checks (rule-based)")
for check in decision.recommended_checks:
    st.markdown(f"- {check}")

with st.expander("Traceability / rationale"):
    for line in decision.rationale:
        st.markdown(f"- {line}")

# ---------------------------------------------------------------------------
# 3. Report generation
# ---------------------------------------------------------------------------
st.header("3. Traceable inspection report")
if st.button("Generate inspection report", type="primary"):
    payload = {"reading": asdict(reading), "decision": asdict(decision)}
    try:
        text, source = reporting.generate_report(payload, api_key=api_key or None)
    except Exception as exc:  # surfaced, not swallowed
        st.error(f"Report generation failed: {exc}")
    else:
        if source == "template":
            st.info(
                "Generated with the offline template (no Gemini key provided). "
                "Add a GEMINI_API_KEY in the sidebar to use Gemini."
            )
        else:
            st.success("Generated with Gemini, grounded in the payload above.")
        st.markdown(text)

# ---------------------------------------------------------------------------
# 4. Audit trail + figures
# ---------------------------------------------------------------------------
st.header("4. Audit trail & model diagnostics")
log_tab, diag_tab = st.tabs(["Decision log", "Diagnostics"])

with log_tab:
    records = recent_decisions()
    if not records:
        st.write("No decisions logged yet.")
    for record in records:
        d = record["decision"]
        st.markdown(
            f"`{d['created_at'][:19]}Z` — **{d['decision']}** "
            f"(risk={d['risk_level']}, p={d['model_probability']}, "
            f"evidence={d['evidence_status']}, model={d['model_version']})"
        )

with diag_tab:
    st.caption(
        "Diagnostics computed on the held-out test split only. No test-set tuning."
    )
    if metrics:
        st.json(metrics["confusion_matrix"])
    for image in ("confusion_matrix.png", "roc_pr_curves.png"):
        path = ARTIFACTS / image
        if path.exists():
            st.image(str(path), caption=image)
