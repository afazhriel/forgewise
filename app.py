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

from src import data_loader, model_store, predict, reporting
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
st.warning(
    "**Demo data only.** Every reading in this app is **manually entered** or a "
    "**simulated dataset example**. This is **not** live industrial telemetry, "
    "and no physical machine has been inspected.",
    icon="🧪",
)

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
        st.caption(
            "Reference held-out evaluation (committed). The runtime model is "
            "reconstructed from the same deterministic configuration."
        )
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
    load_normal = st.button("Load example — healthy (label 0)", use_container_width=True)
with col_actions[1]:
    load_failure = st.button("Load example — failure (label 1)", use_container_width=True)
st.caption(
    "These buttons insert rows from the AI4I dataset (synthetic/simulated). "
    "They are **dataset examples**, not real machine measurements."
)

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


if load_normal or load_failure:
    try:
        st.session_state.reading = _sample_row(0 if load_normal else 1)
    except Exception as exc:
        st.warning(f"Could not load a sample from the dataset: {exc}")

with st.expander("Compare a healthy vs a failure dataset example"):
    st.caption(
        "Runs the current model on one labelled healthy row and one labelled "
        "failure row from the AI4I dataset, side by side. Dataset examples only."
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
                "This is a controlled comparison of two labelled dataset rows; the "
                "score is a model estimate for simulated inputs, not verified fact."
            )
        except Exception as exc:
            st.warning(f"Comparison unavailable: {exc}")

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

# Log only when the *reading* actually changes, so the audit trail stays
# meaningful instead of recording an identical row on every Streamlit rerun.
signature = (
    r_type, round(air, 4), round(process, 4), round(rpm, 4), round(torque, 4),
    round(wear, 4), machine_id, age_minutes, omit_field,
)
if st.session_state.get("last_logged_signature") != signature:
    log_decision(decision, reading)
    st.session_state["last_logged_signature"] = signature

st.caption(
    "The assessment runs automatically: the score, risk level and decision are "
    "recomputed every time any input above changes."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Model-estimated probability", f"{probability:.3f}",
    help="A statistical estimate, not a physical measurement or a guarantee.",
)
m2.metric(
    "Risk category", decision.risk_level,
    help="A policy band on the probability (LOW < 0.30, HIGH >= 0.70). A routing label, not new evidence.",
)
m3.metric(
    "Evidence completeness", decision.evidence_status,
    help="Data-quality status: whether required fields are present and fresh.",
)
m4.metric("Decision", decision.decision)

st.progress(min(max(probability, 0.0), 1.0))
st.caption(
    "A probability of 0.000 means every tree in the forest voted "
    "\"no failure\" for this reading — a coarse estimate, not a physical certainty."
)

with st.expander("What these four numbers mean — they are NOT the same"):
    st.markdown(
        "- **Model-estimated probability** — the classifier's statistical output. "
        "It is *not* a physical measurement and *not* a guarantee. `0.000` simply "
        "means the model's trees all voted \"no failure\".\n"
        "- **Risk category** — a documented *policy band* applied to that "
        "probability (LOW < 0.30, MODERATE 0.30–0.70, HIGH ≥ 0.70). It routes "
        "attention; it is derived from the score, not independent evidence.\n"
        "- **Evidence completeness** — a *data-quality* status: are the required "
        "fields present and fresh? It is independent of the score (a reading can "
        "be high-risk yet incomplete, or low-risk and complete).\n"
        "- **Verified physical machine condition** — whether we actually *know* "
        "the machine's true state. In this prototype it is **NOT VERIFIED** (see "
        "below): there is no physical inspection or ground-truth telemetry."
    )

st.info(
    "**Verified physical machine condition: NOT VERIFIED.** ForgeWise has no "
    "physical inspection and no ground-truth telemetry. Everything on this screen "
    "is a model estimate over manually entered or simulated data.",
    icon="🏭",
)

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

    st.divider()
    st.subheader("Threshold calibration (validation-selected)")
    calibration = load_calibration()
    if calibration:
        val = calibration["validation"]
        ref = calibration["held_out_test_reference"]
        st.caption(
            f"Selected threshold **{val['selected_threshold']}** by maximising "
            f"{calibration['selection_metric']} on a validation split carved from the "
            "training pool (seed "
            f"{val['random_state']}). The held-out test set was **not** used to "
            "select it; `artifacts/metrics.json` is unchanged."
        )

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
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Held-out test (reference only — not used for selection)**")
        st.dataframe(
            [
                _row("current (0.5)", ref["metrics_at_current_0.5"]),
                _row(f"selected ({val['selected_threshold']})", ref["metrics_at_selected"]),
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(ref["note"])
        st.caption(calibration["policy_note"])
    else:
        st.caption(
            "No threshold analysis found. Run `python -m src.calibrate_threshold` "
            "to generate `artifacts/threshold_analysis.json`."
        )
