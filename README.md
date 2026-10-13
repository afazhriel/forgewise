# ForgeWise — The Missing Evidence Engine

Evidence-based maintenance decision support for manufacturing. ForgeWise estimates
machine-failure risk from sensor data **and makes the difference between a model
prediction and a verified fact explicit** — flagging missing or stale evidence and
recommending rule-based follow-up checks.

> AI Builder Cup 2026 — small, working, auditable MVP.

---

## 1. What it does

1. **Predicts** failure probability from a machine's sensor snapshot (scikit-learn baseline).
2. **Separates** that prediction from *verified evidence* (the measured sensor values).
3. **Detects missing / stale evidence** (absent fields, unknown freshness, out-of-range values, inconclusive model output).
4. **Recommends follow-up checks** with explicit, documented rules — never a black box.
5. **Generates a traceable inspection report**, optionally with Google Gemini, grounded strictly in the supplied evidence.
6. **Records an audit trail**: inputs, model version, outputs, and timestamps.

The guiding idea: in maintenance, the dangerous answer is a confident guess built
on missing data. ForgeWise refuses to hide that gap.

---

## 1a. Product shell

The Streamlit app is a small industrial-style product shell with sidebar
navigation. Nothing is assessed until the operator presses **Analyze machine**.
The primary flow is explicit and linear: **1** enter machine inputs → **2**
analyze machine → **3** review the assessment → **4** optionally generate an AI
report. Visual Inspection and Equipment Integration extend that flow; Decision
History and Model Diagnostics are reference views kept out of the main flow.

| View | Purpose |
| --- | --- |
| **Overview** | Last assessment summary: KPI cards, the model score plotted against the documented policy bands, evidence gaps, next action |
| **Machine Analysis** | Sensor inputs grouped in cards, sample loading, explicit **Analyze machine** action, a three-part result hierarchy, optional report request |
| **AI Inspection Report** | Gemini or offline-template report via an explicit **Generate AI Report** button, with distinct loading/success/failure states and real provenance |
| **Visual Inspection** | Two classical-CV tools (distinct tool buttons): an **image-quality check** and a **reference-comparison** difference detector, plus a detector interface |
| **Equipment Integration** | Reference view: four equipment templates (REST/MQTT/Modbus/CSV), a live payload validator, and the ingestion queue |
| **Decision History** | Reference view: summary KPIs plus traceable decision events (time, asset, model, score, evidence, recommendation, report mode) |
| **Model Diagnostics** | Reference view: held-out metrics, confusion matrix, ROC/PR plots, a **real** precision/recall/F1-vs-threshold chart from `threshold_analysis.json`, dataset audit, limitations |

The UI is deliberately spare: a branded header band with honest status pills
(model version, synthetic dataset, manual/simulated input, AI provider
configuration), KPI cards, and a probability-vs-risk-band marker. **Every chart
and number is derived from a committed artifact or the live model output** —
there are no decorative buttons, placeholder charts, fake metrics, or real-time
claims. A detected Gemini key is reported as *configuration only*; the app never
claims Gemini "works" until a real request has actually succeeded.

The results hierarchy separates the four things that are easy to conflate and
states each explicitly:
**model-estimated probability** (a statistical estimate), **risk category** (a
policy band derived from it), **evidence completeness** (data quality), and the
**operational decision** (rule-based). It also states plainly that the verified
physical machine condition is **NOT VERIFIED**.

## 1b. Architecture & workflow

Presentation (`app.py`) is kept separate from the deterministic pipeline
(`src/orchestration.py`). An assessment runs through explicit stages:

1. **Input validation** — structural checks (unknown product type, non-finite values).
2. **Model inference** — `src/predict.py` runs the existing RandomForest.
3. **Evidence completeness & freshness** — `src/decision_engine.py`.
4. **Decision rules** — the documented risk/decision mapping.
5. **Follow-up checks** — permitted, rule-based checks only.
6. **Optional report** — `src/reporting.py`; Gemini if configured, else offline template.
7. **Traceable decision event** — appended to the local decision log.

Each stage records its own status (`ok` / `warning` / `error` / `skipped`), so the
workflow is transparent and testable. There is no agent framework and no
autonomy: the LLM never bypasses the deterministic rules or overrides the model.

---

## 2. Dataset

* **Name:** AI4I 2020 Predictive Maintenance Dataset
* **Source:** <https://archive.ics.uci.edu/dataset/601/ai4i+2020+>
* **Download URL (used by `src/data_loader.py`):**
  `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
* **License:** CC BY 4.0
* **Verified schema (14 columns, 10,000 rows):** `UDI`, `Product ID`, `Type`,
  `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
  `Torque [Nm]`, `Tool wear [min]`, `Machine failure` (target), and the failure-mode
  flags `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
* **Target distribution:** 339 failures / 9,661 non-failures (3.39% positive).
* **Data quality:** 0 missing values, 0 duplicate rows (audited, not assumed).
* The raw CSV/zip is **downloaded, not committed** (see `.gitignore`).

To fetch it:

```powershell
.\.venv\Scripts\python.exe -m src.data_loader   # downloads + prints an honest audit
```

---

## 3. Setup

Requires **Python 3.12.x** (tested on 3.12.10). All paths use `pathlib` and work
on Linux, macOS, and Windows. Commands below are Windows PowerShell; on
Linux/macOS use the activated venv's `python` / `streamlit` directly
(e.g. `python -m streamlit run app.py`).

```powershell
# from C:\Dev\forgewise
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pinned dependencies (also see `requirements.txt`): pandas, numpy, scikit-learn,
joblib, matplotlib, **Pillow** (computer-vision tools), streamlit, pytest, and the
optional `google-genai`. The equipment templates additionally support optional
`paho-mqtt` (MQTT) and `pymodbus` (Modbus); neither is required by the app or its
tests.

---

## 4. Reproduce the pipeline

```powershell
# 1. Acquire + audit the dataset (prints row/column counts, target mix, ranges)
.\.venv\Scripts\python.exe -m src.data_loader

# 2. Train the baseline (writes models/model.joblib + models/model_metadata.json)
.\.venv\Scripts\python.exe -m src.train_model

# 3. Evaluate on the held-out test split (writes artifacts/metrics.json + figures)
.\.venv\Scripts\python.exe -m src.evaluate_model

# 3b. Select a threshold on a validation split (writes artifacts/threshold_analysis.json)
.\.venv\Scripts\python.exe -m src.calibrate_threshold

# 4. Run the test suite
.\.venv\Scripts\python.exe -m pytest -q

# 5. Launch the app
.\.venv\Scripts\streamlit.exe run app.py
```

Reproducibility controls: fixed seed (`random_state=42`), stratified 80/20 split,
the exact test row indices saved in `models/model_metadata.json` (so evaluation is
never re-tuned), a dataset SHA-256 recorded in the model metadata, and an explicit
`model_version`.

---

## 5. Model card

* **Model:** `RandomForestClassifier` (`class_weight="balanced_subsample"`, 400 trees).
* **Features:** `Type`, `Air temperature [K]`, `Process temperature [K]`,
  `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
* **Excluded (leakage):** `UDI`, `Product ID`, and the failure-mode flags
  `TWF/HDF/PWF/OSF/RNF` (they directly encode the label).
* **Model version (this build):** `fw-20261013T074119Z-dc6630cd` (committed
  reference run). A fresh deployment **retrains deterministically** with the same
  configuration and reproduces the metrics below under a new timestamped version
  string, because the model binary itself is not committed (see §8).

Held-out test metrics (2,000 rows, 3.4% positive) — actual values from
`artifacts/metrics.json`:

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.968 |
| PR-AUC (average precision) | 0.773 |
| Recall @ 0.5 | 0.412 |
| Precision @ 0.5 | 0.933 |
| F1 @ 0.5 | 0.571 |
| Accuracy | 0.979 |

Confusion matrix @ 0.5: TN=1930, FP=2, FN=40, TP=28.

**Honest limitations.** The dataset is synthetic/simulated and heavily imbalanced.
At the default 0.5 threshold the model is high-precision but misses ~59% of
failures — a deliberate consequence of not tuning on the test set. Recall and
PR-AUC are the informative metrics here. These numbers are **not** claims about
real-world plant performance; production use would require calibration, a
representative dataset, and threshold selection tied to maintenance costs.

### Threshold calibration (validation-selected, no test-set tuning)

The 0.5 threshold above is an arbitrary default. `src/calibrate_threshold.py`
selects a threshold on a **validation** split carved out of the *training pool*
and reports its effect on the untouched held-out test set. The held-out test
indices are never used to choose the threshold, and `artifacts/metrics.json` is
left unchanged. Selected threshold: **0.20** (by maximising validation F1; ties
break toward recall). Results (`artifacts/threshold_analysis.json`):

| Split | Threshold | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validation | 0.50 | 0.893 | 0.368 | 0.521 | 1929 | 3 | 43 | 25 |
| Validation | **0.20** | 0.639 | 0.779 | 0.702 | 1902 | 30 | 15 | 53 |
| Held-out test (reference) | 0.50 | 0.933 | 0.412 | 0.571 | 1930 | 2 | 40 | 28 |
| Held-out test (reference) | **0.20** | 0.581 | 0.794 | 0.671 | 1893 | 39 | 14 | 54 |

Moving to 0.20 roughly **doubles recall** (0.41 → 0.79 on the held-out set) at
the cost of precision — the right trade for maintenance, where a missed failure
is far costlier than an unnecessary inspection. The app's risk bands
(LOW `< 0.30`, HIGH `>= 0.70`) are a **separate documented policy** and are not
changed automatically by this analysis; the calibration is shown in the app's
Diagnostics tab for transparency.

---

## 6. The Missing Evidence Engine

`src/decision_engine.py` is the trust layer. It tags every statement as:

* `[MODEL]` — a statistical prediction (not a fact)
* `[EVIDENCE]` — a measured value
* `[MISSING]` — information the engine needs but does not have

**Evidence status** (precedence: missing > stale > no model > inconclusive):
`COMPLETE`, `INCOMPLETE`, `STALE`, `NO_MODEL_OUTPUT`, `INCONCLUSIVE`.

**Decision states:** `REQUEST_DATA`, `RECAPTURE_READING`, `ESCALATE_INSPECTION`,
`SCHEDULE_CHECK`, `MONITOR`.

**Mechanism heuristics** (from the public AI4I dataset documentation, used only to
*recommend checks* — never as the model's prediction):

| Flag | Condition |
| --- | --- |
| HDF — heat dissipation | process−air temperature < 8.6 K **and** speed < 1380 rpm |
| PWF — power failure | mechanical power < 3500 W or > 9000 W |
| OSF — overstrain | tool_wear × torque above a type threshold (L:11000, M:12000, H:13000) |
| TWF — tool wear | tool wear in the 200–240 min band |

Policy thresholds (documented, not learned): low `< 0.30`, high `>= 0.70`,
inconclusive band `[0.30, 0.70]`, stale after `60` minutes. (These routing bands
are separate from the validation-selected 0.5→0.20 probability threshold in §5,
which informs a decision but does not auto-change the bands.)

Every assessment is appended to `artifacts/decision_log.jsonl` with the full input,
the decision, the model version, a UTC timestamp, and the report mode
(`gemini` / `template`). Each **explicit Analyze machine action** writes one record;
the log is no longer written on every UI rerun.

> **Durability.** On a hosted deployment the log lives on the app container's
> ephemeral filesystem and is lost on restart or redeploy — it is **not** durable
> audit storage. For production, point the log at managed storage.

### The four things the UI keeps distinct

The app deliberately separates four ideas that are easy to conflate:

1. **Model-estimated probability** — the classifier's statistical output. Not a
   physical measurement and not a guarantee. A value of `0.000` simply means every
   tree in the forest voted "no failure" — a coarse estimate, not a certainty.
2. **Risk category** — a *policy band* applied to that probability (LOW/MODERATE/
   HIGH). It routes attention; it is derived from the score, not new evidence.
3. **Evidence completeness** — a *data-quality* status (are required fields present
   and fresh?), independent of the score. A reading can be high-risk yet
   incomplete, or low-risk and complete.
4. **Verified physical machine condition** — whether the machine's true state is
   actually known. In this prototype it is **NOT VERIFIED**: there is no physical
   inspection and no ground-truth telemetry.

### Data provenance in the UI

Every reading is **manually entered** or a **simulated dataset example** — the app
is explicit that this is *not* live industrial telemetry and that no machine has
been inspected. The "Load example — healthy (label 0)" / "failure (label 1)"
buttons insert labelled rows from the AI4I dataset, and a built-in comparison runs
the model on one healthy and one failure row side by side (labelled as dataset
examples). This makes the healthy-vs-failure contrast reproducible rather than
implied.

---

## 6a. Computer vision (classical, offline)

`src/vision.py` provides **two** deterministic, offline tools (Pillow + NumPy
only — no OpenCV, no network, no trained model). They are exposed as two distinct
tools on the **Visual Inspection** page, and neither one diagnoses a machine or
claims to detect defects:

1. **Image quality / usability** — resolution, brightness, contrast, sharpness
   (variance of the Laplacian) and colour balance, with a **`USABLE` /
   `RECAPTURE`** verdict. This is an evidence-usability gate, the visual analogue
   of the sensor completeness checks.
2. **Reference comparison / visual difference** — compares a known-good reference
   photo against a current photo of the same view and reports the changed
   fraction, mean absolute difference, a *global* SSIM approximation, and the
   bounding box of changes. This is classical change detection: it reports
   **visual differences**, not defect names, and always requires human review.

A **detector interface** (`load_detector`) is included for a future trained model,
but it deliberately returns *"no defect-detection model is installed"* until a
model artifact **and** a supported runtime are actually present. The UI never
fabricates detections.

Thresholds are documented constants in `src/vision.py` and are covered by
`tests/test_vision.py` using synthetic images (noise vs. blurred vs. dark vs.
tiny; identical vs. altered reference).

## 6b. Equipment integration templates

`integrations/` contains **four** starting-point templates so physical
manufacturing equipment can feed readings into ForgeWise. Every template validates
incoming data with the shared, unit-tested normaliser in `src/ingest.py` and
appends valid readings to one queue (`artifacts/ingest_queue.jsonl`, git-ignored).

| # | Template | Transport | Optional dependency |
| --- | --- | --- | --- |
| 1 | `integrations/rest_ingest.py` | HTTP POST `/ingest` | none (stdlib) |
| 2 | `integrations/mqtt_subscriber.py` | MQTT subscribe | `paho-mqtt` |
| 3 | `integrations/modbus_poller.py` | Modbus/TCP poll (OPC-UA-style mapping) | `pymodbus` |
| 4 | `integrations/csv_scada_import.py` | CSV batch (SCADA/MES export) | none (pandas) |

The **Equipment Integration** page documents the payload schema, lets you paste a
JSON payload and **validate it live** with the same normaliser, and can run an
assessment directly from a valid payload. The optional `paho-mqtt` / `pymodbus`
dependencies are **not** required by the app or its test suite.

The payload schema (case-insensitive; aliases accepted):

```json
{
  "machine_id": "cnc-07", "type": "M",
  "air_temperature": 298.1, "process_temperature": 308.6,
  "rotational_speed": 1500, "torque": 40.0, "tool_wear": 50
}
```

A **missing** sensor field becomes *missing evidence* (`INCOMPLETE`), not an error;
non-numeric values and unknown product types are errors; unknown extra keys are
reported but ignored. These templates are examples — add authentication, TLS, and
durable storage before any real deployment.

---

## 7. Gemini report (optional)

Gemini is an **explanation layer**, not the predictor. `src/reporting.py` exposes a
small provider adapter (`GeminiProvider`) that wraps the supported `google-genai`
SDK. The key is resolved in this order and is **never** hardcoded, logged, rendered,
or written to the decision history:

1. Streamlit **Secrets** (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) — the recommended
   path for a hosted deployment;
2. the environment (`GEMINI_API_KEY` / `GOOGLE_API_KEY`);
3. the app's password-masked session field (a manual override).

The sidebar always reports the provider **configuration** honestly: *a Gemini key
was detected* (with the source) once a key is found, otherwise *no Gemini key
detected — offline template*. A detected key is **not** a claim that Gemini works;
the app only states *Gemini-generated* after a real request has returned and passed
validation. Generation is triggered by an explicit **Generate AI Report** button
(or an opt-in checkbox that requests a report during analysis); it is never
automatic, and no request is made until the user asks for one.

The report UI shows three distinct states — a **loading** spinner while the request
is in flight, a **success** banner naming Gemini when a validated Gemini response is
used, and a **failure** banner when the request errored (it then shows the offline
template and the failure reason), or an informational banner when no key is
configured. A configured key never, by itself, changes the reported outcome.

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

Key behaviours:

* **Grounded prompt** — the payload (reading + decision) is supplied verbatim; the
  system instruction forbids inventing sensor values, failure causes, inspection
  history, or maintenance actions.
* **Structured-output validation** — the response must contain the expected
  sections (Summary, Risk, Evidence, Missing, Recommended, Limitations, Human
  verification). Empty, truncated, or malformed output is rejected.
* **Explicit provenance** — every report is labelled **Gemini** or **offline
  deterministic template**; a fallback is never mislabelled as Gemini, and the
  fallback reason (missing key, API error, malformed output) is shown.
* **Graceful degradation** — missing credentials, SDK import errors, API errors,
  and timeouts all fall back to the template (never a crash). With no key/SDK the
  offline template always works, so the demo needs no paid service.

The provider is injectable (`provider=...`) so the test suite exercises success,
API-error, and malformed-output paths with a fake client — **no live API call is
ever made during development or tests**.

---

## 8. Deploy to Streamlit Community Cloud

The repository is ready to deploy as a public Streamlit app.

* **Repository:** `afazhriel/forgewise`
* **Branch:** `main`
* **Entrypoint (main file path):** `app.py`
* **Python version:** 3.12 (pinned via `runtime.txt` → `python-3.12`)
* **Dependencies:** installed from `requirements.txt` (already pinned)

### Startup model behaviour

The trained model binary (`models/model.joblib`, ~15 MB) is **not** committed.
On a fresh deployment `src/model_store.py`:

1. reuses `models/model.joblib` when present (local development); otherwise
2. trains the deterministic baseline from the official UCI AI4I dataset, which
   `src/data_loader.py` downloads over HTTPS on first run, then caches the model
   in-process (and writes it to `models/`).

The first request is therefore slower (dataset download + training, typically
well under a minute); later requests reuse the cached model. If the dataset
cannot be reached, the app shows a clear, actionable error instead of crashing.
Only the official UCI archive is contacted — no executable artifacts are fetched
from untrusted sources, and the schema is validated before training.

> Because the model is retrained at deploy time, its *version string* differs
> from the committed reference run, but the metrics are identical (same seed,
> features, and split — verified by reproducing `artifacts/metrics.json` from a
> fresh training run).

### Steps

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → select repository `afazhriel/forgewise`, branch `main`, main
   file path `app.py`.
3. (Optional) Under **Advanced settings → Secrets**, add a Gemini key to enable
   the LLM report path:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
   Without it, the app uses the offline template report. Never commit this key.
4. Deploy. No `.env`, committed dataset, or committed model binary is required.

The public app URL is assigned by Streamlit on deploy; this document does not
invent one.

---

## 9. Project layout

```text
forgewise/
├── app.py                      # Streamlit product shell (deployment entrypoint)
├── conftest.py                 # makes `src` importable under pytest
├── runtime.txt                 # python-3.12 for Streamlit Community Cloud
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml      # restrained light theme
├── data/                       # raw dataset downloaded here (git-ignored)
├── models/                     # trained model (git-ignored) + committed metadata
├── artifacts/                  # metrics/figures + threshold_analysis.json tracked; logs/queues git-ignored
├── integrations/               # 4 equipment templates (REST, MQTT, Modbus, CSV) + README
│   ├── rest_ingest.py
│   ├── mqtt_subscriber.py
│   ├── modbus_poller.py
│   └── csv_scada_import.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # download, schema validation, audit
│   ├── train_model.py          # reproducible baseline
│   ├── evaluate_model.py       # held-out evaluation + figures
│   ├── calibrate_threshold.py  # validation-set threshold selection (no test tuning)
│   ├── orchestration.py        # deterministic, stage-annotated assessment pipeline
│   ├── model_store.py          # portable load-or-train-at-startup
│   ├── decision_engine.py      # evidence/decision rules (pure, testable)
│   ├── predict.py              # SensorReading -> probability
│   ├── vision.py               # classical CV: image quality + reference comparison
│   ├── ingest.py               # shared equipment-payload normaliser
│   └── reporting.py            # Gemini provider adapter + offline template
└── tests/
    ├── test_data_validation.py
    ├── test_decision_engine.py
    ├── test_model_store.py
    ├── test_reporting.py       # provider adapter, validation, fallback
    ├── test_predictions.py     # healthy/failure examples + input sensitivity
    ├── test_calibrate_threshold.py
    ├── test_orchestration.py   # pipeline stages + input validation + human review
    ├── test_vision.py          # image-quality + reference-comparison metrics
    ├── test_ingest.py          # payload normaliser
    ├── test_integrations.py    # the 4 equipment templates
    └── test_app.py             # navigation, analyze flow, evidence states, report states, history
```

---

## 10. License / attribution

Dataset © UCI Machine Learning Repository, licensed **CC BY 4.0**. ForgeWise source
code is provided for the AI Builder Cup 2026 prototype.
