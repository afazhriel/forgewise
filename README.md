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
joblib, matplotlib, streamlit, pytest, and the optional `google-genai`.

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
the decision, the model version, and a UTC timestamp. The app logs a new row
**only when the reading actually changes**, so the trail stays meaningful instead
of recording an identical row on every Streamlit rerun.

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

## 7. Gemini report (optional)

`src/reporting.py` can produce the inspection report with Google Gemini. It reads
the API key **only** from the environment (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) or
the app's session field — it is never hardcoded or logged.

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

With no key/SDK, ForgeWise falls back to a deterministic, fully offline template
report, so the demo always works. The prompt is grounded in the exact evidence
payload and instructs the model to separate evidence, predictions, and missing data.

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
├── app.py                      # Streamlit UI (deployment entrypoint)
├── conftest.py                 # makes `src` importable under pytest
├── runtime.txt                 # python-3.12 for Streamlit Community Cloud
├── requirements.txt
├── README.md
├── .gitignore
├── data/                       # raw dataset downloaded here (git-ignored)
├── models/                     # trained model (git-ignored) + committed metadata
├── artifacts/                  # metrics/figures + threshold_analysis.json tracked; decision log git-ignored
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # download, schema validation, audit
│   ├── train_model.py          # reproducible baseline
│   ├── evaluate_model.py       # held-out evaluation + figures
│   ├── calibrate_threshold.py  # validation-set threshold selection (no test tuning)
│   ├── model_store.py          # portable load-or-train-at-startup
│   ├── decision_engine.py      # evidence/decision rules (pure, testable)
│   ├── predict.py              # SensorReading -> probability
│   └── reporting.py            # Gemini report + offline template
└── tests/
    ├── test_data_validation.py
    ├── test_decision_engine.py
    ├── test_model_store.py
    ├── test_reporting.py
    ├── test_predictions.py     # healthy/failure examples + input sensitivity
    ├── test_calibrate_threshold.py
    └── test_app.py
```

---

## 10. License / attribution

Dataset © UCI Machine Learning Repository, licensed **CC BY 4.0**. ForgeWise source
code is provided for the AI Builder Cup 2026 prototype.
