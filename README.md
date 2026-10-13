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

Requires **Python 3.12.x** (tested on 3.12.10) on Windows.

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
* **Model version (this build):** `fw-20261013T074119Z-dc6630cd`.

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
inconclusive band `[0.30, 0.70]`, stale after `60` minutes.

Every assessment is appended to `artifacts/decision_log.jsonl` with the full input,
the decision, the model version, and a UTC timestamp.

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

## 8. Project layout

```text
forgewise/
├── app.py                      # Streamlit UI
├── conftest.py                 # makes `src` importable under pytest
├── requirements.txt
├── README.md
├── .gitignore
├── data/                       # raw dataset lives here (git-ignored)
├── models/                     # trained model + metadata (git-ignored)
├── artifacts/                  # metrics, figures, decision log (git-ignored)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # download, schema validation, audit
│   ├── train_model.py          # reproducible baseline
│   ├── evaluate_model.py       # held-out evaluation + figures
│   ├── decision_engine.py      # evidence/decision rules (pure, testable)
│   ├── predict.py              # SensorReading -> probability
│   └── reporting.py            # Gemini report + offline template
└── tests/
    ├── test_data_validation.py
    └── test_decision_engine.py
```

---

## 9. License / attribution

Dataset © UCI Machine Learning Repository, licensed **CC BY 4.0**. ForgeWise source
code is provided for the AI Builder Cup 2026 prototype.
