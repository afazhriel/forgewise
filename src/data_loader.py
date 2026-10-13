"""Dataset acquisition, audit, and validation for ForgeWise.

Dataset: UCI AI4I 2020 Predictive Maintenance Dataset
Source : https://archive.ics.uci.edu/dataset/601/ai4i+2020+
License: CC BY 4.0

The raw dataset is *downloaded* from the official UCI archive and is not
committed to the repository (see ``.gitignore``). This module is the single
place that knows the expected schema, so feature selection is never guessed.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants describing the *verified* dataset schema (inspected from the file).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+"
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
LICENSE = "CC BY 4.0"
ZIP_PATH = DATA_DIR / "ai4i2020.zip"
CSV_PATH = DATA_DIR / "ai4i2020.csv"

TARGET = "Machine failure"

# Columns used as model inputs (nominal feature set).
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c != "Type"]

# Identifier columns and failure-mode flags: excluded from features to avoid
# target leakage (the failure-mode flags directly encode the label).
ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
EXCLUDED_COLUMNS = ID_COLUMNS + FAILURE_MODE_COLUMNS

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

VALID_TYPES = {"L", "M", "H"}


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def download_dataset(force: bool = False) -> Path:
    """Download and extract the AI4I dataset from the official UCI source.

    Returns the path to the extracted CSV. If the CSV already exists and
    ``force`` is False, the download is skipped.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists() and not force:
        return CSV_PATH

    if force or not ZIP_PATH.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            payload = response.read()
        ZIP_PATH.write_bytes(payload)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV found in archive {ZIP_PATH}")
        with archive.open(members[0]) as src, open(CSV_PATH, "wb") as dst:
            dst.write(src.read())

    return CSV_PATH


def load_dataset(path: str | Path | None = None, ensure: bool = True) -> pd.DataFrame:
    """Load the dataset into a DataFrame, downloading it if necessary."""
    csv_path = Path(path) if path is not None else CSV_PATH
    if not csv_path.exists():
        if not ensure:
            raise FileNotFoundError(csv_path)
        download_dataset()
    return pd.read_csv(csv_path)


def dataset_sha256(path: str | Path | None = None) -> str:
    """Return the SHA-256 checksum of the raw CSV (for reproducibility)."""
    csv_path = Path(path) if path is not None else CSV_PATH
    digest = hashlib.sha256()
    with open(csv_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Validation & audit
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable schema problems (empty means valid)."""
    problems: list[str] = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing expected columns: {missing}")

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        problems.append(f"Unexpected extra columns: {extra}")

    if TARGET not in df.columns:
        problems.append(f"Target column '{TARGET}' is absent.")
    elif not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        problems.append(f"Target '{TARGET}' must be binary (0/1).")

    if "Type" in df.columns:
        bad_types = set(df["Type"].dropna().unique()) - VALID_TYPES
        if bad_types:
            problems.append(f"Unexpected Type values: {sorted(bad_types)}")

    return problems


def audit_dataset(df: pd.DataFrame) -> dict:
    """Compute an honest, non-fabricated audit of the loaded dataset."""
    audit: dict = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {
            c: int(n) for c, n in df.isna().sum().items() if n > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if TARGET in df.columns:
        counts = df[TARGET].value_counts(dropna=False).to_dict()
        audit["target"] = {
            "name": TARGET,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "positive_rate": float(df[TARGET].mean()),
        }

    numeric = df.select_dtypes(include="number")
    audit["numeric_ranges"] = {
        c: {
            "min": float(numeric[c].min()),
            "max": float(numeric[c].max()),
            "mean": round(float(numeric[c].mean()), 4),
        }
        for c in numeric.columns
    }

    suspicion: list[str] = []
    if TARGET in df.columns and len(df[df[TARGET] == 1]) == 0:
        suspicion.append("No positive target cases present.")
    if audit["duplicate_rows"] > 0:
        suspicion.append(f"{audit['duplicate_rows']} exact duplicate rows.")
    if "Type" in df.columns:
        unseen = set(df["Type"].dropna().unique()) - VALID_TYPES
        if unseen:
            suspicion.append(f"Unexpected Type codes: {sorted(unseen)}")
    audit["suspicious_values"] = suspicion

    return audit


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using the verified feature set and target column."""
    problems = validate_schema(df)
    if problems:
        raise ValueError("Dataset failed schema validation: " + "; ".join(problems))
    return df[FEATURE_COLUMNS].copy(), df[TARGET].astype(int).copy()


def _main() -> None:
    path = download_dataset()
    df = load_dataset(path)
    audit = audit_dataset(df)
    audit["source_page"] = SOURCE_PAGE
    audit["source_url"] = SOURCE_URL
    audit["license"] = LICENSE
    audit["sha256"] = dataset_sha256(path)
    audit["schema_problems"] = validate_schema(df)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    _main()
