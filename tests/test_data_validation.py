"""Tests for dataset schema validation and auditing."""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_loader


def _valid_frame(n: int = 4) -> pd.DataFrame:
    records = []
    for i in range(n):
        records.append(
            {
                "UDI": i + 1,
                "Product ID": f"M{1000 + i}",
                "Type": ["L", "M", "H", "M"][i % 4],
                "Air temperature [K]": 298.1 + i * 0.1,
                "Process temperature [K]": 308.6 + i * 0.1,
                "Rotational speed [rpm]": 1500 + i,
                "Torque [Nm]": 40.0 + i,
                "Tool wear [min]": 10 + i,
                "Machine failure": i % 2,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        )
    return pd.DataFrame(records, columns=data_loader.EXPECTED_COLUMNS)


def test_valid_frame_passes_schema():
    assert data_loader.validate_schema(_valid_frame()) == []


def test_missing_column_detected():
    df = _valid_frame().drop(columns=["Torque [Nm]"])
    problems = data_loader.validate_schema(df)
    assert any("Missing expected columns" in p for p in problems)


def test_extra_column_detected():
    df = _valid_frame()
    df["surprise"] = 1
    problems = data_loader.validate_schema(df)
    assert any("extra columns" in p.lower() for p in problems)


def test_non_binary_target_detected():
    df = _valid_frame()
    df.loc[0, "Machine failure"] = 7
    problems = data_loader.validate_schema(df)
    assert any("binary" in p for p in problems)


def test_invalid_type_detected():
    df = _valid_frame()
    df.loc[0, "Type"] = "X"
    problems = data_loader.validate_schema(df)
    assert any("Type values" in p for p in problems)


def test_audit_reports_shape_and_target():
    df = _valid_frame(n=4)
    audit = data_loader.audit_dataset(df)
    assert audit["row_count"] == 4
    assert audit["column_count"] == len(data_loader.EXPECTED_COLUMNS)
    assert audit["missing_values_total"] == 0
    assert audit["duplicate_rows"] == 0
    assert audit["target"]["name"] == "Machine failure"
    assert sum(audit["target"]["counts"].values()) == 4


def test_audit_flags_duplicates():
    df = pd.concat([_valid_frame(n=2), _valid_frame(n=2)], ignore_index=True)
    audit = data_loader.audit_dataset(df)
    assert audit["duplicate_rows"] == 2
    assert any("duplicate" in s.lower() for s in audit["suspicious_values"])


def test_split_features_target_excludes_leakage_columns():
    df = _valid_frame(n=6)
    X, y = data_loader.split_features_target(df)
    assert list(X.columns) == data_loader.FEATURE_COLUMNS
    for excluded in data_loader.EXCLUDED_COLUMNS:
        assert excluded not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y) == 6


def test_split_rejects_invalid_frame():
    df = _valid_frame().drop(columns=["Type"])
    with pytest.raises(ValueError):
        data_loader.split_features_target(df)


@pytest.mark.skipif(
    not data_loader.CSV_PATH.exists(), reason="Raw dataset not downloaded"
)
def test_real_dataset_is_valid_and_expected_size():
    df = data_loader.load_dataset()
    assert data_loader.validate_schema(df) == []
    assert df.shape == (10000, 14)
    assert df["Machine failure"].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0
