"""Template 4/4 — CSV / SCADA batch import.

The lowest-friction path: point this at a CSV exported by a SCADA historian, MES,
or data-logger. Column headers are matched to ForgeWise sensor fields via the shared
aliases in :mod:`src.ingest` (so ``Air temperature [K]`` or ``air_temp`` both work),
and each row is validated and appended to ``artifacts/ingest_queue.jsonl``.

No optional dependencies — uses pandas (already required).

Run (from the project root):

    .venv\\Scripts\\python.exe integrations\\csv_scada_import.py \\
        --csv path\\to\\scada_export.csv --machine-id line1-motor3

Expected columns (case-insensitive, aliases accepted): see
``python -c "from src.ingest import payload_schema; print(payload_schema())"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import normalize_payload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "artifacts" / "ingest_queue.jsonl"


def read_csv_to_payloads(path: str | Path, machine_id: str | None = None) -> list[dict]:
    """Read a CSV and return one payload dict per row (pure, testable)."""
    frame = pd.read_csv(path)
    payloads: list[dict] = []
    for _, row in frame.iterrows():
        payload = {str(column): row[column] for column in frame.columns}
        if machine_id and not any(
            str(column).strip().lower() in {"machine_id", "machine id", "asset", "asset_id"}
            for column in frame.columns
        ):
            payload["machine_id"] = machine_id
        payloads.append(payload)
    return payloads


def import_csv(
    path: str | Path,
    machine_id: str | None = None,
    queue_path: Path = QUEUE_PATH,
) -> list[dict]:
    """Validate and append every row of a CSV; return the stored records."""
    records: list[dict] = []
    for payload in read_csv_to_payloads(path, machine_id=machine_id):
        result = normalize_payload(payload)
        record = {
            "source": "csv",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "fields": result.fields,
            "errors": result.errors,
            "warnings": result.warnings,
            "unknown_keys": result.unknown_keys,
        }
        records.append(record)
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeWise CSV/SCADA import template")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--machine-id", default=None)
    args = parser.parse_args()

    records = import_csv(args.csv, machine_id=args.machine_id)
    valid = sum(1 for record in records if not record["errors"])
    print(f"Imported {len(records)} row(s): {valid} valid, {len(records) - valid} with errors.")
    print(f"Queue: {QUEUE_PATH}")


if __name__ == "__main__":
    main()
