"""Template 3/4 — Modbus/TCP (and OPC-UA-style) poller for PLCs.

Many machines expose process values as Modbus holding registers (OPC-UA is also
common; the mapping step is identical — read tags, scale them, forward them). This
template reads a documented register map, converts the raw integers into ForgeWise
sensor fields, validates them with :mod:`src.ingest`, and appends them to
``artifacts/ingest_queue.jsonl``.

Dependency (optional):

    .venv\\Scripts\\python.exe -m pip install pymodbus

Run (from the project root):

    .venv\\Scripts\\python.exe integrations\\modbus_poller.py \\
        --host 192.168.1.50 --unit 1 --machine-id cnc-07 --interval 5

The register map below is an EXAMPLE. Replace the addresses and scaling with the
real map from your equipment's documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import normalize_payload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "artifacts" / "ingest_queue.jsonl"

INSTALL_HINT = (
    "Modbus support requires the optional dependency 'pymodbus'. Install it with:\n"
    "    .venv\\Scripts\\python.exe -m pip install pymodbus"
)

COUNT = 6  # number of holding registers to read

# field -> (register_index, scale, offset)
REGISTER_MAP = {
    "air_temperature": (0, 0.1, 0.0),
    "process_temperature": (1, 0.1, 0.0),
    "rotational_speed": (2, 1.0, 0.0),
    "torque": (3, 0.1, 0.0),
    "tool_wear": (4, 1.0, 0.0),
}

TYPE_CODE = {0: "L", 1: "M", 2: "H"}
TYPE_REGISTER = 5


def registers_to_payload(registers: dict[int, int], machine_id: str) -> dict:
    """Convert raw register values into an ingest payload (pure, testable)."""
    payload: dict = {"machine_id": machine_id}
    for field, (index, scale, offset) in REGISTER_MAP.items():
        if index in registers:
            payload[field] = registers[index] * scale + offset
    if TYPE_REGISTER in registers:
        payload["type"] = TYPE_CODE.get(int(registers[TYPE_REGISTER]), "M")
    return payload


def append_reading(payload: dict, source: str = "modbus", queue_path: Path = QUEUE_PATH) -> dict:
    result = normalize_payload(payload)
    record = {
        "source": source,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "fields": result.fields,
        "errors": result.errors,
        "warnings": result.warnings,
        "unknown_keys": result.unknown_keys,
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    try:
        from pymodbus.client import ModbusTcpClient  # type: ignore
    except ImportError:
        print(INSTALL_HINT)
        return

    parser = argparse.ArgumentParser(description="ForgeWise Modbus/TCP polling template")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=502)
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--machine-id", default="plc-01")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    client = ModbusTcpClient(args.host, port=args.port)
    if not client.connect():
        print(f"Could not connect to Modbus server at {args.host}:{args.port}")
        return

    print(f"Polling unit {args.unit} every {args.interval}s (Ctrl+C to stop)")
    try:
        while True:
            response = client.read_holding_registers(0, count=COUNT, slave=args.unit)
            if response.isError():
                print(f"Read error: {response}")
            else:
                registers = {i: response.registers[i] for i in range(COUNT)}
                payload = registers_to_payload(registers, args.machine_id)
                record = append_reading(payload, source=f"modbus:{args.machine_id}")
                status = "ok" if not record["errors"] else "errors"
                print(f"{payload} -> {status}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        client.close()


if __name__ == "__main__":
    main()
