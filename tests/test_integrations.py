"""Tests for the 4 equipment-integration templates.

Templates are loaded by file path (they are scripts, not a package). Each template
defines pure/conversion helpers at module level and only touches the network inside
``main()``/guarded blocks, so importing them is safe and side-effect free.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = ROOT / "integrations"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, INTEGRATIONS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- template 1: REST -------------------------------------------------------
def test_rest_template_appends_validated_reading(tmp_path):
    rest = _load("fw_rest_ingest", "rest_ingest.py")
    queue = tmp_path / "queue.jsonl"
    record = rest.append_reading(
        {"machine_id": "cnc-07", "type": "M", "air_temperature": 298.1, "torque": 40},
        queue_path=queue,
    )
    assert record["errors"] == []
    assert record["fields"]["machine_id"] == "cnc-07"
    assert queue.exists()
    assert len(queue.read_text(encoding="utf-8").strip().splitlines()) == 1


# --- template 2: MQTT -------------------------------------------------------
def test_mqtt_template_handles_valid_and_invalid_messages(tmp_path):
    mqtt = _load("fw_mqtt", "mqtt_subscriber.py")
    queue = tmp_path / "queue.jsonl"

    good = mqtt.handle_message("factory/line1/reading", b'{"torque": 40, "rpm": 1500}', queue)
    assert good["errors"] == []
    assert good["fields"]["rotational_speed"] == pytest.approx(1500.0)

    bad = mqtt.handle_message("factory/line1/reading", b"{not json", queue)
    assert bad["errors"]
    assert len(queue.read_text(encoding="utf-8").strip().splitlines()) == 2


# --- template 3: Modbus -----------------------------------------------------
def test_modbus_register_scaling():
    modbus = _load("fw_modbus", "modbus_poller.py")
    registers = {0: 2981, 1: 3086, 2: 1500, 3: 400, 4: 50, 5: 1}
    payload = modbus.registers_to_payload(registers, "cnc-07")
    assert payload["air_temperature"] == pytest.approx(298.1)
    assert payload["process_temperature"] == pytest.approx(308.6)
    assert payload["rotational_speed"] == pytest.approx(1500.0)
    assert payload["torque"] == pytest.approx(40.0)
    assert payload["tool_wear"] == pytest.approx(50.0)
    assert payload["type"] == "M"


# --- template 4: CSV / SCADA ------------------------------------------------
def test_csv_template_reads_aliased_headers(tmp_path):
    csv = _load("fw_csv", "csv_scada_import.py")
    path = tmp_path / "scada.csv"
    pd.DataFrame(
        {
            "Air temperature [K]": [298.1, 299.0],
            "Torque [Nm]": [40.0, 55.0],
            "rpm": [1500, 1400],
            "wear": [50, 210],
        }
    ).to_csv(path, index=False)

    payloads = csv.read_csv_to_payloads(path, machine_id="line1-motor3")
    assert len(payloads) == 2
    assert payloads[0]["machine_id"] == "line1-motor3"

    records = csv.import_csv(path, machine_id="line1-motor3", queue_path=tmp_path / "q.jsonl")
    assert len(records) == 2
    assert all(record["errors"] == [] for record in records)
    assert records[0]["fields"]["air_temperature"] == pytest.approx(298.1)
