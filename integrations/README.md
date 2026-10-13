# ForgeWise equipment integration templates

These are **starting-point templates** for connecting physical manufacturing
equipment to ForgeWise. Every template validates incoming data with the shared,
unit-tested normaliser in `src/ingest.py` and appends valid readings to a single
queue file (`artifacts/ingest_queue.jsonl`, git-ignored).

They are examples, not production adapters: add authentication, TLS, retries, and
durable storage before any real deployment. ForgeWise is a decision-support
prototype and is **not** approved for operational machine control.

| # | Template | Transport | Optional dependency |
| --- | --- | --- | --- |
| 1 | `rest_ingest.py` | HTTP POST `/ingest` | none (stdlib) |
| 2 | `mqtt_subscriber.py` | MQTT subscribe | `paho-mqtt` |
| 3 | `modbus_poller.py` | Modbus/TCP poll (OPC-UA-style mapping) | `pymodbus` |
| 4 | `csv_scada_import.py` | CSV batch (SCADA/MES export) | none (pandas) |

## Payload schema

All transports accept the same JSON object. Keys are case-insensitive and several
aliases are accepted (e.g. `air_temperature`, `air temperature [K]`, `air_temp`).

```json
{
  "machine_id": "cnc-07",
  "type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 50
}
```

* `type` ∈ {`L`, `M`, `H`}; defaulted to `M` if omitted (with a warning).
* A **missing** sensor field is not an error — it becomes missing evidence, which
  the Missing Evidence Engine reports as `INCOMPLETE` / `REQUEST_DATA`.
* Non-numeric or non-finite values and unknown product types are **errors**.
* Unknown extra keys are reported but not rejected.

Print the exact schema from the project root:

```powershell
.\.venv\Scripts\python.exe -c "from src.ingest import payload_schema; print(payload_schema())"
```

## Consuming the queue

The queue is a newline-delimited JSON file. Each line records the normalised
`fields` plus any `errors` / `warnings`. You can:

* open it in the app's **Equipment Integration** page (paste a payload to validate
  it live), or
* feed `fields` into `src.orchestration.run_assessment()` yourself.

> The queue lives on the local filesystem and is **ephemeral** on a hosted
> deployment — point it at managed storage for anything durable.
