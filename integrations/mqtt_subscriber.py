"""Template 2/4 — MQTT subscriber for physical manufacturing equipment.

Industrial gateways commonly publish sensor readings over MQTT. This template
subscribes to a topic, validates each JSON payload with :mod:`src.ingest`, and
appends it to ``artifacts/ingest_queue.jsonl`` (the same queue the REST template
uses).

Dependency (optional, not required by the app or its tests):

    .venv\\Scripts\\python.exe -m pip install paho-mqtt

Run (from the project root):

    .venv\\Scripts\\python.exe integrations\\mqtt_subscriber.py \\
        --host 192.168.1.10 --topic factory/line1/+

Expected payload: the same JSON object documented in :func:`src.ingest.payload_schema`.

This is an example template — configure authentication/TLS for a real broker.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import normalize_payload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "artifacts" / "ingest_queue.jsonl"

INSTALL_HINT = (
    "MQTT support requires the optional dependency 'paho-mqtt'. Install it with:\n"
    "    .venv\\Scripts\\python.exe -m pip install paho-mqtt"
)


def append_reading(payload: dict, source: str = "mqtt", queue_path: Path = QUEUE_PATH) -> dict:
    """Validate and append one payload; return the stored record."""
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


def handle_message(topic: str, payload_bytes: bytes, queue_path: Path = QUEUE_PATH) -> dict:
    """Parse one MQTT message and store it. Safe to call from tests."""
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record = {
            "source": f"mqtt:{topic}",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "fields": {},
            "errors": [f"Invalid JSON: {exc}"],
            "warnings": [],
            "unknown_keys": [],
        }
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
    return append_reading(payload, source=f"mqtt:{topic}", queue_path=queue_path)


def main() -> None:
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except ImportError:
        print(INSTALL_HINT)
        return

    parser = argparse.ArgumentParser(description="ForgeWise MQTT ingestion template")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="factory/+/reading")
    args = parser.parse_args()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"Connected ({reason_code}); subscribing to {args.topic}")
        client.subscribe(args.topic)

    def on_message(client, userdata, message):
        record = handle_message(message.topic, message.payload, QUEUE_PATH)
        status = "ok" if not record.get("errors") else "errors"
        print(f"{message.topic}: {status}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"Connecting to mqtt://{args.host}:{args.port} ...")
    client.connect(args.host, args.port, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
