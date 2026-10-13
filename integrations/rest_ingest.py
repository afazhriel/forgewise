"""Template 1/4 — REST / HTTP ingestion for physical manufacturing equipment.

A minimal, dependency-free HTTP receiver. A gateway, edge device, or SCADA system
POSTs a JSON reading to ``/ingest``; the payload is validated with the shared
:mod:`src.ingest` normaliser and appended to ``artifacts/ingest_queue.jsonl``.

Run (from the project root):

    .venv\\Scripts\\python.exe integrations\\rest_ingest.py --port 8080

Test with curl / Invoke-RestMethod:

    curl -X POST http://localhost:8080/ingest \\
         -H "Content-Type: application/json" \\
         -d '{"machine_id":"cnc-07","type":"M","air_temperature":298.1,
              "process_temperature":308.6,"rotational_speed":1500,
              "torque":40.0,"tool_wear":50}'

This template only *queues validated readings*. It does not run the model or make
maintenance decisions, and it is an example — add authentication, TLS, and durable
storage before any real deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import normalize_payload  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "artifacts" / "ingest_queue.jsonl"


def append_reading(payload: dict, source: str = "rest", queue_path: Path = QUEUE_PATH) -> dict:
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


def make_handler(queue_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path.rstrip("/") != "/ingest":
                self._send(404, {"ok": False, "error": "POST to /ingest"})
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                self._send(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
                return
            record = append_reading(payload, source="rest", queue_path=queue_path)
            ok = not record["errors"]
            self._send(200 if ok else 422, {"ok": ok, **{k: record[k] for k in ("errors", "warnings", "fields")}})

        def log_message(self, *args) -> None:  # silence default logging
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeWise REST ingestion template")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), make_handler(QUEUE_PATH))
    print(f"REST ingestion listening on http://{args.host}:{args.port}/ingest")
    print(f"Queue: {QUEUE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
