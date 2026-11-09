"""Local mock of the SkyGuard Evidence API (for offline tests).

python -m skyguard_bridge.mock_api --port 8787 --token-env SKYGUARD_MOCK_TOKEN
Mirrors the live contract: x-skyguard-token, 201 {ts, prev_hash, payload, hash},
sha256(JSON({payload, prev_hash, ts})) with sorted top-level keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class MockEvidenceApi:
    def __init__(self, token: str, device: str = "pixeagle", port: int = 0) -> None:
        self.token, self.device = token, device
        self.rows: List[Dict[str, Any]] = []
        api = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, obj: Any) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path.startswith("/health"):
                    return self._send(200, {"ok": True})
                if self.headers.get("x-skyguard-token") != api.token:
                    return self._send(401, {"error": "unauthorized"})
                return self._send(200, list(reversed(api.rows)))

            def do_POST(self):
                if self.path != "/api/v1/evidence":
                    return self._send(404, {"error": "not found"})
                if self.headers.get("x-skyguard-token") != api.token:
                    return self._send(401, {"error": "unauthorized"})
                n = int(self.headers.get("content-length", "0"))
                try:
                    payload = json.loads(self.rfile.read(n))
                    assert isinstance(payload, dict)
                except Exception:
                    return self._send(400, {"error": "bad json"})
                payload = dict(payload, _device=api.device)
                prev = api.rows[-1]["hash"] if api.rows else "0" * 64
                ts = datetime.now(timezone.utc).isoformat()
                body = {"payload": payload, "prev_hash": prev, "ts": ts}
                h = hashlib.sha256(json.dumps(body, sort_keys=False, separators=(",", ":")).encode()).hexdigest()
                row = {"ts": ts, "prev_hash": prev, "payload": payload, "hash": h}
                api.rows.append(row)
                return self._send(201, row)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    def start(self) -> "MockEvidenceApi":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()

    def chain_ok(self) -> bool:
        prev = "0" * 64
        for r in self.rows:
            if r["prev_hash"] != prev:
                return False
            prev = r["hash"]
        return True


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--token-env", default="SKYGUARD_MOCK_TOKEN")
    a = ap.parse_args(argv)
    tok = os.environ.get(a.token_env) or "mock-token"
    api = MockEvidenceApi(tok, port=a.port)
    print(f"mock SkyGuard API on {api.base_url}")
    api.httpd.serve_forever()


if __name__ == "__main__":
    main()
