"""SkyGuard Evidence API sink (POST /api/v1/evidence, header x-skyguard-token).

Posts happen on a background thread so the video loop never blocks on the
network. The token is read from the environment and is never logged.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("skyguard_bridge.sink")


class EvidenceSink:
    def __init__(
        self,
        base_url: str,
        token: Optional[str],
        *,
        enabled: bool = False,
        timeout_s: float = 8.0,
        queue_size: int = 200,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = (token or "").strip()
        self.enabled = bool(enabled)
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=queue_size)
        self.sent: List[Dict[str, Any]] = []      # server replies (no token inside)
        self.errors: List[str] = []
        self.dropped = 0
        self._thread: Optional[threading.Thread] = None
        if self.enabled and not self._token:
            log.warning("SkyGuard sink enabled but token is missing; posts disabled")
            self.enabled = False
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="skyguard-sink", daemon=True)
            self._thread.start()

    @classmethod
    def from_config(cls, sg_cfg: Dict[str, Any], **kw: Any) -> "EvidenceSink":
        return cls(
            sg_cfg["base_url"],
            os.environ.get(sg_cfg.get("token_env", "SKYGUARD_TOKEN_PIXEAGLE")),
            enabled=sg_cfg.get("enabled", False),
            timeout_s=float(sg_cfg.get("timeout_s", 8.0)),
            queue_size=int(sg_cfg.get("queue_size", 200)),
            **kw,
        )

    def submit(self, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            self._q.put_nowait(payload)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def post_now(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous POST; returns {'status': int, 'body': dict|str}."""
        req = urllib.request.Request(
            self.base_url + "/api/v1/evidence",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={"content-type": "application/json", "x-skyguard-token": self._token},
        )
        try:
            with self._opener(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", "replace")
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            status = e.code
        try:
            body: Any = json.loads(raw)
        except ValueError:
            body = raw[:500]
        return {"status": status, "body": body}

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            try:
                res = self.post_now(item)
                if res["status"] == 201:
                    self.sent.append(res["body"])
                else:
                    self.errors.append(f"HTTP {res['status']}")
                    log.warning("SkyGuard POST failed: HTTP %s", res["status"])
            except Exception as exc:  # network errors must not kill the sensor
                self.errors.append(type(exc).__name__)
                log.warning("SkyGuard POST error: %s", type(exc).__name__)
            finally:
                self._q.task_done()

    def close(self, wait_s: float = 15.0) -> None:
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout=wait_s)
        self._thread = None
