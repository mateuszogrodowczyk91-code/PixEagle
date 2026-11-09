"""Local preview + status server for SkyGuard Centrum (127.0.0.1 only).

GET /stream.mjpg   annotated video (boxes, class, confidence, track id)
GET /snapshot.jpg  last annotated frame
GET /status        JSON: running, fps, input, detector, active tracks, counters
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

COLORS = {"drone": (0, 0, 255), "uav": (0, 0, 255), "helicopter": (0, 140, 255), "airplane": (255, 200, 0),
          "aircraft": (255, 200, 0), "bird": (0, 200, 0), "kite": (200, 200, 0)}


def annotate(frame: np.ndarray, dets: List[Any], max_w: int = 960) -> np.ndarray:
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        c = COLORS.get(str(d.class_name).lower(), (255, 229, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, "#%d %s %d%%" % (d.track_id, d.class_name, round(d.confidence * 100)),
                    (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return img


class PreviewServer:
    def __init__(self, port: int, host: str = "127.0.0.1", jpeg_quality: int = 70, max_fps: float = 10.0) -> None:
        self._jpeg: Optional[bytes] = None
        self._cond = threading.Condition()
        self._last_push = 0.0
        self.min_dt = 1.0 / max_fps
        self.quality = jpeg_quality
        self.status: Dict[str, Any] = {"running": True, "started": time.time()}
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _hdr(self, code, ctype, length=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_GET(self):
                if self.path.startswith("/status"):
                    b = json.dumps(srv.status, default=str).encode()
                    self._hdr(200, "application/json", len(b)); self.wfile.write(b); return
                if self.path.startswith("/snapshot.jpg"):
                    j = srv._jpeg
                    if not j:
                        self._hdr(503, "text/plain", 0); return
                    self._hdr(200, "image/jpeg", len(j)); self.wfile.write(j); return
                if self.path.startswith("/stream.mjpg"):
                    self._hdr(200, "multipart/x-mixed-replace; boundary=frame")
                    last = None
                    try:
                        while srv.status.get("running"):
                            with srv._cond:
                                srv._cond.wait(timeout=2.0)
                                j = srv._jpeg
                            if j is None or j is last:
                                continue
                            last = j
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self._hdr(404, "text/plain", 0)

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="pixeagle-preview").start()

    def push(self, frame: np.ndarray, dets: List[Any]) -> None:
        now = time.time()
        if now - self._last_push < self.min_dt or cv2 is None:
            return
        self._last_push = now
        ok, buf = cv2.imencode(".jpg", annotate(frame, dets), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()

    def close(self) -> None:
        self.status["running"] = False
        with self._cond:
            self._cond.notify_all()
        self.httpd.shutdown()
