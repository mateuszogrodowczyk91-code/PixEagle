"""Video inputs: PTZ camera on a tripod (RTSP), Pixel phone stream, CloudEdge
RTSP (optional), webcam, file, frames pushed over HTTP (ingest), synthetic.

Labels returned for payloads never contain credentials.
"""
from __future__ import annotations

import os
import queue
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def redact_url(url: str) -> str:
    """Drop user:password and secret-looking query values from a URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<url>"
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        parts = parts._replace(netloc=host)
    q = re.sub(r"(?i)(pass(word)?|pwd|token|key|secret)=[^&]*", r"\1=***", parts.query)
    return urlunsplit(parts._replace(query=q))


def cloudedge_rtsp_url(ce: Dict[str, Any]) -> Tuple[Optional[str], str]:
    ip = os.environ.get(ce.get("ip_env", "CLOUDEDGE_IP"), "").strip()
    user = os.environ.get(ce.get("user_env", "CLOUDEDGE_USER"), "").strip() or "admin"
    pwd = os.environ.get(ce.get("password_env", "CLOUDEDGE_PASSWORD"), "")
    ch = "101" if str(ce.get("stream", "main")) == "main" else "102"
    port = int(ce.get("port", 8554))
    label = f"cloudedge rtsp://{ip or '<CLOUDEDGE_IP>'}:{port}/Streaming/Channels/{ch}"
    if not ip or not pwd:
        return None, label
    return f"rtsp://{quote(user, safe='')}:{quote(pwd, safe='')}@{ip}:{port}/Streaming/Channels/{ch}", label


class FrameSource:
    label = "source"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class CvSource(FrameSource):
    def __init__(self, target: Any, label: str, loop: bool = False) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required")
        self.target, self.label, self.loop = target, label, loop
        if isinstance(target, str) and target.lower().startswith("rtsp://"):
            # Force FFMPEG + TCP. Without CAP_FFMPEG, Windows builds may try CAP_IMAGES and leak the URL into errors.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            self.cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(target)
        elif isinstance(target, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(target, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(target)
        else:
            self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video input: {redact_url(label) if isinstance(label, str) else label}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop and isinstance(self.target, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


class SyntheticSource(FrameSource):
    """Sky-like gradient with a small dark object crossing (tests, no camera)."""

    label = "synthetic"

    def __init__(self, w: int = 640, h: int = 360, frames: int = 120, gap: Tuple[int, int] = (60, 80)) -> None:
        self.w, self.h, self.frames, self.gap, self.i = w, h, frames, gap, 0
        sky = np.linspace(200, 150, h).astype(np.uint8)[:, None]
        g = np.repeat(sky, w, 1)
        self.bg = np.dstack([np.clip(g.astype(int) + 40, 0, 255).astype(np.uint8), g, g])

    def read(self):
        if self.i >= self.frames:
            return False, None
        f = self.bg.copy()
        if not (self.gap[0] <= self.i < self.gap[1]):
            x = int(20 + self.i * (self.w - 60) / self.frames)
            y = int(self.h * 0.35 + 20 * np.sin(self.i / 9.0))
            f[y:y + 6, x:x + 14] = 40
        self.i += 1
        return True, f


class IngestSource(FrameSource):
    """HTTP endpoint for JPEG frames pushed by a phone or another process.

    POST /frame  body=JPEG  header x-pixeagle-ingest: <secret>
    Optional header x-heading-deg (camera azimuth, degrees from north).
    """

    def __init__(self, host: str, port: int, secret: Optional[str]) -> None:
        if not secret:
            raise RuntimeError("ingest needs PIXEAGLE_INGEST_SECRET")
        self.label = f"ingest http://{host}:{port}/frame"
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
        self.last_heading: Optional[float] = None
        src = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                if self.path != "/frame" or self.headers.get("x-pixeagle-ingest") != secret:
                    self.send_response(403); self.end_headers(); return
                n = int(self.headers.get("content-length", "0"))
                if n <= 0 or n > 8_000_000:
                    self.send_response(413); self.end_headers(); return
                img = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.send_response(400); self.end_headers(); return
                hd = self.headers.get("x-heading-deg")
                if hd:
                    try:
                        src.last_heading = float(hd)
                    except ValueError:
                        pass
                try:
                    src.q.put_nowait(img)
                except queue.Full:
                    pass
                self.send_response(204); self.end_headers()

        self.httpd = ThreadingHTTPServer((host, port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def read(self):
        try:
            return True, self.q.get(timeout=30)
        except queue.Empty:
            return False, None

    def release(self):
        self.httpd.shutdown()


def open_source(cfg: Dict[str, Any]) -> FrameSource:
    inp = cfg["input"]
    t = str(inp["type"])
    if t == "synthetic":
        return SyntheticSource()
    if t in ("pixel", "ptz"):
        env = "PIXEAGLE_PIXEL_URL" if t == "pixel" else "PIXEAGLE_PTZ_URL"
        url = str(inp.get("url") or os.environ.get(env, ""))
        if not url:
            raise RuntimeError(f"input.url (or {env}) is required for {t}: rtsp://... or http://.../video")
        return CvSource(url, inp.get("label") or f"{t} {redact_url(url)}")
    if t == "cloudedge":
        ce = dict(inp.get("cloudedge") or {})
        last_err = None
        preferred = str(ce.get("stream") or "sub")  # sub=102 for detection FPS; main=101 full-res
        order = [preferred] + [s for s in ("sub", "main") if s != preferred]
        for stream in order:
            ce["stream"] = stream
            url, label = cloudedge_rtsp_url(ce)
            if url is None:
                raise RuntimeError("CloudEdge needs CLOUDEDGE_IP and CLOUDEDGE_PASSWORD in the environment")
            try:
                return CvSource(url, inp.get("label") or label)
            except RuntimeError as e:
                last_err = e
        # USB webcam fallback when RTSP will not open
        if bool(inp.get("webcam_fallback", True)):
            idx = int(inp.get("webcam_index", 0))
            try:
                return CvSource(idx, inp.get("label") or f"webcam {idx} (cloudedge fallback)")
            except RuntimeError as e:
                last_err = e
        raise last_err or RuntimeError("CloudEdge RTSP open failed")
    if t == "webcam":
        idx = int(inp.get("webcam_index", 0))
        return CvSource(idx, inp.get("label") or f"webcam {idx}")
    if t == "file":
        path = str(inp["file"])
        return CvSource(path, inp.get("label") or f"file {os.path.basename(path)}", loop=bool(inp.get("loop")))
    if t == "ingest":
        ing = inp["ingest"]
        return IngestSource(ing["host"], int(ing["port"]), os.environ.get(ing.get("secret_env", "PIXEAGLE_INGEST_SECRET")))
    raise RuntimeError(f"unknown input.type {t!r}")


class OnvifPanReader:
    """READ-ONLY ONVIF PTZ position (GetStatus) -> camera heading in degrees.

    Sends only GetStatus. It never sends ContinuousMove / AbsoluteMove /
    RelativeMove: manual PTZ control stays in the camera's own app/joystick,
    and there is no automatic slew-to-cue or target following.
    """

    def __init__(self, oc: Dict[str, Any]) -> None:
        self.oc = oc
        self.heading: Optional[float] = None
        self._stop = threading.Event()
        self.user = os.environ.get(oc.get("user_env", "PTZ_USER"), "")
        self.pwd = os.environ.get(oc.get("password_env", "PTZ_PASSWORD"), "")

    def envelope(self) -> bytes:
        import base64, datetime as dt, hashlib
        from xml.sax.saxutils import escape
        sec = ""
        if self.user:
            nonce = os.urandom(16)
            created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + self.pwd.encode()).digest()).decode()
            sec = (
                '<s:Header><Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
                f"<UsernameToken><Username>{escape(self.user)}</Username>"
                '<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
                f"{digest}</Password>"
                '<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-soap-message-security-1.0#Base64Binary">'
                f"{base64.b64encode(nonce).decode()}</Nonce>"
                f'<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>'
                "</UsernameToken></Security></s:Header>"
            )
        body = (
            '<s:Body><GetStatus xmlns="http://www.onvif.org/ver20/ptz/wsdl">'
            f'<ProfileToken>{escape(str(self.oc.get("profile_token", "")))}</ProfileToken></GetStatus></s:Body>'
        )
        return f'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">{sec}{body}</s:Envelope>'.encode()

    @staticmethod
    def parse_pan(xml: str) -> Optional[float]:
        m = re.search(r"PanTilt\b[^>]*\bx=\"(-?[0-9.]+)\"", xml)
        return float(m.group(1)) if m else None

    def pan_to_heading(self, pan_norm: float) -> float:
        lo, hi = float(self.oc.get("pan_min_deg", -180)), float(self.oc.get("pan_max_deg", 180))
        deg = lo + (pan_norm + 1.0) / 2.0 * (hi - lo)
        return round((float(self.oc.get("pan_zero_heading_deg", 0.0)) + deg) % 360.0, 2)

    def poll_once(self) -> Optional[float]:
        import urllib.request
        url = f"http://{self.oc['host']}:{int(self.oc.get('port', 80))}/onvif/ptz_service"
        req = urllib.request.Request(url, data=self.envelope(), method="POST",
                                     headers={"content-type": "application/soap+xml; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=3) as r:
            pan = self.parse_pan(r.read().decode("utf-8", "replace"))
        if pan is not None:
            self.heading = self.pan_to_heading(pan)
        return self.heading

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.poll_once()
                except Exception:
                    self.heading = None
                self._stop.wait(float(self.oc.get("poll_s", 1.0)))
        threading.Thread(target=loop, name="onvif-pan", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
