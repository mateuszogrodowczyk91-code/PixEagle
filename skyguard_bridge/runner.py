"""Headless detection-only loop: video -> detector -> track events -> SkyGuard."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from . import safety
from .config import device_name, load_config
from .detectors import build_detector
from .sink import EvidenceSink
from .sources import redact_log_line, OnvifPanReader, open_source
from .tracks import TrackEventManager

log = logging.getLogger("skyguard_bridge")


class _RedactFilter(logging.Filter):
    """Strip rtsp://user:pass@ (and http/https) from every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_log_line(str(record.msg))
            if record.args:
                record.args = tuple(
                    redact_log_line(a) if isinstance(a, str) else a for a in record.args
                )
        except Exception:
            pass
        return True


class _RedactStream:
    """Wrap a text stream so OpenCV/ffmpeg prints cannot leak userinfo."""

    def __init__(self, inner):
        self._inner = inner

    def write(self, data):
        if not data:
            return 0
        text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
        return self._inner.write(redact_log_line(text))

    def flush(self):
        return self._inner.flush()

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        return getattr(self._inner, "isatty", lambda: False)()


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cfg: Dict[str, Any], *, sink: Optional[EvidenceSink] = None,
        events_out: Optional[str] = None, preview_port: Optional[int] = None,
        stop_event: Optional[Any] = None) -> Dict[str, Any]:
    safety.assert_config_detection_only(cfg)
    baseline = frozenset(safety.loaded_forbidden_modules())
    os.chdir(REPO_ROOT)
    src = open_source(cfg)
    det = build_detector(cfg["detector"], REPO_ROOT)
    cam = cfg["camera"]
    pan = None
    if cam.get("ptz_onvif", {}).get("enabled"):
        pan = OnvifPanReader(cam["ptz_onvif"])
        pan.start()

    def heading() -> Optional[float]:
        if pan is not None and pan.heading is not None:
            return pan.heading
        h = getattr(src, "last_heading", None)
        return h if h is not None else cam.get("heading_deg")

    sg = cfg["skyguard"]
    mgr = TrackEventManager(
        device=device_name(cfg), input_type=str(cfg["input"]["type"]), video_label=src.label,
        detector_name=det.name, min_interval_s=float(sg["min_interval_s"]),
        post_start_end=bool(sg["post_start_end"]), lost_after_s=float(cfg["detector"]["lost_after_s"]),
        hfov_deg=cam.get("hfov_deg"), heading_provider=heading,
        max_events_per_min=int(sg.get("max_events_per_min", 60)),
    )
    sink = sink or EvidenceSink.from_config(sg)
    rt = cfg["runtime"]
    min_dt = 1.0 / float(rt["max_fps"]) if rt.get("max_fps") else 0.0
    max_frames = int(rt.get("max_frames") or 0)
    is_live = cfg["input"]["type"] not in ("file", "synthetic")
    fh = open(events_out, "a", encoding="utf-8", buffering=1) if events_out else None
    preview = None
    if preview_port is not None:
        from .preview import PreviewServer
        preview = PreviewServer(preview_port)
    stats: Dict[str, Any] = {"frames": 0, "detections": 0, "events": 0, "posted": 0, "device": mgr.device,
                             "detector": det.name, "input": mgr.input_type, "skyguard_enabled": sink.enabled}
    t_start = time.time()
    fps_t, fps_n = time.time(), 0
    try:
        while not (stop_event is not None and stop_event.is_set()):
            t0 = time.time()
            ok, frame = src.read()
            if not ok:
                break
            stats["frames"] += 1
            # file/synthetic: deterministic media clock (10 fps); live: wall clock
            ts = time.time() if is_live else t_start + stats["frames"] / 10.0
            dets = det(frame)
            stats["detections"] += len(dets)
            for e in mgr.update(dets, ts, (frame.shape[1], frame.shape[0])):
                _emit(e, sink, fh, stats)
            if preview is not None:
                preview.push(frame, dets)
                fps_n += 1
                if time.time() - fps_t >= 2.0:
                    stats["fps"] = round(fps_n / (time.time() - fps_t), 1)
                    fps_t, fps_n = time.time(), 0
                preview.status.update(stats, video_source=src.label, frame_w=frame.shape[1], frame_h=frame.shape[0],
                                      active_tracks=[{"track_id": d.track_id, "class_name": d.class_name,
                                                      "confidence": round(d.confidence, 3)} for d in dets],
                                      updated=time.time())
            if max_frames and stats["frames"] >= max_frames:
                break
            if is_live and min_dt:
                time.sleep(max(0.0, min_dt - (time.time() - t0)))
        for e in mgr.flush():
            _emit(e, sink, fh, stats)
    finally:
        if preview is not None:
            preview.close()
        src.release()
        if pan:
            pan.stop()
        sink.close()
        if fh:
            fh.close()
    safety.assert_runtime_detection_only(baseline)
    stats["rate_limited"] = mgr.dropped
    stats["sink_sent"] = len(sink.sent)
    stats["sink_errors"] = list(sink.errors)
    return stats


def _emit(e: Dict[str, Any], sink: EvidenceSink, fh, stats: Dict[str, Any]) -> None:
    stats["events"] += 1
    if fh:
        fh.write(json.dumps(e) + "\n")
    if sink.submit(e):
        stats["posted"] += 1
    log.info("%s track=%s %s conf=%.2f", e["phase"], e["track_id"], e["class_name"], e["confidence"])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m skyguard_bridge",
                                 description="PixEagle -> SkyGuard passive visual sensor (detection only)")
    ap.add_argument("-c", "--config", default=os.path.join(REPO_ROOT, "skyguard_bridge", "skyguard_sensor.yaml"))
    ap.add_argument("--input", choices=["ptz", "pixel", "cloudedge", "webcam", "file", "ingest", "synthetic"])
    ap.add_argument("--url", help="ptz/pixel stream URL (rtsp:// or http://)")
    ap.add_argument("--file")
    ap.add_argument("--webcam", type=int)
    ap.add_argument("--detector", choices=["auto", "onnx", "directml", "yolo", "motion"])
    ap.add_argument("--all-classes", action="store_true", help="do not filter to aerial classes")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--events-out", help="write events as JSON lines")
    ap.add_argument("--send", action="store_true", help="enable posting to SkyGuard for this run")
    ap.add_argument("--no-send", action="store_true", help="never post (e.g. when SkyGuard Centrum forwards the events)")
    ap.add_argument("--preview-port", type=int, help="serve annotated MJPEG + /status on 127.0.0.1:PORT")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    _rf = _RedactFilter()
    for h in logging.root.handlers:
        h.addFilter(_rf)
    log.addFilter(_rf)
    # OpenCV / ffmpeg write to the real stderr fd; wrap Python stderr for anything that uses it.
    if not isinstance(sys.stderr, _RedactStream):
        try:
            sys.stderr = _RedactStream(sys.stderr)  # type: ignore[assignment]
        except Exception:
            pass
    cfg = load_config(a.config if os.path.exists(a.config) else None)
    if a.input:
        cfg["input"]["type"] = a.input
    if a.url:
        cfg["input"]["url"] = a.url
    if a.file:
        cfg["input"]["file"] = os.path.abspath(a.file)
        if not a.input:
            cfg["input"]["type"] = "file"
    if a.webcam is not None:
        cfg["input"]["webcam_index"] = a.webcam
    if a.detector:
        cfg["detector"]["type"] = a.detector
    if a.all_classes:
        cfg["detector"]["classes"] = []
    if a.max_frames is not None:
        cfg["runtime"]["max_frames"] = a.max_frames
    if a.send:
        cfg["skyguard"]["enabled"] = True
    if a.no_send:
        cfg["skyguard"]["enabled"] = False
    stats = run(cfg, events_out=a.events_out, preview_port=a.preview_port)
    print(json.dumps(stats, indent=2))
    return 0 if not stats["sink_errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
