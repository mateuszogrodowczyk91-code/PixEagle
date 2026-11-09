"""Track lifecycle + rate limiting -> ``uas_visual_track`` payloads.

Input per frame: a list of :class:`Detection` (bbox, class, confidence,
track id). This is the same information PixEagle's ``NormalizedDetection``
carries (see ``src/classes/detection_adapter.py``); use :func:`from_normalized`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

EVENT_NAME = "uas_visual_track"


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    stable_id: bool = True


def from_normalized(dets: Iterable[Any], labels: Optional[Dict[int, str]] = None) -> List[Detection]:
    """Convert PixEagle ``NormalizedDetection`` objects."""
    labels = labels or {}
    out: List[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.aabb_xyxy)
        out.append(Detection(
            track_id=int(d.track_id),
            class_id=int(d.class_id),
            class_name=str(labels.get(int(d.class_id), d.class_id)),
            confidence=float(d.confidence),
            bbox_xyxy=(x1, y1, x2, y2),
            stable_id=bool(getattr(d, "track_id_is_stable", True)),
        ))
    return out


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def estimate_bearing(cx_norm: float, hfov_deg: Optional[float], heading_deg: Optional[float]) -> Optional[float]:
    """Azimuth of the target: camera heading + horizontal offset in the FOV."""
    if hfov_deg is None or heading_deg is None:
        return None
    return round((float(heading_deg) + (cx_norm - 0.5) * float(hfov_deg)) % 360.0, 2)


@dataclass
class _TrackState:
    first_ts: float
    last_seen: float
    last_post: float = -1e18
    frames: int = 0
    max_conf: float = 0.0
    last: Optional[Dict[str, Any]] = None
    announced: bool = False


@dataclass
class TrackEventManager:
    device: str
    input_type: str
    video_label: str
    detector_name: str
    min_interval_s: float = 2.0
    post_start_end: bool = True
    lost_after_s: float = 1.5
    hfov_deg: Optional[float] = None
    heading_provider: Callable[[], Optional[float]] = lambda: None
    max_events_per_min: int = 60          # global cap across all tracks
    dropped: int = 0
    _window: List[float] = field(default_factory=list)
    _tracks: Dict[int, _TrackState] = field(default_factory=dict)

    def _payload(self, phase: str, det: Detection, st: _TrackState, frame_ts: float,
                 frame_wh: Tuple[int, int]) -> Dict[str, Any]:
        w, h = frame_wh
        x1, y1, x2, y2 = det.bbox_xyxy
        cx = ((x1 + x2) / 2.0) / max(w, 1)
        cy = ((y1 + y2) / 2.0) / max(h, 1)
        heading = self.heading_provider()
        p: Dict[str, Any] = {
            "source": "pixeagle",
            "event": EVENT_NAME,
            "device": self.device,
            "input": self.input_type,
            "video_source": self.video_label,
            "mode": "detection_only",
            "phase": phase,
            "track_id": det.track_id,
            "track_id_stable": det.stable_id,
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 3),
            "max_confidence": round(st.max_conf, 3),
            "bbox_xyxy": [int(v) for v in det.bbox_xyxy],
            "bbox_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            "center_norm": [round(cx, 4), round(cy, 4)],
            "frame_w": w,
            "frame_h": h,
            "frame_time": iso_utc(frame_ts),
            "track_started": iso_utc(st.first_ts),
            "track_frames": st.frames,
            "detector": self.detector_name,
        }
        bearing = estimate_bearing(cx, self.hfov_deg, heading)
        if bearing is not None:
            p["bearing_deg_est"] = bearing
            p["bearing_basis"] = {"hfov_deg": self.hfov_deg, "heading_deg": heading}
        return p

    def _allow(self, ts: float) -> bool:
        self._window = [t for t in self._window if ts - t < 60.0]
        if len(self._window) >= self.max_events_per_min:
            self.dropped += 1
            return False
        self._window.append(ts)
        return True

    def update(self, dets: List[Detection], frame_ts: float, frame_wh: Tuple[int, int]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for d in dets:
            st = self._tracks.get(d.track_id)
            is_new = st is None
            if is_new:
                st = _TrackState(first_ts=frame_ts, last_seen=frame_ts)
                self._tracks[d.track_id] = st
            st.last_seen = frame_ts
            st.frames += 1
            st.max_conf = max(st.max_conf, d.confidence)
            if is_new and self.post_start_end:
                phase: Optional[str] = "start"
            elif frame_ts - st.last_post >= self.min_interval_s:
                phase = "update"
            else:
                phase = None
            payload = self._payload(phase or "update", d, st, frame_ts, frame_wh)
            st.last = payload
            if phase and self._allow(frame_ts):
                st.last_post = frame_ts
                st.announced = True
                events.append(payload)
        for tid in [t for t, s in self._tracks.items() if frame_ts - s.last_seen > self.lost_after_s]:
            events.extend(self._end(tid))
        return events

    def _end(self, tid: int) -> List[Dict[str, Any]]:
        st = self._tracks.pop(tid)
        if not (self.post_start_end and st.last and st.announced):
            return []
        p = dict(st.last)
        p.update(phase="end", track_frames=st.frames, max_confidence=round(st.max_conf, 3),
                 track_ended=iso_utc(st.last_seen),
                 duration_s=round(st.last_seen - st.first_ts, 3))
        return [p]

    def flush(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tid in list(self._tracks):
            out.extend(self._end(tid))
        return out


def status_track_dict(det: Detection, frame_w: int, frame_h: int, *, ts: Optional[float] = None) -> Dict[str, Any]:
    """Compact track for PreviewServer /status (Centrum auto-follow needs geometry)."""
    w = max(1, int(frame_w))
    h = max(1, int(frame_h))
    x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    out: Dict[str, Any] = {
        "track_id": int(det.track_id),
        "class_id": int(det.class_id),
        "class_name": str(det.class_name),
        "confidence": round(float(det.confidence), 3),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
        "center_norm": [round(cx, 4), round(cy, 4)],
        "frame_w": w,
        "frame_h": h,
    }
    if ts is not None:
        out["updated"] = float(ts)
    return out

