"""Detectors producing :class:`tracks.Detection` lists.

* ``yolo``   - PixEagle's own ``UltralyticsBackend.detect_and_track`` (YOLO +
               ByteTrack/BoT-SORT). Same code path SmartTracker uses
               (``SmartTracker.track_and_draw`` -> ``backend.detect_and_track``
               -> ``NormalizedDetection``), but without AppController,
               follower or MAVSDK.
* ``motion`` - OpenCV background subtraction + IoU tracker; needs only
               OpenCV, useful for a static tripod camera looking at the sky.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from .tracks import Detection, from_normalized

log = logging.getLogger("skyguard_bridge.detectors")

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class MotionDetector:
    name = "pixeagle-bridge/motion-mog2"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.min_area = int(cfg.get("motion_min_area_px", 40))
        self.max_frac = float(cfg.get("motion_max_area_frac", 0.2))
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.max_blobs = int(cfg.get("motion_max_blobs", 12))
        self.min_hits = int(cfg.get("motion_min_hits", 4))
        self.tracks: Dict[int, Tuple[Tuple[int, int, int, int], int]] = {}  # id -> (bbox, misses)
        self.hits: Dict[int, int] = {}
        self.next_id = 1
        self.frame_i = 0

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        self.frame_i += 1
        mask = self.bg.apply(frame)
        if self.frame_i < 5:  # background warm-up
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        boxes = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < self.min_area or area > self.max_frac * w * h:
                continue
            boxes.append((x, y, x + bw, y + bh, min(1.0, cv2.contourArea(c) / max(area, 1) * 0.5 + 0.3)))
        if len(boxes) > self.max_blobs:
            # Camera moving / exposure jump / rain: not a usable static-sky frame.
            return []
        out: List[Detection] = []
        used = set()
        for tid, (tb, misses) in list(self.tracks.items()):
            best, bi = 0.0, -1
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                s = _iou(tb, b[:4])
                if s == 0.0:  # small fast objects: centre-distance gate
                    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    d = ((tcx - bcx) ** 2 + (tcy - bcy) ** 2) ** 0.5
                    s = 0.01 if d < 4 * max(tb[2] - tb[0], tb[3] - tb[1], 10) else 0.0
                if s > best:
                    best, bi = s, i
            if bi >= 0:
                used.add(bi)
                b = boxes[bi]
                self.tracks[tid] = (b[:4], 0)
                self.hits[tid] = self.hits.get(tid, 0) + 1
                if self.hits[tid] >= self.min_hits:  # confirmed track only
                    out.append(Detection(tid, -1, "moving_object", float(b[4]), tuple(int(v) for v in b[:4])))
            elif misses + 1 > 10:
                del self.tracks[tid]
                self.hits.pop(tid, None)
            else:
                self.tracks[tid] = (tb, misses + 1)
        for i, b in enumerate(boxes):
            if i in used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = (b[:4], 0)
            self.hits[tid] = 1
        return out


class PixEagleYoloDetector:
    """Wraps PixEagle ``classes.backends.ultralytics_backend.UltralyticsBackend``."""

    def __init__(self, cfg: Dict[str, Any], repo_root: str) -> None:
        src = os.path.join(repo_root, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from classes.backends.detection_backend import DevicePreference
        from classes.backends.ultralytics_backend import UltralyticsBackend

        self.cfg = cfg
        self.backend = UltralyticsBackend({})
        if not self.backend.is_available:
            raise RuntimeError("ultralytics not installed (pip install -r requirements-ultralytics.txt)")
        dev = DevicePreference.CUDA if str(cfg.get("device", "cpu")).startswith("cuda") else DevicePreference.CPU
        self.backend.load_model(model_path=cfg["model_path"], device=dev, fallback_enabled=True,
                                context="skyguard_bridge")
        self.labels = self.backend.get_model_labels()
        self.classes = {str(c).lower() for c in cfg.get("classes") or []}
        self.name = f"pixeagle/{self.backend.backend_name}+{cfg.get('tracker', 'bytetrack')}"

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        _, dets = self.backend.detect_and_track(
            frame, conf=float(self.cfg["conf"]), iou=float(self.cfg["iou"]),
            max_det=int(self.cfg["max_det"]), tracker_type=str(self.cfg.get("tracker", "bytetrack")),
        )
        out = from_normalized(dets, self.labels)
        if self.classes:
            out = [d for d in out if d.class_name.lower() in self.classes]
        return out


def build_detector(cfg: Dict[str, Any], repo_root: str):
    kind = str(cfg.get("type", "auto"))
    if kind in ("yolo", "auto"):
        try:
            return PixEagleYoloDetector(cfg, repo_root)
        except Exception as exc:
            if kind == "yolo":
                raise
            log.warning("YOLO backend unavailable (%s); using motion detector", exc)
    return MotionDetector(cfg)
