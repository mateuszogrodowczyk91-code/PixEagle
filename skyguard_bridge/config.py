"""Configuration loading (YAML file + environment overrides)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_BASE_URL = "https://hfluinbjituqaaagdkhl.supabase.co/functions/v1/skyguard"

DEFAULTS: Dict[str, Any] = {
    "mode": "detection_only",
    "skyguard": {
        "enabled": False,               # master toggle, OFF by default
        "base_url": DEFAULT_BASE_URL,
        "token_env": "SKYGUARD_TOKEN_PIXEAGLE",
        "min_interval_s": 2.0,          # per track, between 'update' posts
        "post_start_end": True,
        "max_events_per_min": 60,
        "timeout_s": 8.0,
        "queue_size": 200,
    },
    "input": {
        "type": "ptz",                  # ptz | pixel | cloudedge | webcam | file | ingest | synthetic
        "label": "",
        "url": "",                      # ptz/pixel: rtsp://... or http://.../video
        "webcam_index": 0,
        "file": "resources/test1.mp4",
        "loop": False,
        "cloudedge": {
            "ip_env": "CLOUDEDGE_IP",
            "user_env": "CLOUDEDGE_USER",
            "password_env": "CLOUDEDGE_PASSWORD",
            "port": 8554,
            "stream": "main",           # main=101, sub=102
        },
        "ingest": {"host": "0.0.0.0", "port": 8765, "secret_env": "PIXEAGLE_INGEST_SECRET"},
    },
    "camera": {
        "hfov_deg": None,               # horizontal field of view, enables bearing
        "heading_deg": None,            # fixed mount azimuth (tripod), deg from north
        "ptz_onvif": {                  # READ-ONLY pan readout for bearing
            "enabled": False,
            "host": "",
            "port": 80,
            "user_env": "PTZ_USER",
            "password_env": "PTZ_PASSWORD",
            "profile_token": "",
            "pan_min_deg": -180.0,
            "pan_max_deg": 180.0,
            "pan_zero_heading_deg": 0.0,
            "poll_s": 1.0,
        },
    },
    "detector": {
        "type": "auto",                 # auto | onnx | directml | yolo | motion
        "model_path": "models/yolo26n.onnx",
        "device": "cpu",
        "conf": 0.35,
        "iou": 0.3,
        "max_det": 20,
        "tracker": "bytetrack",
        # Aerial classes only, so people/cars never enter the evidence chain.
        # COCO models: airplane/bird/kite; drone models: drone/uav/helicopter.
        # Empty list = all classes.
        "classes": ["drone", "uav", "airplane", "aircraft", "helicopter", "bird", "kite"],
        "motion_min_area_px": 40,
        "motion_max_area_frac": 0.2,
        "motion_max_blobs": 12,
        "motion_min_hits": 4,
        "lost_after_s": 1.5,
    },
    "runtime": {"max_fps": 10.0, "max_frames": 0, "headless": True},
}

INPUT_DEVICE_NAMES = {
    "pixel": "pixeagle-pixel",
    "ingest": "pixeagle-pixel",
    "ptz": "pixeagle-ptz",
    "cloudedge": "pixeagle-cloudedge",
    "webcam": "pixeagle-webcam",
    "file": "pixeagle-file",
    "synthetic": "pixeagle-synthetic",
}


def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _truthy(val: str) -> bool:
    return val.strip().lower() in {"1", "true", "yes", "on", "tak"}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        import yaml  # PyYAML is already a PixEagle dependency

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _merge(cfg, data)
    env = os.environ
    if "SKYGUARD_PIXEAGLE_ENABLED" in env:
        cfg["skyguard"]["enabled"] = _truthy(env["SKYGUARD_PIXEAGLE_ENABLED"])
    if env.get("SKYGUARD_HOST"):
        cfg["skyguard"]["base_url"] = env["SKYGUARD_HOST"].rstrip("/")
    return cfg


def device_name(cfg: Dict[str, Any]) -> str:
    return INPUT_DEVICE_NAMES.get(str(cfg["input"]["type"]), "pixeagle")
