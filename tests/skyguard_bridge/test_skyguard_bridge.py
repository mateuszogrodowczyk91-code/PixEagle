import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from skyguard_bridge import safety  # noqa: E402
from skyguard_bridge.config import DEFAULTS, device_name  # noqa: E402
from skyguard_bridge.mock_api import MockEvidenceApi  # noqa: E402
from skyguard_bridge.runner import run  # noqa: E402
from skyguard_bridge.sink import EvidenceSink  # noqa: E402
from skyguard_bridge.sources import OnvifPanReader, cloudedge_rtsp_url, redact_url  # noqa: E402
from skyguard_bridge.tracks import Detection, TrackEventManager, estimate_bearing  # noqa: E402


def _mgr(**kw):
    return TrackEventManager(device="pixeagle-ptz", input_type="ptz", video_label="ptz test",
                             detector_name="t", **kw)


def test_rate_limit_start_update_end():
    m = _mgr(min_interval_s=2.0, lost_after_s=1.0)
    d = Detection(7, 0, "drone", 0.8, (10, 10, 20, 20))
    phases = []
    for i in range(50):  # 5 s at 10 fps
        phases += [e["phase"] for e in m.update([d], 100 + i / 10, (640, 360))]
    phases += [e["phase"] for e in m.update([], 107.0, (640, 360))]
    assert phases[0] == "start" and phases[-1] == "end"
    assert phases.count("update") == 2  # t=2.0, t=4.0
    assert phases.count("start") == 1 and phases.count("end") == 1


def test_payload_shape_and_bearing():
    m = _mgr(hfov_deg=60.0, heading_provider=lambda: 90.0)
    e = m.update([Detection(1, 0, "drone", 0.9, (480, 100, 500, 120))], 1.0, (640, 360))[0]
    assert e["source"] == "pixeagle" and e["event"] == "uas_visual_track"
    assert e["device"] == "pixeagle-ptz" and e["input"] == "ptz" and e["mode"] == "detection_only"
    assert e["bearing_deg_est"] == estimate_bearing(490 / 640, 60.0, 90.0)
    assert 105 < e["bearing_deg_est"] < 110


def test_device_names():
    for t, name in [("pixel", "pixeagle-pixel"), ("ptz", "pixeagle-ptz"), ("cloudedge", "pixeagle-cloudedge")]:
        c = copy.deepcopy(DEFAULTS)
        c["input"]["type"] = t
        assert device_name(c) == name


def test_safety_refuses_follow_and_offboard():
    for key in ("follow", "offboard", "auto_slew_to_cue", "ptz_auto_track", "mavlink_commands"):
        c = copy.deepcopy(DEFAULTS)
        c[key] = True
        with pytest.raises(safety.DetectionOnlyViolation):
            safety.assert_config_detection_only(c)
    c = copy.deepcopy(DEFAULTS)
    c["mode"] = "follow"
    with pytest.raises(safety.DetectionOnlyViolation):
        safety.assert_config_detection_only(c)


def test_disabled_by_default_no_posts():
    s = EvidenceSink.from_config(DEFAULTS["skyguard"])
    assert s.enabled is False and s.submit({"x": 1}) is False


def test_secrets_not_in_labels(monkeypatch):
    assert redact_url("rtsp://admin:geheim@10.0.0.5:554/s?token=abc") == "rtsp://10.0.0.5:554/s?token=***"
    monkeypatch.setenv("CLOUDEDGE_IP", "192.168.1.20")
    monkeypatch.setenv("CLOUDEDGE_PASSWORD", "p@ss/word")
    url, label = cloudedge_rtsp_url(DEFAULTS["input"]["cloudedge"])
    assert url.endswith("@192.168.1.20:8554/Streaming/Channels/101") and "p%40ss%2Fword" in url
    assert "p@ss" not in label and "admin" not in label
    monkeypatch.delenv("CLOUDEDGE_PASSWORD")
    assert cloudedge_rtsp_url(DEFAULTS["input"]["cloudedge"])[0] is None


def test_onvif_readonly_pan_parse():
    r = OnvifPanReader(dict(DEFAULTS["camera"]["ptz_onvif"], pan_zero_heading_deg=10))
    xml = '<tt:PanTilt x="0.5" y="0.1" space="x"/>'
    assert r.parse_pan(xml) == 0.5
    assert r.pan_to_heading(0.5) == 100.0
    env = r.envelope().decode()
    assert "GetStatus" in env and "Move" not in env


def test_end_to_end_synthetic_against_mock():
    before = set(safety.loaded_forbidden_modules())
    api = MockEvidenceApi("t0k3n", device="pixeagle").start()
    try:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["input"]["type"] = "synthetic"
        cfg["detector"]["type"] = "motion"
        cfg["skyguard"].update(enabled=True, base_url=api.base_url)
        sink = EvidenceSink(api.base_url, "t0k3n", enabled=True)
        stats = run(cfg, sink=sink)
    finally:
        api.stop()
    assert stats["sink_errors"] == [] and stats["sink_sent"] == stats["posted"] > 0
    phases = [r["payload"]["phase"] for r in api.rows]
    assert "start" in phases and "end" in phases
    assert all(r["payload"]["device"] == "pixeagle-synthetic" for r in api.rows)
    assert api.chain_ok()
    assert set(safety.loaded_forbidden_modules()) == before


def test_wrong_token_rejected():
    api = MockEvidenceApi("right").start()
    try:
        res = EvidenceSink(api.base_url, "wrong", enabled=True).post_now({"a": 1})
    finally:
        api.stop()
    assert res["status"] == 401


def test_global_event_cap():
    m = _mgr(max_events_per_min=5)
    out = []
    for i in range(20):
        out += m.update([Detection(100 + i, 0, "drone", 0.5, (1, 1, 5, 5))], 10.0 + i * 0.01, (64, 64))
    assert len(out) == 5 and m.dropped == 15


def test_preview_server_status_and_snapshot():
    import json as _json
    import urllib.request
    import numpy as np
    from skyguard_bridge.preview import PreviewServer
    srv = PreviewServer(0, max_fps=1000)
    try:
        srv.push(np.zeros((360, 640, 3), np.uint8), [Detection(1, 0, "drone", 0.9, (10, 10, 40, 30))])
        srv.status["frames"] = 1
        base = "http://127.0.0.1:%d" % srv.port
        assert _json.loads(urllib.request.urlopen(base + "/status").read())["frames"] == 1
        assert urllib.request.urlopen(base + "/snapshot.jpg").read()[:2] == b"\xff\xd8"
    finally:
        srv.close()
