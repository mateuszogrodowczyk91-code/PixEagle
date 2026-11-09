"""Detection-only guard.

The SkyGuard integration is a passive sensor. Anything that steers, follows,
pursues or intercepts a target is out of scope and is refused here.
"""
from __future__ import annotations

import sys
from typing import Any, Mapping

# Modules that can command an aircraft, gimbal or follow a target.
FORBIDDEN_MODULE_PREFIXES = (
    "mavsdk",
    "pymavlink",
    "classes.follower",
    "classes.followers",
    "classes.offboard_commander",
    "classes.setpoint_sender",
    "classes.setpoint_handler",
    "classes.px4_interface_manager",
    "classes.gimbal_tracker",
)

# Config keys that would enable any kind of control; all must be false/absent.
FORBIDDEN_TRUE_KEYS = (
    "follow",
    "follower_enabled",
    "offboard",
    "offboard_enabled",
    "mavlink_commands",
    "auto_slew_to_cue",
    "ptz_auto_track",
    "intercept",
)


class DetectionOnlyViolation(RuntimeError):
    """Raised when the configuration or runtime is not detection-only."""


def assert_config_detection_only(cfg: Mapping[str, Any]) -> None:
    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                p = f"{path}.{k}" if path else str(k)
                if str(k).lower() in FORBIDDEN_TRUE_KEYS and bool(v):
                    raise DetectionOnlyViolation(
                        f"'{p}' is enabled; SkyGuard bridge runs detection-only"
                    )
                walk(v, p)

    walk(cfg, "")
    mode = str(cfg.get("mode", "detection_only"))
    if mode != "detection_only":
        raise DetectionOnlyViolation(f"mode must be 'detection_only', got {mode!r}")


def loaded_forbidden_modules() -> list[str]:
    return sorted(
        m for m in list(sys.modules)
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_MODULE_PREFIXES)
    )


def assert_runtime_detection_only(baseline: "frozenset[str] | None" = None) -> None:
    """Fail if control modules were loaded (beyond ``baseline``, e.g. a test harness)."""
    bad = [m for m in loaded_forbidden_modules() if not baseline or m not in baseline]
    if bad:
        raise DetectionOnlyViolation(
            "control modules loaded in detection-only process: " + ", ".join(bad)
        )
