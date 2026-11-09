#!/usr/bin/env bash
# SkyGuard passive visual sensor (detection only). Token from env SKYGUARD_TOKEN_PIXEAGLE.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -x .venv/bin/python ] && PY=.venv/bin/python
CFG="skyguard_sensor.local.yaml"
[ -f "$CFG" ] || CFG="skyguard_bridge/skyguard_sensor.yaml"
exec "$PY" -m skyguard_bridge -c "$CFG" "$@"
