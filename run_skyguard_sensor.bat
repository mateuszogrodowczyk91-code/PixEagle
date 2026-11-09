@echo off
REM SkyGuard passive visual sensor (detection only). Token from env SKYGUARD_TOKEN_PIXEAGLE.
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
set CFG=skyguard_sensor.local.yaml
if not exist %CFG% set CFG=skyguard_bridge\skyguard_sensor.yaml
%PY% -m skyguard_bridge -c %CFG% %*
