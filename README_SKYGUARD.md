# SkyGuard – pasywny czujnik wizualny (PixEagle)

Moduł `skyguard_bridge/` zamienia PixEagle w **pasywny czujnik** dla systemu SkyGuard:
obraz z kamery → detekcja (YOLO z PixEagle albo detektor ruchu OpenCV) → ślady
(track) → zdarzenia `uas_visual_track` w łańcuchu dowodowym SkyGuard
(`POST /api/v1/evidence`, nagłówek `x-skyguard-token`).

> **Tylko wykrywanie.** Moduł nie steruje dronem, gimbalem ani głowicą PTZ.
> Nie importuje MAVSDK/MAVLink, followera ani offboard commandera PixEagle
> (sprawdza to `skyguard_bridge/safety.py` przy starcie i na końcu pracy).
> Klucze `follow`, `offboard`, `mavlink_commands`, `auto_slew_to_cue`,
> `ptz_auto_track` ustawione na `true` blokują start. Głowicę PTZ obsługujesz
> ręcznie w jej aplikacji; moduł tylko **odczytuje** jej pozycję (ONVIF
> `GetStatus`), żeby policzyć namiar.

Moduł jest **opcjonalny i domyślnie wyłączony** (`skyguard.enabled: false`).
Bez `--send` / `SKYGUARD_PIXEAGLE_ENABLED=1` nic nie jest wysyłane.

## Wejścia wideo

| `input.type` | Do czego | Nazwa w payloadzie (`device`) |
|---|---|---|
| `ptz` | kamera PTZ na statywie (RTSP), główne zastosowanie na PC | `pixeagle-ptz` |
| `pixel` | strumień z telefonu (RTSP / HTTP MJPEG), opcjonalnie | `pixeagle-pixel` |
| `ingest` | klatki JPEG wysyłane HTTP `POST /frame` (np. z telefonu) | `pixeagle-pixel` |
| `cloudedge` | kamera CloudEdge, opcjonalnie (RTSP, hasło z `CLOUDEDGE_PASSWORD`) | `pixeagle-cloudedge` |
| `webcam` / `file` / `synthetic` | testy | `pixeagle-webcam` / `-file` / `-synthetic` |

Na samym telefonie Pixel detekcja działa w aplikacji SkyGuard (źródło
`pixel-visual`); ten moduł jest dla PTZ / CloudEdge / testów na PC.

Wszystkie urządzenia PixEagle używają **jednego tokena** `SKYGUARD_TOKEN_PIXEAGLE`
(serwer dopisuje `_device: "pixeagle"`); `device` i `input` mówią, z której kamery.

## Windows (komputer NBL)

```powershell
git clone -b feature/skyguard-sensor https://github.com/mateuszogrodowczyk91-code/PixEagle.git
cd PixEagle
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install opencv-python-headless numpy pyyaml
# YOLO (zalecane): pip install -r requirements-ultralytics.txt  oraz rejestracja modelu: python add_model.py --help
copy skyguard_bridge\skyguard_sensor.yaml skyguard_sensor.local.yaml   # tu wpisz url kamery PTZ, hfov_deg, heading_deg

# token: wpisz w swoim środowisku (nie w pliku w repo)
$env:SKYGUARD_TOKEN_PIXEAGLE = "<token pixeagle>"
$env:PIXEAGLE_PTZ_URL = "rtsp://user:haslo@192.168.1.50:554/stream1"

# test bez wysyłania
python -m skyguard_bridge -c skyguard_sensor.local.yaml --input synthetic --detector motion
# praca z wysyłaniem do SkyGuard
python -m skyguard_bridge -c skyguard_sensor.local.yaml --input ptz --send -v
```
Albo `run_skyguard_sensor.bat` (te same zmienne środowiskowe).

CloudEdge (opcjonalnie): w aplikacji CloudEdge włącz ONVIF, potem
`$env:CLOUDEDGE_IP="192.168.1.x"; $env:CLOUDEDGE_PASSWORD="..."` i `--input cloudedge`.

## Linux

```bash
git clone -b feature/skyguard-sensor https://github.com/mateuszogrodowczyk91-code/PixEagle.git
cd PixEagle && python3 -m venv .venv && . .venv/bin/activate
pip install opencv-python-headless numpy pyyaml
export SKYGUARD_TOKEN_PIXEAGLE=...        # np. z pliku tokens.env (chmod 600)
export PIXEAGLE_PTZ_URL='rtsp://user:haslo@192.168.1.50:554/stream1'
./run_skyguard_sensor.sh --input ptz --send -v
```

## Ustawienia (skyguard_bridge/skyguard_sensor.yaml)

- `skyguard.enabled` – wysyłanie (domyślnie **wyłączone**), `min_interval_s: 2` –
  najwyżej 1 aktualizacja na ślad co 2 s, plus zawsze `start` i `end` śladu;
  `max_events_per_min: 60` – limit globalny.
- `detector.type` – `auto` (YOLO z PixEagle, jeśli zainstalowane, inaczej ruch),
  `yolo`, `motion`. `detector.classes` – tylko klasy powietrzne
  (drone, uav, airplane, helicopter, bird, kite…), więc ludzie i auta nie trafiają do łańcucha.
- `camera.hfov_deg` + `camera.heading_deg` (statyw) albo `camera.ptz_onvif.enabled`
  (odczyt pan z PTZ) → pole `bearing_deg_est` (szacowany namiar).

## Payload

```json
{"source":"pixeagle","event":"uas_visual_track","device":"pixeagle-ptz","input":"ptz",
 "video_source":"ptz rtsp://192.168.1.50:554/stream1","mode":"detection_only",
 "phase":"start|update|end","track_id":3,"track_id_stable":true,"class_id":4,
 "class_name":"airplane","confidence":0.71,"max_confidence":0.74,
 "bbox_xyxy":[x1,y1,x2,y2],"bbox_norm":[...],"center_norm":[cx,cy],
 "frame_w":1920,"frame_h":1080,"frame_time":"2026-11-09T13:20:00.123Z",
 "track_started":"...","track_frames":12,"detector":"pixeagle/ultralytics+bytetrack",
 "bearing_deg_est":112.4,"bearing_basis":{"hfov_deg":60,"heading_deg":90}}
```
Hasła i tokeny nigdy nie trafiają do payloadu ani logów (URL-e są czyszczone).

## Testy

```bash
pip install pytest
python -m pytest tests/skyguard_bridge -q
python -m skyguard_bridge.mock_api --port 8787     # lokalna atrapa API do testów
```

## Licencja i autorstwo

PixEagle © Alireza Ghaderi i współtwórcy, licencja Apache 2.0 (plik `LICENSE`,
projekt źródłowy: https://github.com/alireza787b/PixEagle). Moduł
`skyguard_bridge/` jest dodatkiem na tej samej licencji; kod PixEagle nie został zmieniony
(dopisano tylko jedną linię w `.gitignore`).
