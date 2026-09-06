"""config.template.py — Configuration template for the Microgreens Python Backend.

This is the ONLY file you need to edit when setting up your own deployment.
All other modules import constants from ``config``.

Setup:
    1. Copy this file to ``config.py`` (which is git-ignored and will never
       be committed): ``cp config.template.py config.py``
    2. Fill in ``BLYNK_AUTH_TOKEN`` once you generate it in the Blynk
       console (Template -> Devices -> Device Info).
    3. Fill in ``ESP32_CAM_IP`` once you note the IP from the ESP32-CAM's
       serial monitor output at boot.
    4. Agree on Virtual Pin numbers with your firmware and update the
       ``VIRTUAL_PINS`` dictionary. The keys are stable — only the values
       change.

Alternatively, leave the placeholders below as-is and supply real values
via environment variables (``BLYNK_AUTH_TOKEN``, ``ESP32_CAM_IP``,
``LOG_LEVEL``) — see ``main.py`` for the override logic. This is the
recommended approach for servers and CI.
"""

# ── !! FILL THESE IN BEFORE RUNNING (or set as environment variables) !! ─────

BLYNK_AUTH_TOKEN: str = "YOUR_BLYNK_TOKEN_HERE"
# Where to get it: Blynk Console -> your Template -> Devices -> click device
# -> Device Info tab -> Auth Token (copy button)

ESP32_CAM_IP: str = "YOUR_ESP32_CAM_IP_HERE"
# Where to get it: open Serial Monitor at 115200 baud after flashing the
# ESP32-CAM; the local IP is printed at boot (e.g. "192.168.1.42").

# ── Virtual Pin Map ────────────────────────────────────────────────────────
# Agree these with your firmware developer. Keys are semantic (never change
# them). Values are the Blynk virtual pin strings configured in the ESP32
# firmware. Format is always "V<number>".
#
#   DIRECTION KEY:
#     READ  = ESP32 writes this pin  -> Python reads it  (sensor data)
#     WRITE = Python writes this pin -> ESP32 reads it   (actuator commands)

VIRTUAL_PINS: dict = {
    # ── Sensor Inputs (READ) ─────────────────────────────────────────────
    "temperature":    "V0",
    "humidity":       "V1",
    "soil_moisture":  "V2",
    "light_sensor":   "V3",

    # ── Actuator Outputs (WRITE) ─────────────────────────────────────────
    "pump_command":   "V4",
    "fan_command":    "V5",
    "light_command":  "V6",

    # ── Overrides (READ) ───────────────────────────────────────────────────
    "manual_override": "V7",
    "light_override":  "V8",

    # ── Status / Metadata (READ) ───────────────────────────────────────────
    "device_status":   "V9",
    "uptime_seconds":  "V10",
}

# Convenience accessor (read from the dict — never hardcode pin strings elsewhere)
VP = VIRTUAL_PINS  # short alias: VP["temperature"] == "V0"

# ── Blynk REST API ────────────────────────────────────────────────────────
BLYNK_API_BASE_URL: str = "https://blynk.cloud/external/api"
# Full endpoint pattern examples:
#   GET  {BASE}/get?token={TOKEN}&{PIN}            -> read a pin value
#   GET  {BASE}/update?token={TOKEN}&{PIN}={VALUE} -> write a pin value

# ── ESP32-CAM Endpoints ───────────────────────────────────────────────────
ESP32_CAM_PORT: int = 8080
ESP32_CAM_SNAPSHOT_URL: str = f"http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/shot.jpg"
# The /capture endpoint is served by Arduino's standard CameraWebServer example.
# If your firmware uses a custom endpoint path, update the string above.

# ── Polling & Rate-Limit Settings ─────────────────────────────────────────
MAIN_LOOP_INTERVAL_SEC: int = 30       # Seconds between full sense->think->act cycles
HTTP_REQUEST_TIMEOUT_SEC: int = 10     # Abort any HTTP call that hangs beyond this
BLYNK_WRITE_COOLDOWN_SEC: float = 0.2  # Pause between consecutive Blynk writes
                                        # (Blynk free tier: ~1 req/s hard limit)

# ── Decision Thresholds (used by rule-based fallback & ML training labels) ──
SOIL_MOISTURE_MIN_PCT: float = 40.0  # Below -> turn pump ON
SOIL_MOISTURE_MAX_PCT: float = 75.0  # Above -> turn pump OFF
AIR_TEMP_MAX_C: float = 28.0         # Above -> turn fan ON
AIR_TEMP_MIN_C: float = 18.0         # Below -> turn fan OFF
HUMIDITY_MAX_PCT: float = 80.0       # Above -> increase ventilation
HUMIDITY_MIN_PCT: float = 50.0       # Below -> reduce ventilation

# ── Light Schedule (24-hour, server local time) ───────────────────────────
LIGHT_ON_HOUR: int = 6    # 06:00 -> lights ON
LIGHT_OFF_HOUR: int = 22  # 22:00 -> lights OFF

# ── File Paths ─────────────────────────────────────────────────────────────
LOG_FILE_PATH: str = "logs/backend.log"
SENSOR_LOG_CSV: str = "data/sensor_log.csv"
DECISION_LOG_CSV: str = "data/decision_log.csv"
IMAGE_SAVE_DIR: str = "images"

# ── ML Model Paths (set when your trained models are ready) ──────────────
SENSOR_MODEL_PATH: str = "models/sensor_model.pkl"    # scikit-learn Pipeline
VISION_MODEL_PATH: str = "models/vision_model.keras"  # Keras/TF SavedModel

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"  # Options: DEBUG | INFO | WARNING | ERROR
