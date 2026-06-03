"""
main.py — Microgreens Backend Service Entry Point
==================================================
This is the process you start on your server, Raspberry Pi, or any
machine with Python 3.10+ and network access to both Blynk Cloud and
the ESP32-CAM's local IP.

BACKGROUND SERVICE LOOP
  Every MAIN_LOOP_INTERVAL_SEC seconds:
    1. FETCH   — read sensor data from Blynk REST API
    2. FETCH   — capture image frame from ESP32-CAM
    3. INFER   — pass data to FusionController → get decision dict
    4. PUSH    — write actuator commands back to Blynk
    5. LOG     — append sensor + decision rows to CSV

MANUAL OVERRIDE SAFETY
  If the ESP32 signals manual_override=True (e.g., the physical operator
  has taken control via the Blynk dashboard), the Python backend skips
  the PUSH step entirely — it will never fight a human operator.

HOW TO RUN
  # Install dependencies first:
  pip install -r requirements.txt

  # Start the service:
  python main.py

  # To run as a persistent daemon on Linux (systemd):
  sudo cp microgreens.service /etc/systemd/system/
  sudo systemctl enable --now microgreens

ENVIRONMENT VARIABLES (optional, override config.py values)
  BLYNK_AUTH_TOKEN   — override BLYNK_AUTH_TOKEN from environment
  ESP32_CAM_IP       — override ESP32_CAM_IP from environment
  LOG_LEVEL          — override LOG_LEVEL  (e.g., LOG_LEVEL=DEBUG python main.py)
"""

import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from blynk_client import BlynkClient, SensorReading
from camera_client import CameraClient
from ml_model import FusionController

# ─────────────────────────────────────────────────────────────────────────────
# Environment variable overrides
# (allows secret injection without editing config.py on a server)
# ─────────────────────────────────────────────────────────────────────────────
if os.environ.get("BLYNK_AUTH_TOKEN"):
    config.BLYNK_AUTH_TOKEN = os.environ["BLYNK_AUTH_TOKEN"]

if os.environ.get("ESP32_CAM_IP"):
    config.ESP32_CAM_IP        = os.environ["ESP32_CAM_IP"]
    config.ESP32_CAM_SNAPSHOT_URL = (
        f"http://{config.ESP32_CAM_IP}:{config.ESP32_CAM_PORT}/capture"
    )

if os.environ.get("LOG_LEVEL"):
    config.LOG_LEVEL = os.environ["LOG_LEVEL"].upper()

# ─────────────────────────────────────────────────────────────────────────────
# Logging — file + stdout, configured before any other module logs
# ─────────────────────────────────────────────────────────────────────────────
Path(config.LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE_PATH, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
# CSV data loggers  (builds your training dataset over time)
# ─────────────────────────────────────────────────────────────────────────────

_SENSOR_CSV_HEADERS = [
    "timestamp_utc", "temperature_c", "humidity_pct", "soil_moisture_pct",
    "manual_override", "uptime_seconds", "device_status",
    "img_green_ratio", "img_veg_ratio", "img_sharpness"
]


_DECISION_CSV_HEADERS = [
    "timestamp_utc", "pump_on", "fan_on", "lights_on",
    "source", "confidence", "label_int",
]


def _append_csv(filepath: str, headers: list, row: list) -> None:
    """Write one row to a CSV file. Creates the file with a header if new."""
    path   = Path(filepath)
    is_new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(headers)
        writer.writerow(row)


def log_sensor(reading: SensorReading, image_features: dict = None) -> None:
    if image_features is None:
        image_features = {}
        
    ts = datetime.utcfromtimestamp(reading.timestamp).isoformat(timespec="seconds")
    _append_csv(
        config.SENSOR_LOG_CSV,
        _SENSOR_CSV_HEADERS,
        [
            ts,
            round(reading.temperature_c     or 0.0, 2),
            round(reading.humidity_pct      or 0.0, 2),
            round(reading.soil_moisture_pct or 0.0, 2),
            int(reading.manual_override),
            reading.uptime_seconds or -1,
            reading.device_status  or "",
            round(image_features.get("green_ratio", 0.0), 4),
            round(image_features.get("veg_ratio", 0.0), 4),
            round(image_features.get("sharpness", 0.0), 2),
        ],
    )


def log_decision(decisions: dict) -> None:
    from ml_model import encode_label
    ts    = datetime.utcnow().isoformat(timespec="seconds")
    label = encode_label(
        decisions.get("pump_on",   False),
        decisions.get("fan_on",    False),
        decisions.get("lights_on", False),
    )
    _append_csv(
        config.DECISION_LOG_CSV,
        _DECISION_CSV_HEADERS,
        [
            ts,
            int(decisions.get("pump_on",    False)),
            int(decisions.get("fan_on",     False)),
            int(decisions.get("lights_on",  False)),
            decisions.get("source",     "unknown"),
            round(decisions.get("confidence", 0.0), 4),
            label,
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown handler
# ─────────────────────────────────────────────────────────────────────────────
_keep_running: bool = True


def _on_shutdown(signum: int, _frame) -> None:
    global _keep_running
    logger.info("[Main] Shutdown signal (%d) received — finishing current cycle.", signum)
    _keep_running = False


signal.signal(signal.SIGINT,  _on_shutdown)
signal.signal(signal.SIGTERM, _on_shutdown)


# ─────────────────────────────────────────────────────────────────────────────
# Single sense → think → act cycle
# ─────────────────────────────────────────────────────────────────────────────

def run_cycle(
    blynk:      BlynkClient,
    camera:     CameraClient,
    controller: FusionController,
    cycle_num:  int,
) -> None:
    """
    One complete sense → think → act iteration.

    Failures inside a cycle are caught and logged — the service never
    crashes due to a single bad network call or inference error.
    """
    logger.info("━━━  Cycle #%04d  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", cycle_num)

    # ── STEP 1: FETCH sensor data from Blynk ─────────────────────────────────
    sensor_reading: SensorReading = blynk.fetch_sensors()
    log_sensor(sensor_reading)

    if not sensor_reading.is_usable():
        logger.warning("[Cycle] Sensor data unusable — skipping cycle.")
        return

    # ── STEP 3: FETCH image frame from ESP32-CAM ──────────────────────────────
    frame          = None
    image_features = {}

    if camera.is_reachable():
        frame = camera.fetch_frame()
        if frame is not None:
            image_features = camera.extract_features(frame)
            logger.info(
                "[Cycle] Image features — green=%.3f  veg=%.3f  sharp=%.1f",
                image_features.get("green_ratio", 0),
                image_features.get("veg_ratio",   0),
                image_features.get("sharpness",   0),
            )
        else:
            logger.warning("[Cycle] Frame capture failed — proceeding without image.")
    else:
        logger.warning("[Cycle] ESP32-CAM unreachable — skipping image capture.")

    # ── STEP 4: INFER — run ML / rule engine ─────────────────────────────────
    decisions: dict = controller.run_inference(sensor_reading, image_features, frame)

    # ── STEP 4.1: EXTREME HEAT SAFETY (The Swamp Cooler) ─────────────────────
    if getattr(sensor_reading, 'temperature_c', 0) and sensor_reading.temperature_c >= 32.0:
        # Only trigger safety override if the human operator hasn't taken manual control
        if not getattr(sensor_reading, 'manual_override', False):
            logger.warning("[Safety] Extreme heat (>32°C)! Triggering Swamp Cooler.")
            
            # 1. Turn Fan and Misters ON
            blynk.push_commands({"fan_on": True, "pump_on": True})
            
            # 2. Wait exactly 3 seconds for the mist to fill the tray
            import time
            time.sleep(3)
            
            # 3. Turn Misters OFF (Fan stays on to blow the cool air)
            blynk.push_commands({"pump_on": False})
            
            # 4. Tell the AI to keep the fan on, but leave the pump off so it doesn't flood
            decisions["fan_on"] = True
            decisions["pump_on"] = False

    # ── STEP 4.5: APPLY MANUAL OVERRIDES ─────────────────────────────────────
    if getattr(sensor_reading, 'manual_override', False):
        logger.info("[Override] Pump/Fan override active. Removing from AI control.")
        decisions.pop("pump_on", None)
        decisions.pop("fan_on", None)

    if getattr(sensor_reading, 'light_override', False):
        logger.info("[Override] Light override active. Removing from AI control.")
        decisions.pop("lights_on", None)

    # Log what the AI actually decided (using safe .get() since overrides might remove keys)
    logger.info(
        "[Cycle] Decision → pump=%-3s  fan=%-3s  lights=%-3s  "
        "source=%-10s  confidence=%.2f",
        "ON"  if decisions.get("pump_on", False)   else "OFF",
        "ON"  if decisions.get("fan_on", False)    else "OFF",
        "ON"  if decisions.get("lights_on", False) else "OFF",
        decisions.get("source", "unknown"),
        decisions.get("confidence", 0.0),
    )

    # ── STEP 5: PUSH commands to Blynk ───────────────────────────────────────
    success = blynk.push_commands(decisions)

    if success:
        log_decision(decisions)
        logger.info("[Cycle] ✓ Commands delivered to Blynk.")
    else:
        logger.error("[Cycle] ✗ One or more commands failed to deliver.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup checks
# ─────────────────────────────────────────────────────────────────────────────

def startup_checks(blynk: BlynkClient, camera: CameraClient) -> None:
    """Warn early about connectivity issues. Does NOT block startup."""
    logger.info("[Startup] Checking Blynk connectivity...")
    if blynk.is_reachable():
        logger.info("[Startup] ✓ Blynk cloud reachable.")
    else:
        logger.warning(
            "[Startup] ✗ Cannot reach Blynk. Check BLYNK_AUTH_TOKEN in config.py "
            "and your internet connection. Will keep retrying each cycle."
        )

    logger.info("[Startup] Checking ESP32-CAM connectivity...")
    if camera.is_reachable():
        logger.info("[Startup] ✓ ESP32-CAM reachable at %s.", config.ESP32_CAM_SNAPSHOT_URL)
    else:
        logger.warning(
            "[Startup] ✗ ESP32-CAM not reachable at %s. "
            "Check ESP32_CAM_IP in config.py. Vision features will be skipped.",
            config.ESP32_CAM_SNAPSHOT_URL,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   Microgreens Backend Service  v2.0.0            ║")
    logger.info("║   Cycle interval : %3ds                          ║", config.MAIN_LOOP_INTERVAL_SEC)
    logger.info("║   Log file       : %-28s ║", config.LOG_FILE_PATH)
    logger.info("╚══════════════════════════════════════════════════╝")

    # ── Instantiate clients and controller ───────────────────────────────────
    blynk      = BlynkClient()
    camera     = CameraClient()
    controller = FusionController()

    # ── Run startup connectivity checks ──────────────────────────────────────
    startup_checks(blynk, camera)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN BACKGROUND SERVICE LOOP
    # ─────────────────────────────────────────────────────────────────────────
    cycle_num = 0
    logger.info("[Main] Entering main service loop. Press Ctrl+C to stop.")

    while _keep_running:
        cycle_start = time.monotonic()
        cycle_num  += 1

        try:
            run_cycle(blynk, camera, controller, cycle_num)
        except Exception as exc:
            # Broad catch ensures a buggy cycle never kills the service
            logger.exception(
                "[Main] Unhandled exception in cycle #%d: %s — service continues.",
                cycle_num, exc,
            )

        # ── Sleep for the remainder of the configured interval ────────────────
        elapsed    = time.monotonic() - cycle_start
        sleep_time = max(0.0, config.MAIN_LOOP_INTERVAL_SEC - elapsed)

        logger.debug(
            "[Main] Cycle #%d complete in %.1fs — sleeping %.1fs.",
            cycle_num, elapsed, sleep_time,
        )

        # Break sleep into 1-second chunks so SIGTERM is handled promptly
        for _ in range(int(sleep_time)):
            if not _keep_running:
                break
            time.sleep(1)

    # ── Clean shutdown ────────────────────────────────────────────────────────
    logger.info("[Main] Shutdown: sending ALL-OFF safety command to Blynk...")
    blynk.push_commands({"pump_on": False, "fan_on": False, "lights_on": False})
    logger.info("[Main] Service stopped cleanly. Goodbye.")


if __name__ == "__main__":
    main()
