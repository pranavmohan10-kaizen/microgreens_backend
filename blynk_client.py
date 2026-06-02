"""
blynk_client.py — Blynk REST API Client
========================================
Responsibilities:
  - Read sensor values from Blynk virtual pins (data the ESP32 pushed up)
  - Write actuator commands to Blynk virtual pins (Python pushes down)
  - Handle all HTTP errors, timeouts, and malformed responses gracefully

The rest of the application only ever touches two public methods:
  - BlynkClient.fetch_sensors()  → returns SensorReading dataclass
  - BlynkClient.push_commands()  → accepts DecisionOutput dict, returns bool

Blynk External HTTP API reference:
  https://docs.blynk.io/en/blynk.cloud/get-datastream-value
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """
    Represents one complete snapshot of all sensor virtual pins.
    Fields default to None so callers can check if a read succeeded.
    """
    timestamp: float             = field(default_factory=time.time)

    # Core environment sensors
    temperature_c:    Optional[float] = None   # °C   — from VP["temperature"]
    humidity_pct:     Optional[float] = None   # %    — from VP["humidity"]
    soil_moisture_pct: Optional[float] = None  # %    — from VP["soil_moisture"]

    # Metadata
    device_status:    Optional[str]   = None   # freeform string from ESP32
    manual_override:  bool            = False  # True = operator has control
    uptime_seconds:   Optional[int]   = None   # ESP32 uptime in seconds

    def is_usable(self) -> bool:
        """
        Return True only when the three core sensor values are present
        and within physically plausible ranges. main.py calls this before
        passing the reading to the ML model.
        """
        return (
            self.temperature_c    is not None and -40 <= self.temperature_c <= 85
            and self.humidity_pct is not None and   0 <= self.humidity_pct  <= 100
            and self.soil_moisture_pct is not None and 0 <= self.soil_moisture_pct <= 100
        )

    def as_dict(self) -> dict:
        """Return core sensor fields as a plain dict (for ML feature extraction)."""
        return {
            "temperature_c":     self.temperature_c,
            "humidity_pct":      self.humidity_pct,
            "soil_moisture_pct": self.soil_moisture_pct,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class BlynkClient:
    """
    Thin, stateless wrapper around the Blynk External HTTP API.

    All methods are safe to call in a loop — they catch every network
    exception and return None / False rather than crashing the service.
    """

    def __init__(
        self,
        auth_token: str  = config.BLYNK_AUTH_TOKEN,
        base_url: str    = config.BLYNK_API_BASE_URL,
        timeout: int     = config.HTTP_REQUEST_TIMEOUT_SEC,
        write_cooldown: float = config.BLYNK_WRITE_COOLDOWN_SEC,
    ):
        if auth_token == "YOUR_BLYNK_AUTH_TOKEN_HERE":
            logger.warning(
                "[BlynkClient] Auth token is still the placeholder value. "
                "Update BLYNK_AUTH_TOKEN in config.py before running."
            )

        self._token        = auth_token
        self._base         = base_url.rstrip("/")
        self._timeout      = timeout
        self._write_cooldown = write_cooldown

        # Persistent session: reuses TCP connection, adds User-Agent
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MicrogreensPythonBackend/2.0"})

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_sensors(self) -> SensorReading:
        """
        Read all sensor virtual pins from Blynk and return a SensorReading.

        On any network or parsing failure the corresponding field stays None.
        The caller should check reading.is_usable() before proceeding.

        Returns:
            SensorReading — always returns an object, never raises.
        """
        reading = SensorReading()

        # Map config pin keys → SensorReading field names + type coercion
        pin_map = {
            "temperature":    ("temperature_c",     float),
            "humidity":       ("humidity_pct",      float),
            "soil_moisture":  ("soil_moisture_pct", float),
            "uptime_seconds": ("uptime_seconds",    int),
            "device_status":  ("device_status",     str),
            "manual_override":("manual_override",   lambda v: bool(int(float(v)))),
        }

        for pin_key, (attr_name, coerce) in pin_map.items():
            raw = self._read_pin(config.VP[pin_key])
            if raw is not None:
                try:
                    setattr(reading, attr_name, coerce(raw))
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "[BlynkClient] Could not parse %s='%s': %s",
                        pin_key, raw, exc,
                    )

        logger.info(
            "[BlynkClient] Sensor fetch complete — "
            "T=%.1f°C  H=%.1f%%  M=%.1f%%  override=%s  usable=%s",
            reading.temperature_c    or float("nan"),
            reading.humidity_pct     or float("nan"),
            reading.soil_moisture_pct or float("nan"),
            reading.manual_override,
            reading.is_usable(),
        )
        return reading

    def push_commands(self, decisions: dict) -> bool:
        """
        Write actuator decisions back to Blynk virtual pins.

        Args:
            decisions: dict output from ml_model.run_inference(), e.g.
                       {"pump_on": True, "fan_on": False, "lights_on": True}

        Returns:
            True if ALL writes succeeded, False if any failed.

        The method respects BLYNK_WRITE_COOLDOWN_SEC between writes to avoid
        hitting Blynk's free-tier rate limit (~1 request/second).
        """
        # Map decision keys → config virtual pin keys
        write_plan = [
            ("pump_on",   "pump_command",  decisions.get("pump_on",   False)),
            ("fan_on",    "fan_command",   decisions.get("fan_on",    False)),
            ("lights_on", "light_command", decisions.get("lights_on", False)),
        ]

        all_ok = True
        for decision_key, pin_key, state in write_plan:
            value  = 1 if state else 0
            pin    = config.VP[pin_key]
            ok     = self._write_pin(pin, value)
            if not ok:
                logger.error(
                    "[BlynkClient] Failed to write %s=%d to %s.",
                    decision_key, value, pin,
                )
                all_ok = False
            time.sleep(self._write_cooldown)   # respect rate limit

        if all_ok:
            logger.info(
                "[BlynkClient] Commands pushed → pump=%s  fan=%s  lights=%s",
                "ON" if decisions.get("pump_on")   else "OFF",
                "ON" if decisions.get("fan_on")    else "OFF",
                "ON" if decisions.get("lights_on") else "OFF",
            )
        return all_ok

    def is_reachable(self) -> bool:
        """
        Quick connectivity check. Reads the temperature pin; returns True
        if Blynk responds with any valid HTTP 200. Useful at startup.
        """
        return self._read_pin(config.VP["temperature"]) is not None

    # ─────────────────────────────────────────────────────────────────────────
    # Private HTTP helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_pin(self, pin: str) -> Optional[str]:
        """
        GET /get?token=...&{pin}
        Returns the raw string value from Blynk, or None on any error.
        """
        url    = f"{self._base}/get"
        params = {"token": self._token, pin: ""}

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()

            # Blynk returns a JSON array ["value"] OR a raw number/string
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return str(data[0])
            elif isinstance(data, (int, float, str)):
                return str(data)

            logger.warning("[BlynkClient] Unexpected response format for %s: %s", pin, data)
            return None
        
        except Timeout:
            logger.error("[BlynkClient] Timeout reading %s (limit=%ds).", pin, self._timeout)
        except ConnectionError:
            logger.error("[BlynkClient] Network unreachable while reading %s.", pin)
        except RequestException as exc:
            logger.error("[BlynkClient] HTTP error reading %s: %s", pin, exc)
        except (ValueError, KeyError) as exc:
            logger.error("[BlynkClient] JSON parse error for %s: %s", pin, exc)

        return None

    def _write_pin(self, pin: str, value: int) -> bool:
        """
        GET /update?token=...&{pin}={value}
        Returns True on HTTP 200, False on any error.

        Blynk uses a GET-based write (not POST) for the external HTTP API.
        """
        url    = f"{self._base}/update"
        params = {"token": self._token, pin: str(value)}

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            return True

        except Timeout:
            logger.error("[BlynkClient] Timeout writing %s=%s.", pin, value)
        except ConnectionError:
            logger.error("[BlynkClient] Network unreachable while writing %s.", pin)
        except RequestException as exc:
            logger.error("[BlynkClient] HTTP error writing %s=%s: %s", pin, value, exc)

        return False
