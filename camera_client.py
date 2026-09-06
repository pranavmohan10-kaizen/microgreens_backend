"""
camera_client.py — ESP32-CAM HTTP Client
=========================================
Responsibilities:
  - Fetch a JPEG snapshot from the ESP32-CAM's /capture endpoint via HTTP
  - Decode the JPEG into a NumPy array (OpenCV BGR) for the ML vision pipeline
  - Optionally save the raw image to disk for dataset building
  - Extract lightweight colour/texture features as a fallback when a
    full CNN model is not yet available

Public API (all main.py needs):
  camera = CameraClient()
  frame  = camera.fetch_frame()       # np.ndarray | None
  feats  = camera.extract_features(frame)  # dict of scalar metrics

OpenCV is a soft dependency — if absent, fetch_frame() returns None and
extract_features() returns an empty dict. The service keeps running.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

import config

logger = logging.getLogger(__name__)

# Optional dependency — degrade gracefully if not installed
try:
    import numpy as np
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning(
        "[CameraClient] OpenCV (cv2) not found. "
        "Install with: pip install opencv-python-headless\n"
        "Frame capture will return None until OpenCV is available."
    )


class CameraClient:
    """
    Fetches frames from the ESP32-CAM's built-in HTTP web server.

    The standard Arduino CameraWebServer sketch exposes:
      GET /capture  → single JPEG snapshot   ← this client uses this
      GET /stream   → MJPEG stream           ← for live dashboard use

    If the ESP32-CAM uses a different endpoint path, update
    ESP32_CAM_SNAPSHOT_URL in config.py — no code changes needed here.
    """

    def __init__(
        self,
        snapshot_url: str = config.ESP32_CAM_SNAPSHOT_URL,
        timeout: int      = config.HTTP_REQUEST_TIMEOUT_SEC,
        image_save_dir: str = config.IMAGE_SAVE_DIR,
        save_images: bool = True,
    ):
        if "YOUR_ESP32_CAM_IP_HERE" in snapshot_url:
            logger.warning(
                "[CameraClient] ESP32_CAM_IP is still the placeholder value. "
                "Update it in config.py (or the ESP32_CAM_IP env var) once "
                "the ESP32-CAM has booted."
            )

        self._url       = snapshot_url
        self._timeout   = timeout
        self._save_dir  = Path(image_save_dir)
        self._save_imgs = save_images

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MicrogreensPythonBackend/2.0"})

        if save_images:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_frame(self) -> Optional["np.ndarray"]:
        """
        Fetch one JPEG snapshot and decode it into a BGR NumPy array.

        Returns:
            np.ndarray of shape (H, W, 3) in BGR colour space, or
            None if the camera is unreachable, OpenCV is absent, or
            the JPEG could not be decoded.
        """
        if not _CV2_AVAILABLE:
            logger.error("[CameraClient] fetch_frame() called but OpenCV is not installed.")
            return None

        jpeg_bytes = self._http_get_jpeg()
        if jpeg_bytes is None:
            return None

        if self._save_imgs:
            self._save_jpeg(jpeg_bytes)

        return self._decode_jpeg(jpeg_bytes)

    def extract_features(self, frame: Optional["np.ndarray"]) -> dict:
        """
        Compute lightweight colour and texture statistics from a BGR frame.

        These scalar features serve two purposes:
          1. Fallback vision input when a full CNN model is not trained yet
          2. Supplementary features that can be appended to the sensor vector

        Args:
            frame: BGR NumPy array from fetch_frame(), or None.

        Returns:
            dict of named scalar floats, or empty dict if frame is None
            or OpenCV is unavailable.

        Feature glossary:
            green_ratio   — green channel dominance (healthy plant proxy)
            veg_ratio     — fraction of pixels in vegetation hue range (HSV H≈35–85)
            sharpness     — Laplacian variance (low = wilting/drooping leaves)
            brightness    — mean pixel brightness (detects light/dark conditions)
            mean_hue      — average hue value in HSV space
            mean_saturation — average colour saturation (low = yellowing)
        """
        if not _CV2_AVAILABLE or frame is None:
            return {}

        try:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            b, g, r = cv2.split(frame)
            b_mean  = float(np.mean(b))
            g_mean  = float(np.mean(g))
            r_mean  = float(np.mean(r))
            green_ratio = g_mean / (r_mean + b_mean + 1.0)   # +1 avoids div/0

            h_channel = hsv[:, :, 0]
            veg_mask  = cv2.inRange(h_channel, 35, 85)
            total_px  = frame.shape[0] * frame.shape[1]
            veg_ratio = float(np.sum(veg_mask > 0)) / float(total_px)

            sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))

            return {
                "green_ratio":      round(green_ratio, 4),
                "veg_ratio":        round(veg_ratio, 4),
                "sharpness":        round(sharpness, 2),
                "brightness":       round(brightness, 2),
                "mean_hue":         round(float(np.mean(h_channel)), 2),
                "mean_saturation":  round(float(np.mean(hsv[:, :, 1])), 2),
            }

        except Exception as exc:
            logger.error("[CameraClient] Feature extraction failed: %s", exc)
            return {}

    def is_reachable(self) -> bool:
        """
        Quick reachability check via HTTP HEAD.
        Returns True if the ESP32-CAM responds, False otherwise.
        Useful at startup to warn early if the camera is offline.
        """
        try:
            resp = self._session.head(self._url, timeout=self._timeout)
            return resp.status_code < 400
        except RequestException:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _http_get_jpeg(self) -> Optional[bytes]:
        """
        HTTP GET the snapshot URL. Returns raw JPEG bytes or None on failure.
        """
        try:
            response = self._session.get(
                self._url,
                timeout=self._timeout,
                stream=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "jpeg" not in content_type and "image" not in content_type:
                logger.warning(
                    "[CameraClient] Unexpected Content-Type '%s'. "
                    "Expected image/jpeg — proceeding anyway.",
                    content_type,
                )

            raw = response.content
            logger.debug("[CameraClient] Fetched %d bytes from %s.", len(raw), self._url)
            return raw

        except Timeout:
            logger.error(
                "[CameraClient] Request timed out after %ds — "
                "is ESP32-CAM online at %s?",
                self._timeout, self._url,
            )
        except ConnectionError:
            logger.error(
                "[CameraClient] Connection refused/unreachable — "
                "check ESP32_CAM_IP in config.py."
            )
        except RequestException as exc:
            logger.error("[CameraClient] HTTP error fetching frame: %s", exc)

        return None

    def _decode_jpeg(self, jpeg_bytes: bytes) -> Optional["np.ndarray"]:
        """
        Decode raw JPEG bytes to a BGR NumPy array via OpenCV.
        Returns None if decoding fails.
        """
        try:
            buf   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                logger.error(
                    "[CameraClient] cv2.imdecode returned None — "
                    "JPEG may be corrupted or truncated."
                )
            else:
                logger.info(
                    "[CameraClient] Frame decoded: %dx%d px.",
                    frame.shape[1], frame.shape[0],
                )
            return frame
        except Exception as exc:
            logger.error("[CameraClient] JPEG decode exception: %s", exc)
            return None

    def _save_jpeg(self, jpeg_bytes: bytes) -> None:
        """
        Persist raw JPEG to IMAGE_SAVE_DIR with a UTC timestamp filename.
        Failures are logged as warnings — never crash the main loop.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath  = self._save_dir / f"frame_{timestamp}.jpg"
        try:
            filepath.write_bytes(jpeg_bytes)
            logger.debug("[CameraClient] Saved frame → %s (%d B)", filepath, len(jpeg_bytes))
        except OSError as exc:
            logger.warning("[CameraClient] Could not save frame to disk: %s", exc)
