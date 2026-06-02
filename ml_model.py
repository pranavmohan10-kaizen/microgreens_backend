"""
ml_model.py — Machine Learning Inference Engine
================================================
This module is the "brain" of the backend. It accepts structured data
from the two clients and outputs a decisions dictionary that main.py
feeds directly into BlynkClient.push_commands().

INPUT CONTRACT
--------------
run_inference(sensor_reading, image_features) takes:
  - sensor_reading  : SensorReading dataclass (from blynk_client.py)
  - image_features  : dict of scalar floats   (from camera_client.py)

OUTPUT CONTRACT (STABLE — main.py depends on this shape)
-----------------
Returns a dict with exactly these keys:

  {
    "pump_on":   bool,   # True  → activate water pump
    "fan_on":    bool,   # True  → activate ventilation fan
    "lights_on": bool,   # True  → activate grow lights
    "source":    str,    # "ml_sensor" | "ml_full" | "rules" | "safe_off"
    "confidence": float, # 0.0–1.0  (rules always return 1.0)
  }

DECISION LAYERS (priority order)
----------------------------------
  1. SensorModel  — scikit-learn pipeline trained on tabular sensor data
  2. VisionModel  — Keras/TF CNN trained on plant health images
  3. FusionLayer  — combines both; applies vision safety overrides
  4. RuleEngine   — deterministic threshold fallback (always available)

HOW TO ACTIVATE YOUR REAL MODELS
----------------------------------
  Sensor model (scikit-learn):
    a) Train a Pipeline and save: joblib.dump(pipeline, config.SENSOR_MODEL_PATH)
    b) Set  SensorModel.ENABLED = True
    c) Uncomment the 3-line block inside SensorModel._run()

  Vision model (Keras/TF):
    a) Train model and save: model.save(config.VISION_MODEL_PATH)
    b) Set  VisionModel.ENABLED = True
    c) Uncomment the 3-line block inside VisionModel._run()
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np

# Optional ML dependencies — degrade gracefully
try:
    import joblib as _joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

try:
    import tensorflow as _tf
    _TF_OK = True
except ImportError:
    _TF_OK = False

import config
from blynk_client import SensorReading

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  (shared utilities)
# ─────────────────────────────────────────────────────────────────────────────

def build_sensor_feature_vector(reading: SensorReading, image_features: dict) -> np.ndarray:
    """
    Merge sensor readings and image features into one flat NumPy vector.
    Add / remove features here and retrain your pipeline to match.

    Current 9-element vector:
      [0] temperature_c
      [1] humidity_pct
      [2] soil_moisture_pct
      [3] hour_sin              ← cyclic encoding (avoids 23→0 discontinuity)
      [4] hour_cos
      [5] is_light_hours        ← 1 during configured grow-light schedule
      [6] img_green_ratio       ← 0.0 if camera unavailable
      [7] img_veg_ratio
      [8] img_sharpness_norm    ← sharpness normalised to [0, 1] range
    """
    import math

    now   = datetime.now()
    hour  = now.hour + now.minute / 60.0
    angle = 2 * math.pi * hour / 24.0

    is_light_hours = float(config.LIGHT_ON_HOUR <= now.hour < config.LIGHT_OFF_HOUR)

    # Normalise sharpness to roughly [0, 1]  (500 = visually sharp baseline)
    sharpness_raw  = image_features.get("sharpness", 0.0)
    sharpness_norm = min(sharpness_raw / 500.0, 1.0)

    return np.array([
        reading.temperature_c     or 0.0,
        reading.humidity_pct      or 0.0,
        reading.soil_moisture_pct or 0.0,
        math.sin(angle),
        math.cos(angle),
        is_light_hours,
        image_features.get("green_ratio", 0.0),
        image_features.get("veg_ratio",   0.0),
        sharpness_norm,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Tabular Sensor Model  (scikit-learn)
# ─────────────────────────────────────────────────────────────────────────────

class SensorModel:
    """
    Wraps a trained scikit-learn Pipeline.

    Training label format:
      Each training sample maps a feature vector → a 3-bit integer label:
        bit 0 (LSB) = pump   (0/1)
        bit 1       = fan    (0/1)
        bit 2       = lights (0/1)
      So label 5 = 0b101 → pump=ON, fan=OFF, lights=ON.

    Helper: ml_model.encode_label(pump, fan, lights)  → int
            ml_model.decode_label(int)                → (pump, fan, lights)
    """

    # ── Set to True once you have a trained .pkl model ───────────────────────
    ENABLED: bool = True

    def __init__(self):
        self._pipeline = None
        if self.ENABLED:
            self._pipeline = self._load()

    def _load(self):
        import pathlib
        path = pathlib.Path(config.SENSOR_MODEL_PATH)
        if not _JOBLIB_OK:
            logger.error("[SensorModel] joblib not installed — cannot load model.")
            return None
        if not path.exists():
            logger.warning("[SensorModel] Model file not found: %s", path)
            return None
        try:
            pipeline = _joblib.load(path)
            logger.info("[SensorModel] Loaded pipeline from %s.", path)
            return pipeline
        except Exception as exc:
            logger.error("[SensorModel] Failed to load: %s", exc)
            return None

    def predict(self, feature_vector: np.ndarray) -> tuple[Optional[dict], float]:
        """
        Returns (partial_decision_dict, confidence) or (None, 0.0).
        """
        if self._pipeline is None:
            return None, 0.0

        # Run the actual Machine Learning math
        X          = feature_vector.reshape(1, -1)
        proba      = self._pipeline.predict_proba(X)[0]   
        label_idx  = int(np.argmax(proba))
        confidence = float(proba[label_idx])
        
        return decode_label(label_idx), confidence


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Vision Model  (Keras / TF)
# ─────────────────────────────────────────────────────────────────────────────

class VisionModel:
    """
    Wraps a trained Keras CNN for plant health classification.

    Expected output classes (define to match your training):
      "healthy"       → no override needed
      "wilting"       → increase watering
      "overwatered"   → stop watering immediately
      "mold_risk"     → force fan ON
      "light_stress"  → check lighting schedule
    """

    CLASS_LABELS = ["healthy", "stressed", "wilting"]
    INPUT_SIZE   = (224, 224)   # (height, width) — must match training

    # ── Set to True once you have a trained .h5 / SavedModel ─────────────────
    ENABLED: bool = True

    def __init__(self):
        self._model = None
        if self.ENABLED:
            self._model = self._load()

    def _load(self):
        import pathlib
        path = pathlib.Path(config.VISION_MODEL_PATH)
        if not _TF_OK:
            logger.error("[VisionModel] TensorFlow not installed — cannot load model.")
            return None
        if not path.exists():
            logger.warning("[VisionModel] Model file not found: %s", path)
            return None
        try:
            model = _tf.keras.models.load_model(str(path), compile=False)
            logger.info("[VisionModel] Loaded model from %s.", path)
            return model
        except Exception as exc:
            logger.error("[VisionModel] Failed to load: %s", exc)
            return None

    def predict(self, frame: Optional[np.ndarray]) -> tuple[str, float]:
        """
        Returns (health_label, confidence).
        Falls back to ("unknown", 0.0) when model or frame is absent.
        """
        if self._model is None or frame is None:
            return "unknown", 0.0

        try:
            import cv2
            resized = cv2.resize(frame, self.INPUT_SIZE)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

            proba = self._model.predict(tensor, verbose=0)[0]
            idx = int(np.argmax(proba))

            return self.CLASS_LABELS[idx], float(proba[idx])

        except Exception as exc:
            logger.error("[VisionModel] Inference error: %s", exc)
            return "unknown", 0.0

        


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Rule Engine  (always available, never wrong)
# ─────────────────────────────────────────────────────────────────────────────

def rule_engine(reading: SensorReading, vision_label: str = "unknown") -> dict:
    """
    Deterministic threshold-based decision maker.
    Used as the fallback when ML confidence is too low.
    Uses thresholds defined in config.py — change them there.

    Logic:
      PUMP ON  → soil moisture below minimum AND vision ≠ "overwatered"
      FAN  ON  → temperature above max OR humidity above max
                 OR vision == "mold_risk"
      LIGHTS ON → current hour is within the configured grow-light window
    """
    now  = datetime.now()
    hour = now.hour

    pump = (
        reading.soil_moisture_pct is not None
        and reading.soil_moisture_pct < config.SOIL_MOISTURE_MIN_PCT
        and vision_label != "overwatered"
    )

    fan = (
        (reading.temperature_c is not None and reading.temperature_c > config.AIR_TEMP_MAX_C)
        or (reading.humidity_pct is not None and reading.humidity_pct > config.HUMIDITY_MAX_PCT)
        or vision_label == "mold_risk"
    )

    lights = config.LIGHT_ON_HOUR <= hour < config.LIGHT_OFF_HOUR

    logger.info(
        "[RuleEngine] Decision → pump=%s  fan=%s  lights=%s "
        "(M=%.1f%%  T=%.1f°C  H=%.1f%%  vis=%s  hour=%02d)",
        "ON" if pump   else "OFF",
        "ON" if fan    else "OFF",
        "ON" if lights else "OFF",
        reading.soil_moisture_pct or -1,
        reading.temperature_c     or -1,
        reading.humidity_pct      or -1,
        vision_label, hour,
    )

    return {
        "pump_on":    pump,
        "fan_on":     fan,
        "lights_on":  lights,
        "source":     "rules",
        "confidence": 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Controller  ← the only class main.py touches
# ─────────────────────────────────────────────────────────────────────────────

# Minimum ML confidence required before trusting ML over rule engine
_CONFIDENCE_THRESHOLD: float = 0.70

class FusionController:
    """
    Orchestrates SensorModel + VisionModel and returns a final decision dict.

    main.py usage::
        controller = FusionController()
        decisions  = controller.run_inference(sensor_reading, image_features, frame)
        blynk.push_commands(decisions)

    The output dict shape is stable — main.py never needs updating when
    you swap model implementations.
    """

    def __init__(self):
        self.sensor_model = SensorModel()
        self.vision_model = VisionModel()
        logger.info(
            "[FusionController] Ready.  SensorModel.ENABLED=%s  VisionModel.ENABLED=%s",
            SensorModel.ENABLED, VisionModel.ENABLED,
        )

    def run_inference(
        self,
        sensor_reading: SensorReading,
        image_features: dict,
        frame: Optional["np.ndarray"] = None,
    ) -> dict:
        """
        Full inference pipeline.

        Args:
            sensor_reading : SensorReading from BlynkClient.fetch_sensors()
            image_features : dict from CameraClient.extract_features()
            frame          : raw BGR np.ndarray from CameraClient.fetch_frame()
                             (only needed once VisionModel.ENABLED = True)

        Returns:
            Decision dict with keys: pump_on, fan_on, lights_on, source, confidence
        """

        # ── Guard: invalid sensor data ────────────────────────────────────────
        if not sensor_reading.is_usable():
            logger.warning(
                "[FusionController] Sensor data not usable — returning safe OFF state."
            )
            return {
                "pump_on": False, "fan_on": False, "lights_on": False,
                "source": "safe_off", "confidence": 0.0,
            }

        # ── Layer 2: Vision inference ─────────────────────────────────────────
        vision_label, vision_conf = self.vision_model.predict(frame)
        logger.debug("[FusionController] Vision: label=%s  conf=%.2f", vision_label, vision_conf)

        # ── Layer 1: Sensor model inference ──────────────────────────────────
        feature_vec = build_sensor_feature_vector(sensor_reading, image_features)
        sensor_decision, sensor_conf = self.sensor_model.predict(feature_vec)
        logger.debug("[FusionController] SensorML: conf=%.2f", sensor_conf)

        # ── Layer 3: Fusion / fallback logic ──────────────────────────────────
        if sensor_conf >= _CONFIDENCE_THRESHOLD and sensor_decision is not None:
            # Apply vision-derived safety overrides on top of ML decision
            final = self._apply_vision_override(sensor_decision, vision_label, vision_conf)
            source = "ml_full" if vision_conf >= _CONFIDENCE_THRESHOLD else "ml_sensor"
            final.update({"source": source, "confidence": sensor_conf})
            logger.info("[FusionController] ML decision used (source=%s).", source)
            return final

        # Fall back to rule engine — always available, always correct
        return rule_engine(sensor_reading, vision_label)

    @staticmethod
    def _apply_vision_override(base: dict, vision_label: str, vision_conf: float) -> dict:
        """
        Apply safety overrides from the vision model onto a sensor-ML decision.
        Vision can only make things SAFER — never forces actuators ON without cause.
        """
        result = dict(base)   # copy

        if vision_label == "overwatered" and vision_conf >= _CONFIDENCE_THRESHOLD:
            result["pump_on"] = False   # Hard stop: never water if overwatered
            logger.warning("[FusionController] Vision override: pump FORCED OFF (overwatered).")

        if vision_label == "mold_risk" and vision_conf >= _CONFIDENCE_THRESHOLD:
            result["fan_on"] = True     # Force ventilation to combat mold
            logger.warning("[FusionController] Vision override: fan FORCED ON (mold_risk).")

        if vision_label == "wilting" and not result.get("pump_on"):
            result["pump_on"] = True    # Wilting detected: override ML to water
            logger.warning("[FusionController] Vision override: pump FORCED ON (wilting).")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Training Utilities  (helpers for dataset building)
# ─────────────────────────────────────────────────────────────────────────────

def encode_label(pump: bool, fan: bool, lights: bool) -> int:
    """Encode three booleans as a 3-bit integer for ML training labels."""
    return int(pump) | (int(fan) << 1) | (int(lights) << 2)


def decode_label(label: int) -> dict:
    """Decode a 3-bit integer label back to a partial decision dict."""
    return {
        "pump_on":   bool(label & 0b001),
        "fan_on":    bool(label & 0b010),
        "lights_on": bool(label & 0b100),
    }
