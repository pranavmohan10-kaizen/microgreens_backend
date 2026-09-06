# 🌱 Microgreens Backend — ML Fusion Controller

An autonomous, closed-loop control system for a microgreens grow environment. A Python service polls a Blynk-connected sensor rig and an ESP32-CAM every 30 seconds, fuses tabular sensor data with computer-vision plant-health signals, and drives the pump, fan, and grow lights — with hard-coded safety guardrails that no model is allowed to override.

## Architecture: The Machine Learning Fusion Controller

Most DIY grow-automation projects are either "dumb" threshold scripts or a single black-box model with no fallback. This backend is built as a **layered fusion controller** (`ml_model.py`) that combines multiple independent decision sources, ranked by trust:

```
┌─────────────────────────────────────────────────────────────┐
│  1. SensorModel   scikit-learn Pipeline over tabular sensor  │
│                    features (temp, humidity, soil moisture,  │
│                    cyclic time-of-day encoding)               │
│  2. VisionModel   Keras/TF CNN classifying ESP32-CAM frames   │
│                    into healthy / stressed / wilting          │
│  3. FusionLayer   Vision can only make decisions SAFER —      │
│                    it vetoes or forces actuators, never       │
│                    fights the sensor model arbitrarily        │
│  4. RuleEngine    Deterministic threshold fallback — always   │
│                    available, always correct, used whenever   │
│                    ML confidence drops below 70%               │
└─────────────────────────────────────────────────────────────┘
```

Every cycle, sensor data (`blynk_client.py`) and a camera frame (`camera_client.py`) are merged into one feature vector. If the sensor model's confidence clears the threshold, its decision is used and then checked against the vision model's read on plant health. If confidence is too low — or a model isn't trained yet — the system falls back to a transparent, deterministic rule engine. **The service is never left without a decision it can explain.**

## Heuristic Guardrails: Preventing AI Hallucinations From Drowning the Plants

An ML model is a probabilistic guess. Left unchecked, a confused model can decide to run the water pump when the soil is already saturated, or fail to react fast enough to a genuine heat emergency. This backend wraps every model decision in deterministic, non-negotiable safety logic that runs *after* inference, in `main.py::run_cycle`:

- **Overwatering Guardrail** — if soil moisture is already between 60–100%, any AI decision to turn the pump ON is vetoed outright, regardless of model confidence.
- **Swamp Cooler Emergency Protocol** — if air temperature reaches 32°C and no human has taken manual control, the system force-activates the fan and pumps a 3-second burst of water for evaporative cooling, then shuts the pump back off — independent of what any model recommended.
- **Manual Override Intercept** — if the hardware reports a human has taken control via the Blynk dashboard, the corresponding actuator (pump/fan or lights) is stripped from the decision entirely before it's pushed. The software will never fight a human operator.

These guardrails are intentionally simple, auditable, and untouchable by model weights — they are the last line of defense between a bad inference and a dead crop.

## Project Structure

| File | Responsibility |
|---|---|
| `main.py` | Service entry point — the sense → think → act loop, safety guardrails, CSV logging |
| `config.py` | Local configuration (git-ignored — never commit this) |
| `config.template.py` | Public configuration template — copy this to create `config.py` |
| `blynk_client.py` | Blynk REST API client — reads sensors, writes actuator commands |
| `camera_client.py` | ESP32-CAM HTTP client — fetches frames, extracts vision features |
| `ml_model.py` | The Fusion Controller — SensorModel, VisionModel, RuleEngine |
| `train_sensor.py` / `train_vision.py` | Training scripts for the two models |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-fork-url>
cd microgreens_backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your credentials

**Never edit `config.template.py` with real secrets — it is committed to the public repo.** Instead:

```bash
cp config.template.py config.py
```

Then either:

- **Edit `config.py` directly** and fill in `BLYNK_AUTH_TOKEN` and `ESP32_CAM_IP`, or
- **Use a `.env` file** (recommended — keeps secrets out of any Python file):

  ```bash
  echo "BLYNK_AUTH_TOKEN=your_real_token" >> .env
  echo "ESP32_CAM_IP=192.168.1.42" >> .env
  ```

  `config.py` automatically loads `.env` via `python-dotenv` if present. Both `config.py` and `.env` are excluded by `.gitignore` — they will never be committed.

Where to get each credential:

| Credential | Source |
|---|---|
| `BLYNK_AUTH_TOKEN` | Blynk Console → your Template → Devices → click device → Device Info tab |
| `ESP32_CAM_IP` | ESP32-CAM Serial Monitor (115200 baud) at boot — the local IP is printed there |

### 3. (Optional) Train your models

The fusion controller runs on the deterministic rule engine out of the box. To enable the ML layers:

```bash
python train_sensor.py   # produces models/sensor_model.pkl
python train_vision.py   # produces models/vision_model.keras
```

### 4. Run the service

```bash
python main.py
```

To run as a persistent daemon on Linux:

```bash
sudo cp microgreens.service /etc/systemd/system/
sudo systemctl enable --now microgreens
```

## Security Notes

- `config.py` and `.env` are git-ignored by default — do not force-add them.
- If you are open-sourcing a fork of a repo that **previously committed real secrets**, rotating those credentials is not optional: removing a token from the working tree does not remove it from git history. Rotate the Blynk token and treat the old ESP32-CAM IP as exposed, then scrub history (e.g. `git filter-repo`) before making the repo public.
- Model binaries (`*.pkl`, `*.keras`, `*.h5`) and sensor logs (`data/`, `images/`, `logs/`) are git-ignored by default — they're environment-specific and can be large. Use Git LFS or a model registry if you need to version them.

## License

Add your preferred license here before publishing (e.g. MIT, Apache-2.0).
