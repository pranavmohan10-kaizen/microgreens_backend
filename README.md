# 🌱 Microgreens Backend Service

This repository contains the backend code for our automated microgreens growing system! It connects our sensors and cameras to our IoT dashboard and runs the logic to keep the plants happy.

## 📁 File Breakdown

Here is exactly what each file in this repository does:

*   **`main.py`**: The heart of the system. This script runs a continuous loop (every 30 seconds) that fetches sensor data, takes photos, and triggers the watering system.
*   **`config.py`**: The configuration file. It holds our network settings, IP addresses, and secret API tokens.
*   **`ml_model.py`**: The brain of the operation. This contains the rule engine that analyzes soil moisture and camera data to make the final decision on whether to turn the water pump on.
*   **`blynk_client.py`**: The messenger for our dashboard. It sends our live sensor readings up to the Blynk IoT app and listens for any manual pump commands.
*   **`camera_client.py`**: The photographer. It connects to our ESP32-CAM over the local Wi-Fi to capture photos of the microgreens.
*   **`requirements.txt`**: The setup list. It lists all the external Python libraries needed to make this code work.

---

## 🚀 Next Steps: What We Need From Hardware

To get this code fully communicating with the physical world, we need three specific things from the hardware side:

1.  **Boot the ESP32-CAM:** Power up the camera module so it connects to the local Wi-Fi network.
2.  **Get the Camera IP Address:** Once it is booted, find its local IP address and send it over so we can replace the placeholder `ESP32_CAM_IP` inside `config.py`.
3.  **Generate the Blynk Auth Token:** Create the project in the Blynk app and send over the unique Auth Token so we can update `BLYNK_AUTH_TOKEN` in `config.py` and bring the dashboard live.
