import pandas as pd
import numpy as np
import os

def transform_dataset():
    print("Loading old data.csv...")
    df_old = pd.read_csv("data.csv")

    # 1. Parse timestamps to get cyclic hour features
    timestamps = pd.to_datetime(df_old['Timestamp'])
    hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
    hour_angles = 2 * np.pi * hours / 24.0

    # 2. Build the exact 9-element feature vector your backend expects
    df_new = pd.DataFrame()
    df_new['temperature_c']     = df_old['Ambient_Temperature']
    df_new['humidity_pct']      = df_old['Humidity']
    df_new['soil_moisture_pct'] = df_old['Soil_Moisture']
    df_new['hour_sin']          = np.sin(hour_angles)
    df_new['hour_cos']          = np.cos(hour_angles)
    df_new['is_light_hours']    = ((timestamps.dt.hour >= 6) & (timestamps.dt.hour < 22)).astype(float)
    
    # Mock the missing camera features so the Tabular model doesn't break
    df_new['img_green_ratio']   = 0.0
    df_new['img_veg_ratio']     = 0.0
    df_new['img_sharpness']     = 0.0

    # 3. Generate the target labels (Pump, Fan, Lights) using your config thresholds
    pump_on   = (df_new['soil_moisture_pct'] < 40.0).astype(int)
    fan_on    = ((df_new['temperature_c'] > 28.0) | (df_new['humidity_pct'] > 80.0)).astype(int)
    lights_on = df_new['is_light_hours'].astype(int)

    # Encode to 3-bit integer (bit 0 = pump, bit 1 = fan, bit 2 = lights)
    df_new['label_int'] = (
    pump_on
    + fan_on * 2
    + lights_on * 4
)

    # 4. Save to the path Claude will be looking for
    os.makedirs("data", exist_ok=True)
    save_path = "data/sensor_log.csv"
    df_new.to_csv(save_path, index=False)
    
    print(f"Success! Transformed {len(df_new)} rows.")
    print(f"Saved Claude-ready dataset to: {save_path}")

if __name__ == "__main__":
    transform_dataset()