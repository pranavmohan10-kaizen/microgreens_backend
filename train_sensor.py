"""
train_sensor.py
----------------
Trains the tabular decision engine for the microgreens automation system.

Model    : RandomForestClassifier (n_estimators=100, max_depth=6)
Pipeline : SimpleImputer(constant, 0.0) -> RandomForestClassifier
Data     : data/sensor_log.csv (read with pandas)
Target   : label_int (integer 0-7, a 3-bit pump/fan/lights actuator state)
Artifact : models/sensor_model.pkl  (joblib-serialized fitted Pipeline)

No external file dependencies beyond data/sensor_log.csv and scikit-learn.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_PATH = "data/sensor_log.csv"
MODEL_PATH = "models/sensor_model.pkl"

# The model is trained on EXACTLY these 9 columns, in EXACTLY this order.
# This ordering is the contract with the production backend: any inference
# vector must present features in the same sequence.
FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "soil_moisture_pct",
    "hour_sin",
    "hour_cos",
    "is_light_hours",
    "img_green_ratio",
    "img_veg_ratio",
    "img_sharpness",
]
TARGET_COLUMN = "label_int"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data():
    """Read the sensor log and split it into the feature matrix X and target y.

    Selecting FEATURE_COLUMNS explicitly (rather than "everything except the
    label") guarantees the strict 9-element ordering and will raise a clear
    KeyError if an expected column is missing from the CSV.
    """
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns missing from {DATA_PATH}: {missing}. "
            f"Expected features {FEATURE_COLUMNS} and target '{TARGET_COLUMN}'."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def build_pipeline():
    """Assemble the imputation + classification pipeline.

    Step 1 imputes any NaN values (e.g. dropped sensor readings) with the
    constant 0.0 BEFORE classification, so the RandomForest never sees a NaN.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} rows with {len(FEATURE_COLUMNS)} features.")

    # Stratify on the label so all actuator states appear in both splits when
    # class counts permit; fall back gracefully if any class is too small.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        print("Stratified split not possible (a class has too few samples); "
              "using a non-stratified split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Persist the fitted pipeline (imputer + classifier together).
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
