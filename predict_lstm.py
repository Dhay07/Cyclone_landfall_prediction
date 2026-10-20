import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "lstm")

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_X_PATH = os.path.join(MODEL_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "scaler_y.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


# ============================================================
# Load configuration
# ============================================================

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

FEATURES = CONFIG["input_features"]
TARGET_FEATURES = CONFIG["target_features"]

WINDOW_SIZE = CONFIG["window_size"]
LEAD_TIME = CONFIG["lead_time"]

INTERVAL_HOURS = CONFIG["interval_hours"]
PREDICTION_HORIZON_HOURS = CONFIG["prediction_horizon_hours"]

MODEL_VERSION = CONFIG["model_version"]


# ============================================================
# Load model and scalers
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_X_PATH, "rb") as f:
    x_scaler = pickle.load(f)

with open(SCALER_Y_PATH, "rb") as f:
    y_scaler = pickle.load(f)


# ============================================================
# Prepare input sequence
# ============================================================

def prepare_track(track_df):

    # Check required features
    missing_features = [
        feature for feature in FEATURES
        if feature not in track_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required LSTM features: {missing_features}"
        )

    # Check minimum observations
    if len(track_df) < WINDOW_SIZE:
        raise ValueError(
            f"LSTM requires at least {WINDOW_SIZE} observations."
        )

    # Use latest WINDOW_SIZE observations
    recent_data = track_df[FEATURES].tail(WINDOW_SIZE).values

    # Convert to numeric array
    recent_data = np.asarray(
        recent_data,
        dtype=np.float32
    )

    # Scale using training scaler
    recent_scaled = x_scaler.transform(recent_data)

    # Reshape to:
    # (samples, time steps, features)
    X = recent_scaled.reshape(
        1,
        WINDOW_SIZE,
        len(FEATURES)
    )

    return X


# ============================================================
# Predict landfall
# ============================================================

def predict_landfall(track_df):

    if len(track_df) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} observations are required."
        )

    # Prepare latest two observations
    X = prepare_track(track_df)

    # Predict in scaled space
    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    # Convert prediction back to original units
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )[0]

    predicted_lat = float(prediction[0])
    predicted_lon = float(prediction[1])

    # Latest cyclone position
    latest = track_df.iloc[-1]

    result = {
        "predicted_lat": predicted_lat,
        "predicted_lon": predicted_lon,

        "current_lat": float(latest["Lat"]),
        "current_lon": float(latest["Lon"]),

        "number_observations": len(track_df),

        "window_size": WINDOW_SIZE,
        "lead_time": LEAD_TIME,

        "interval_hours": INTERVAL_HOURS,
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,

        "model_version": MODEL_VERSION
    }

    return result