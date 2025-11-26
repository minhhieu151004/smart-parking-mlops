import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import json
from io import StringIO
from datetime import datetime, timedelta

# Constants
N_STEPS = 288
FUTURE_STEP = 12  # 1 hour ahead (12 × 5 minutes)
TIME_STEP_MINUTES = 5

# Globals (SageMaker will reuse these between invokes)
scaler_car = None
scaler_hour = None


# ============================================================
# 1. Load Model
# ============================================================
def model_fn(model_dir):
    """Load model + scalers when container starts."""
    global scaler_car, scaler_hour

    model_path = os.path.join(model_dir, "1")
    scaler_car_path = os.path.join(model_dir, "scaler_car_count.pkl")
    scaler_hour_path = os.path.join(model_dir, "scaler_hour.pkl")

    print(f"[model_fn] Loading model at: {model_path}")
    model = tf.keras.models.load_model(model_path)

    scaler_car = joblib.load(scaler_car_path)
    scaler_hour = joblib.load(scaler_hour_path)

    print("[model_fn] Model + scalers loaded successfully")
    return model


# ============================================================
# 2. Parse Input
# ============================================================
def input_fn(request_body, content_type):
    """Convert JSON or CSV into a pandas DataFrame."""
    print(f"[input_fn] Received content type: {content_type}")

    if content_type == "application/json":
        data = json.loads(request_body)
        df = pd.DataFrame(data)

    elif content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    # ---- Clean car_count ----
    df["car_count"] = pd.to_numeric(df["car_count"], errors="coerce")
    df = df.dropna(subset=["car_count"])
    df["car_count"] = df["car_count"].astype("float32")

    # ---- Parse timestamp ----
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    print(f"[input_fn] Parsed {len(df)} rows")
    return df


# ============================================================
# INTERNAL PREPROCESSOR
# ============================================================
def _preprocess(df):
    """Resample + scale + extract last LSTM sequence."""
    df = df.set_index("timestamp").sort_index()

    df_resampled = df.resample(f"{TIME_STEP_MINUTES}T").mean().interpolate("time")

    df_resampled["hour"] = df_resampled.index.hour

    df_resampled["car_count_scaled"] = scaler_car.transform(df_resampled[["car_count"]])
    df_resampled["hour_scaled"] = scaler_hour.transform(df_resampled[["hour"]])

    seq = df_resampled[["car_count_scaled", "hour_scaled"]].values[-N_STEPS:]

    if len(seq) < N_STEPS:
        raise ValueError(f"Not enough data: need {N_STEPS} points.")

    return seq.reshape(1, N_STEPS, 2), df_resampled.index[-1]


# ============================================================
# 3. Predict
# ============================================================
def predict_fn(input_data, model):
    """Run prediction and return a dict (SageMaker serializable)."""
    print("[predict_fn] Running prediction...")

    X, last_timestamp = _preprocess(input_data)

    scaled_pred = model.predict(X)[0][0]
    actual_pred = scaler_car.inverse_transform([[scaled_pred]])[0][0]

    predicted_value = int(round(actual_pred))

    # Compute prediction timestamp
    future_minutes = TIME_STEP_MINUTES * FUTURE_STEP
    predicted_ts = (
        last_timestamp.floor(f"{TIME_STEP_MINUTES}T")
        + timedelta(minutes=future_minutes)
    )

    result = {
        "predicted_car_count": predicted_value,
        "for_timestamp": predicted_ts.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("[predict_fn] Prediction:", result)
    return result


# ============================================================
# 4. Output Formatter
# ============================================================
def output_fn(prediction, accept):
    """Convert Python dict → JSON string."""
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported Accept type: {accept}")
