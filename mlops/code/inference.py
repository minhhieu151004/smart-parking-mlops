import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import json
import logging
from io import StringIO
from datetime import datetime, timedelta

# Cấu hình log để xem lỗi trong CloudWatch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Hằng số ===
N_STEPS = 288
FUTURE_STEP = 12
TIME_STEP_MINUTES = 5


# === WRAPPER CLASS (QUAN TRỌNG) ===
# Chúng ta tạo class này để gói Model và Scaler đi chung với nhau
class ModelHandler:
    def __init__(self, model, scaler_car, scaler_hour):
        self.model = model
        self.scaler_car = scaler_car
        self.scaler_hour = scaler_hour


# ---------------------------------------------------------
# 1. MODEL_FN
# ---------------------------------------------------------
def model_fn(model_dir):
    """
    Load model và các scaler, sau đó đóng gói vào ModelHandler.
    """
    logger.info(f"Loading model artifacts from: {model_dir}")

    # DEBUG: In ra danh sách file để kiểm tra cấu trúc thư mục
    try:
        logger.info(f"Files in model_dir: {os.listdir(model_dir)}")
    except Exception as e:
        logger.warning(f"Could not list files: {e}")

    # Xử lý đường dẫn model (Hỗ trợ cả trường hợp có folder '1' hoặc không)
    path_with_version = os.path.join(model_dir, "1")
    if os.path.exists(path_with_version):
        model_path = path_with_version
    else:
        model_path = model_dir  # Fallback nếu model nằm ngay root

    scaler_car_path = os.path.join(model_dir, "scaler_car_count.pkl")
    scaler_hour_path = os.path.join(model_dir, "scaler_hour.pkl")

    # Load Model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Keras model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Keras model from {model_path}")
        raise e

    # Load Scalers
    try:
        scaler_car = joblib.load(scaler_car_path)
        scaler_hour = joblib.load(scaler_hour_path)
        logger.info("Scalers loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load scalers. Ensure .pkl files are in model.tar.gz")
        raise e

    # Trả về object chứa TẤT CẢ mọi thứ cần thiết
    return ModelHandler(model, scaler_car, scaler_hour)


# ---------------------------------------------------------
# 2. INPUT_FN
# ---------------------------------------------------------
def input_fn(request_body, content_type):
    logger.info(f"Received input with Content-Type: {content_type}")

    if content_type == "application/json":
        data = json.loads(request_body)
        df = pd.DataFrame(data)
    elif content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), sep=",")  # Sửa decode lỗi
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    # Data validation cơ bản
    if "car_count" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Input data must contain 'car_count' and 'timestamp' columns")

    # Xử lý format
    df["car_count"] = pd.to_numeric(df["car_count"], errors="coerce")
    df = df.dropna(subset=["car_count"])
    df["car_count"] = df["car_count"].astype("float32")

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        raise ValueError("Dataframe is empty after preprocessing.")

    return df


# ---------------------------------------------------------
# 3. PREDICT_FN
# ---------------------------------------------------------
def predict_fn(df, model_handler):
    """
    Nhận input_df và model_handler (được return từ model_fn)
    """
    logger.info("Executing prediction logic...")

    # Lấy model và scaler từ handler
    model = model_handler.model
    scaler_car = model_handler.scaler_car
    scaler_hour = model_handler.scaler_hour

    # --- Logic Preprocess nội bộ ---
    df = df.set_index("timestamp").sort_index()

    # Resample
    try:
        df_resampled = df.resample(f"{TIME_STEP_MINUTES}T").mean().interpolate("time")
    except ValueError as e:
        # Xử lý lỗi nếu format thời gian không chuẩn cho resample
        logger.error(f"Resample failed: {e}")
        raise e

    df_resampled["hour"] = df_resampled.index.hour

    # Transform
    df_resampled["car_count_scaled"] = scaler_car.transform(df_resampled[["car_count"]])
    df_resampled["hour_scaled"] = scaler_hour.transform(df_resampled[["hour"]])

    # Tạo sequence
    if len(df_resampled) < N_STEPS:
        msg = f"Not enough data points after resampling. Need {N_STEPS}, got {len(df_resampled)}."
        logger.error(msg)
        raise ValueError(msg)

    # Lấy đúng N_STEPS cuối cùng
    seq = df_resampled[["car_count_scaled", "hour_scaled"]].values[-N_STEPS:]
    X = seq.reshape(1, N_STEPS, 2)
    last_timestamp = df_resampled.index[-1]

    # --- Prediction ---
    try:
        scaled_pred = model.predict(X)[0][0]
    except Exception as e:
        logger.error(f"Error during model.predict: {e}")
        raise e

    actual_pred = scaler_car.inverse_transform([[scaled_pred]])[0][0]
    predicted_value = int(round(max(0, actual_pred)))  # Đảm bảo không âm

    # Tính timestamp tương lai
    future_minutes = TIME_STEP_MINUTES * FUTURE_STEP
    predicted_ts = (last_timestamp.floor(f"{TIME_STEP_MINUTES}T")
                    + timedelta(minutes=future_minutes))

    response = {
        "predicted_car_count": predicted_value,
        "for_timestamp": predicted_ts.strftime("%Y-%m-%d %H:%M:%S")
    }

    logger.info(f"Prediction result: {response}")
    return response


# ---------------------------------------------------------
# 4. OUTPUT_FN
# ---------------------------------------------------------
def output_fn(prediction, accept):
    logger.info(f"Formatting output for accept type: {accept}")
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")