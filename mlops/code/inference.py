import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import json
import logging
import sys
from io import StringIO
from datetime import datetime, timedelta

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("INFERENCE.PY (CONV2D-LSTM VERSION) LOADING...")
logger.info(f"Python: {sys.version}")
logger.info(f"TensorFlow: {tf.__version__}")
logger.info("=" * 60)

# === HẰNG SỐ CẤU HÌNH ===
TOTAL_LOOK_BACK = 288  
LSTM_TIMESTEPS = 4     
ROWS = 8               
COLS = 9               
CHANNELS = 2           # (car_count, hour)

# Cấu hình dự đoán
FUTURE_STEP = 12       # Dự đoán 60 phút sau
TIME_STEP_MINUTES = 5


# === WRAPPER CLASS ===
class ModelHandler:
    """Wrapper class để giữ model và scalers"""
    def __init__(self, model, scaler_car, scaler_hour):
        self.model = model
        self.scaler_car = scaler_car
        self.scaler_hour = scaler_hour
        logger.info("✅ ModelHandler initialized")


# ---------------------------------------------------------
# 1. MODEL_FN - load model và scalers
# ---------------------------------------------------------
def model_fn(model_dir):
    """Load model và scalers từ model_dir."""
    try:
        logger.info("=" * 60)
        logger.info(f"MODEL_FN called with model_dir: {model_dir}")
        
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                level = root.replace(model_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    logger.info(f"{subindent}{f}")
        
        # --- 1. Load Model ---
        path_with_version = os.path.join(model_dir, "1")
        if os.path.exists(path_with_version):
            model_path = path_with_version
            logger.info(f"Using versioned model path: {model_path}")
        else:
            model_path = model_dir
            logger.info(f"Using root model path: {model_path}")

        logger.info(f"Loading Keras model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Keras model loaded successfully")

        # --- 2. Load Scalers ---
        scaler_car_path = os.path.join(model_dir, "scaler_car_count.pkl")
        scaler_hour_path = os.path.join(model_dir, "scaler_hour.pkl")

        if not os.path.exists(scaler_car_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_car_path}")
        if not os.path.exists(scaler_hour_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_hour_path}")

        scaler_car = joblib.load(scaler_car_path)
        scaler_hour = joblib.load(scaler_hour_path)
        logger.info("✅ Scalers loaded successfully")

        return ModelHandler(model, scaler_car, scaler_hour)

    except Exception as e:
        logger.error(f"❌ ERROR in model_fn: {e}", exc_info=True)
        raise


# ---------------------------------------------------------
# 2. INPUT_FN - Xử lý dữ liệu đầu vào
# ---------------------------------------------------------
def input_fn(request_body, content_type):
    """Parse dữ liệu đầu vào (JSON/CSV) thành DataFrame."""
    try:
        logger.info("=" * 60)
        logger.info(f"INPUT_FN called. Content-Type: {content_type}")
        
        if content_type == "application/json":
            data = json.loads(request_body)
            df = pd.DataFrame(data)
            
        elif content_type == "text/csv":
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            df = pd.read_csv(StringIO(request_body), sep=",")
            
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        if "car_count" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("Input missing required columns: 'car_count', 'timestamp'")

        df["car_count"] = pd.to_numeric(df["car_count"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        
        df = df.dropna(subset=["car_count", "timestamp"])
        df["car_count"] = df["car_count"].astype("float32")

        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing")

        logger.info(f"✅ Input parsed. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"❌ ERROR in input_fn: {e}", exc_info=True)
        raise


# ---------------------------------------------------------
# 3. PREDICT_FN - (ĐÃ SỬA CHO CONV2D)
# ---------------------------------------------------------
def predict_fn(input_data, model_handler):
    """
    Xử lý dữ liệu -> Reshape 5D -> Predict -> Inverse Scale.
    """
    try:
        logger.info("=" * 60)
        logger.info("PREDICT_FN called")

        model = model_handler.model
        scaler_car = model_handler.scaler_car
        scaler_hour = model_handler.scaler_hour

        # 1. Sắp xếp dữ liệu
        df = input_data.set_index("timestamp").sort_index()
        
        # 2. Resample & Interpolate 
        df_resampled = df.resample(f"{TIME_STEP_MINUTES}T").mean().interpolate("time")
        
        # 3. Feature Engineering (Hour)
        df_resampled["hour"] = df_resampled.index.hour 

        # 4. Scaling
        df_resampled["car_count_scaled"] = scaler_car.transform(df_resampled[["car_count"]])
        df_resampled["hour_scaled"] = scaler_hour.transform(df_resampled[["hour"]])
        
        if len(df_resampled) < TOTAL_LOOK_BACK:
            msg = f"❌ Not enough data! Need {TOTAL_LOOK_BACK} steps (24h), got {len(df_resampled)}"
            logger.error(msg)
            raise ValueError(msg)

        # 5. Tạo Sequence 
        seq_2d = df_resampled[["car_count_scaled", "hour_scaled"]].values[-TOTAL_LOOK_BACK:]
        last_timestamp = df_resampled.index[-1]
        
        logger.info(f"Sequence 2D shape: {seq_2d.shape}")

        # 6. RESHAPE
        try:
            # Reshape (4, 8, 9, 2)
            seq_reshaped = seq_2d.reshape(LSTM_TIMESTEPS, ROWS, COLS, CHANNELS)
            
            X = np.expand_dims(seq_reshaped, axis=0)
            
            logger.info(f"✅ Reshaped for Conv2D: {X.shape}")
            
        except ValueError as e:
            logger.error(f"Reshape error. Check dimensions: {TOTAL_LOOK_BACK} vs {LSTM_TIMESTEPS}x{ROWS}x{COLS}")
            raise e

        # 7. Predict
        logger.info("Making prediction...")
        scaled_pred = model.predict(X, verbose=0)[0][0]
        logger.info(f"Raw scaled prediction: {scaled_pred}")

        # 8. Inverse kết quả về số nguyên
        actual_pred = scaler_car.inverse_transform([[scaled_pred]])[0][0]
        
        # Làm tròn số xe 
        predicted_value = int(round(max(0, actual_pred)))
        logger.info(f"Final prediction: {predicted_value} cars")

        # 9. Tính thời gian dự đoán (prediction for)
        future_minutes = TIME_STEP_MINUTES * FUTURE_STEP
        predicted_ts = (last_timestamp.floor(f"{TIME_STEP_MINUTES}T") + 
                        timedelta(minutes=future_minutes))

        response = {
            "predicted_car_count": predicted_value,
            "for_timestamp": predicted_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "input_last_timestamp": last_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return response

    except Exception as e:
        logger.error(f"❌ ERROR in predict_fn: {e}", exc_info=True)
        raise


# ---------------------------------------------------------
# 4. OUTPUT_FN - Logic trả kết quả
# ---------------------------------------------------------
def output_fn(prediction, accept):
    """Format output trả về."""
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")