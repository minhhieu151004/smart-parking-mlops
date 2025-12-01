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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("INFERENCE.PY IS BEING LOADED")
logger.info(f"Python version: {sys.version}")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info("=" * 60)

# === Hằng số ===
N_STEPS = 288
FUTURE_STEP = 12
TIME_STEP_MINUTES = 5


# === WRAPPER CLASS ===
class ModelHandler:
    """Wrapper class to hold model and scalers together"""
    def __init__(self, model, scaler_car, scaler_hour):
        self.model = model
        self.scaler_car = scaler_car
        self.scaler_hour = scaler_hour
        logger.info("✅ ModelHandler initialized")


# ---------------------------------------------------------
# 1. MODEL_FN - REQUIRED BY SAGEMAKER
# ---------------------------------------------------------
def model_fn(model_dir):
    """
    Load model and scalers from model_dir.
    This function is REQUIRED by SageMaker.
    
    Args:
        model_dir (str): Directory containing model artifacts
        
    Returns:
        ModelHandler: Object containing model and scalers
    """
    try:
        logger.info("=" * 60)
        logger.info(f"MODEL_FN called with model_dir: {model_dir}")
        
        # List all files in model_dir for debugging
        if os.path.exists(model_dir):
            logger.info("Files in model_dir:")
            for root, dirs, files in os.walk(model_dir):
                level = root.replace(model_dir, '').count(os.sep)
                indent = '  ' * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = '  ' * (level + 1)
                for file in files:
                    logger.info(f"{subindent}{file}")
        else:
            logger.error(f"❌ model_dir does not exist: {model_dir}")
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        # Try to load model from versioned path first
        path_with_version = os.path.join(model_dir, "1")
        if os.path.exists(path_with_version):
            model_path = path_with_version
            logger.info(f"Using versioned model path: {model_path}")
        else:
            model_path = model_dir
            logger.info(f"Using root model path: {model_path}")

        # Define scaler paths
        scaler_car_path = os.path.join(model_dir, "scaler_car_count.pkl")
        scaler_hour_path = os.path.join(model_dir, "scaler_hour.pkl")

        # Load TensorFlow model
        logger.info(f"Loading Keras model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Keras model loaded successfully")

        # Load scalers
        logger.info(f"Loading scaler_car from: {scaler_car_path}")
        if not os.path.exists(scaler_car_path):
            raise FileNotFoundError(f"scaler_car_count.pkl not found at {scaler_car_path}")
        scaler_car = joblib.load(scaler_car_path)
        logger.info("✅ scaler_car loaded successfully")

        logger.info(f"Loading scaler_hour from: {scaler_hour_path}")
        if not os.path.exists(scaler_hour_path):
            raise FileNotFoundError(f"scaler_hour.pkl not found at {scaler_hour_path}")
        scaler_hour = joblib.load(scaler_hour_path)
        logger.info("✅ scaler_hour loaded successfully")

        # Create and return ModelHandler
        handler = ModelHandler(model, scaler_car, scaler_hour)
        logger.info("✅ model_fn completed successfully")
        logger.info("=" * 60)
        return handler

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ ERROR in model_fn: {e}", exc_info=True)
        logger.error("=" * 60)
        raise


# ---------------------------------------------------------
# 2. INPUT_FN - REQUIRED BY SAGEMAKER
# ---------------------------------------------------------
def input_fn(request_body, content_type):
    """
    Parse and preprocess input data.
    This function is REQUIRED by SageMaker.
    
    Args:
        request_body: Raw request body
        content_type: Content type of the request
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    try:
        logger.info("=" * 60)
        logger.info(f"INPUT_FN called with content_type: {content_type}")
        
        if content_type == "application/json":
            data = json.loads(request_body)
            df = pd.DataFrame(data)
            logger.info(f"Parsed JSON data, shape: {df.shape}")
            
        elif content_type == "text/csv":
            # Handle bytes input
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            df = pd.read_csv(StringIO(request_body), sep=",")
            logger.info(f"Parsed CSV data, shape: {df.shape}")
            
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        # Validate required columns
        if "car_count" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("Input data must contain 'car_count' and 'timestamp' columns")

        # Process car_count
        df["car_count"] = pd.to_numeric(df["car_count"], errors="coerce")
        df = df.dropna(subset=["car_count"])
        df["car_count"] = df["car_count"].astype("float32")

        # Process timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing")

        logger.info(f"✅ Input processed successfully. Final shape: {df.shape}")
        logger.info("=" * 60)
        return df

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ ERROR in input_fn: {e}", exc_info=True)
        logger.error("=" * 60)
        raise


# ---------------------------------------------------------
# 3. PREDICT_FN - REQUIRED BY SAGEMAKER
# ---------------------------------------------------------
def predict_fn(input_data, model_handler):
    """
    Make predictions using the model.
    This function is REQUIRED by SageMaker.
    
    Args:
        input_data: Preprocessed DataFrame from input_fn
        model_handler: ModelHandler object from model_fn
        
    Returns:
        dict: Prediction results
    """
    try:
        logger.info("=" * 60)
        logger.info("PREDICT_FN called")

        # Extract model and scalers
        model = model_handler.model
        scaler_car = model_handler.scaler_car
        scaler_hour = model_handler.scaler_hour

        # Set timestamp as index and sort
        df = input_data.set_index("timestamp").sort_index()
        logger.info(f"Data sorted by timestamp, length: {len(df)}")

        # Resample to 5-minute intervals
        try:
            df_resampled = df.resample(f"{TIME_STEP_MINUTES}T").mean().interpolate("time")
            logger.info(f"Data resampled to {len(df_resampled)} rows")
        except Exception as e:
            logger.error(f"Resample failed: {e}")
            raise

        # Add hour feature
        df_resampled["hour"] = df_resampled.index.hour

        # Scale features
        df_resampled["car_count_scaled"] = scaler_car.transform(df_resampled[["car_count"]])
        df_resampled["hour_scaled"] = scaler_hour.transform(df_resampled[["hour"]])
        logger.info("Features scaled successfully")

        # Check if we have enough data
        if len(df_resampled) < N_STEPS:
            msg = f"Not enough data. Need {N_STEPS}, got {len(df_resampled)}"
            logger.error(msg)
            raise ValueError(msg)

        # Create sequence (take last N_STEPS)
        seq = df_resampled[["car_count_scaled", "hour_scaled"]].values[-N_STEPS:]
        X = seq.reshape(1, N_STEPS, 2)
        last_timestamp = df_resampled.index[-1]
        logger.info(f"Sequence created. Shape: {X.shape}, Last timestamp: {last_timestamp}")

        # Make prediction
        logger.info("Making prediction...")
        scaled_pred = model.predict(X, verbose=0)[0][0]
        logger.info(f"Scaled prediction: {scaled_pred}")

        # Inverse transform
        actual_pred = scaler_car.inverse_transform([[scaled_pred]])[0][0]
        predicted_value = int(round(max(0, actual_pred)))  # Ensure non-negative
        logger.info(f"Actual prediction: {predicted_value}")

        # Calculate future timestamp
        future_minutes = TIME_STEP_MINUTES * FUTURE_STEP
        predicted_ts = (last_timestamp.floor(f"{TIME_STEP_MINUTES}T") + 
                       timedelta(minutes=future_minutes))

        response = {
            "predicted_car_count": predicted_value,
            "for_timestamp": predicted_ts.strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"✅ Prediction completed: {response}")
        logger.info("=" * 60)
        return response

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ ERROR in predict_fn: {e}", exc_info=True)
        logger.error("=" * 60)
        raise


# ---------------------------------------------------------
# 4. OUTPUT_FN - REQUIRED BY SAGEMAKER
# ---------------------------------------------------------
def output_fn(prediction, accept):
    """
    Format the prediction output.
    This function is REQUIRED by SageMaker.
    
    Args:
        prediction: Prediction result from predict_fn
        accept: Requested output format
        
    Returns:
        str: Formatted output
    """
    try:
        logger.info("=" * 60)
        logger.info(f"OUTPUT_FN called with accept: {accept}")
        
        if accept == "application/json" or accept == "*/*":
            output = json.dumps(prediction)
            logger.info(f"✅ Output formatted as JSON")
            logger.info("=" * 60)
            return output
        else:
            raise ValueError(f"Unsupported accept type: {accept}")
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ ERROR in output_fn: {e}", exc_info=True)
        logger.error("=" * 60)
        raise


# ---------------------------------------------------------
# MODULE INITIALIZATION CHECK
# ---------------------------------------------------------
logger.info("=" * 60)
logger.info("✅ All handler functions defined successfully:")
logger.info(f"  - model_fn: {model_fn}")
logger.info(f"  - input_fn: {input_fn}")
logger.info(f"  - predict_fn: {predict_fn}")
logger.info(f"  - output_fn: {output_fn}")
logger.info("=" * 60)