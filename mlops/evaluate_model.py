#!/usr/bin/env python3
import argparse
import os
import json
import logging
import tarfile
import glob
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import boto3
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- CONFIGURATION ---
N_STEPS = 288           # Input
TIME_STEP_MINUTES = 5   # Time step
FUTURE_STEP = 12        # dự đoán 60' sau
OLD_MODEL_DIR = "/tmp/old_model" 

# --- CONFIG  ---
TIME_STEPS = 4
ROWS = 8
COLS = 9

def extract_model_artifact(model_tar_path, extract_dir):
    """Extract model.tar.gz to destination directory"""
    if not os.path.exists(model_tar_path):
        logger.warning(f"⚠️ Model file not found at: {model_tar_path}")
        return False
    
    logger.info(f"Extracting {model_tar_path} into {extract_dir}...")
    try:
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        return True
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return False

def get_latest_approved_model_artifact(sm_client, model_package_group_name):
    """Tìm URI của model mới nhất đang Approved (đang được triển khai trên Endpoint)"""

    try:
        # Get list of model packages
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        packages = response.get('ModelPackageSummaryList', [])
        if not packages:
            return None
            
        # Get the latest one
        latest_pkg = packages[0]
        pkg_arn = latest_pkg['ModelPackageArn']
        
        # Get details to find S3 URI
        details = sm_client.describe_model_package(ModelPackageName=pkg_arn)
        s3_uri = details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        
        logger.info(f"🔎 Found latest Approved model: {pkg_arn}")
        return s3_uri
        
    except Exception as e:
        logger.warning(f"⚠️ Cannot get old model from Registry: {e}")
        return None

def download_from_s3(s3_uri, local_path, region):
    """Download file from S3 to local."""
    try:
        s3 = boto3.client('s3', region_name=region)
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        
        logger.info(f"⬇️ Downloading {s3_uri}...")
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        logger.error(f"❌ S3 Download Error: {e}")
        return False

def preprocess_test_csv(df, scaler_car, scaler_hour):
    try:
        # 1. Parse Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.set_index('timestamp').sort_index()
        
        # 2. Resample 
        df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
        
        # Feature Engineering
        df_resampled['hour'] = df_resampled.index.hour
        
        # 3. Scaling 
        df_resampled['car_count'] = df_resampled['car_count'].astype(float)
        df_resampled['hour'] = df_resampled['hour'].astype(float)

        df_resampled['car_count_scaled'] = scaler_car.transform(df_resampled[['car_count']])
        df_resampled['hour_scaled'] = scaler_hour.transform(df_resampled[['hour']])
        
        # 4. Tạo Sequence
        data_matrix = df_resampled[['car_count_scaled', 'hour_scaled']].values
        raw_car_counts = df_resampled['car_count'].values
        
        X, y_true = [], []
        limit = len(data_matrix) - N_STEPS - FUTURE_STEP
        
        if limit <= 0:
            logger.error(f"Test data too short. Need at least {N_STEPS + FUTURE_STEP} rows.")
            return None, None

        for i in range(limit):
            X.append(data_matrix[i : i + N_STEPS])
            y_true.append(raw_car_counts[i + N_STEPS + FUTURE_STEP])
            
        X = np.array(X)          # (Samples, 288, 2)
        y_true = np.array(y_true)

        # 5. --- RESHAPE ---
        # (Samples, 4, 8, 9, 2)
        try:
            X_reshaped = X.reshape(X.shape[0], TIME_STEPS, ROWS, COLS, 2)
            logger.info(f" ✅ Reshaped X_test to: {X_reshaped.shape}")
            return X_reshaped, y_true
        except Exception as e:
            logger.error(f"Reshape Test Data Error: {e}")
            return None, None

    except Exception as e:
        logger.error(f"Preprocessing Test Data Error: {e}")
        return None, None

def evaluate_single_model(model_extract_dir, df_test_raw, model_name="Model"):
    """Load model + scaler, preprocess test data, and evaluate."""
    try:
        # 1. Load Scalers 
        found_scalers = glob.glob(os.path.join(model_extract_dir, "**", "*.pkl"), recursive=True)
        scaler_car = None
        scaler_hour = None
        
        for f in found_scalers:
            if "scaler_car_count" in os.path.basename(f):
                scaler_car = joblib.load(f)
            elif "scaler_hour" in os.path.basename(f):
                scaler_hour = joblib.load(f)
        
        if not scaler_car or not scaler_hour:
            logger.error(f"[{model_name}] Missing Scaler (.pkl) files in model artifact!")
            return None
            
        # 2. Preprocess Data
        logger.info(f"[{model_name}] Preprocessing test CSV...")
        X_test, y_true = preprocess_test_csv(df_test_raw.copy(), scaler_car, scaler_hour)
        
        if X_test is None: return None
        logger.info(f"[{model_name}] Test Set Size: {len(X_test)} samples")

        # 3. Load Keras Model 
        #  model_dir/1/saved_model.pb or model_dir/saved_model.pb
        subfolders = [f.path for f in os.scandir(model_extract_dir) if f.is_dir() and f.name.isdigit()]
        if subfolders:
            model_path = subfolders[0] # Take folder '1'
        else:
            model_path = model_extract_dir # Take root folder

        logger.info(f"[{model_name}] Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)

        # 4. Predict
        logger.info(f"[{model_name}] Predicting...")
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # 5. Inverse Transform
        y_pred_actual = scaler_car.inverse_transform(y_pred_scaled)
        
        y_pred_actual = y_pred_actual.flatten()
        y_true = y_true.flatten()
        
        # 6. Tính toán metric
        mae = mean_absolute_error(y_true, y_pred_actual)
        mse = mean_squared_error(y_true, y_pred_actual)
        
        logger.info(f"[{model_name}] Result -> MAE: {mae:.4f} cars, MSE: {mse:.4f}")
        return {"mae": mae, "mse": mse}

    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parameters from Pipeline
    parser.add_argument("--new-model-tar", type=str, default="/opt/ml/processing/new_model/model.tar.gz")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test/parking_test.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    # Extra parameters to auto-download old model
    parser.add_argument("--model-package-group-name", type=str, required=False)
    parser.add_argument("--region", type=str, default="ap-southeast-1")
    
    args, _ = parser.parse_known_args()

    # 1. Lấy dữ liệu test
    logger.info(f"📂 Reading Test CSV from: {args.test_data}")
    try:
        if os.path.isdir(args.test_data):
            csv_files = glob.glob(os.path.join(args.test_data, "*.csv"))
            if csv_files:
                test_file_path = csv_files[0]
            else:
                raise FileNotFoundError("No .csv file found in test input directory")
        else:
            test_file_path = args.test_data

        df_test_raw = pd.read_csv(test_file_path)
        logger.info(f"📦 Raw Test Data: {len(df_test_raw)} rows.")
    except Exception as e:
        logger.error(f"❌ Error reading Test file: {e}")
        exit(1)

    # 2. Evaluate NEW MODEL 
    new_model_dir = "/tmp/new_model"
    os.makedirs(new_model_dir, exist_ok=True)
    
    metrics_new = None
    if extract_model_artifact(args.new_model_tar, new_model_dir):
        metrics_new = evaluate_single_model(new_model_dir, df_test_raw, "NEW_MODEL")
    
    # 3. Evaluate OLD MODEL 
    metrics_old = None
    os.makedirs(OLD_MODEL_DIR, exist_ok=True)
    
    if args.model_package_group_name:
        sm_client = boto3.client('sagemaker', region_name=args.region)
        old_model_uri = get_latest_approved_model_artifact(sm_client, args.model_package_group_name)
        
        if old_model_uri:
            local_old_tar = os.path.join(OLD_MODEL_DIR, "model.tar.gz")
            if download_from_s3(old_model_uri, local_old_tar, args.region):
                if extract_model_artifact(local_old_tar, OLD_MODEL_DIR):
                    metrics_old = evaluate_single_model(OLD_MODEL_DIR, df_test_raw, "OLD_MODEL")
        else:
            logger.warning("⚠️ No Approved model found in Registry (First run?).")
    else:
        logger.warning("⚠️ Model Package Group Name not provided. Skipping comparison.")

    if metrics_old is None:
        metrics_old = {"mae": 9999.0, "mse": 9999.0}

    # 4. So sánh
    mae_new = metrics_new["mae"] if metrics_new else 9999.0
    mae_old = metrics_old["mae"] if metrics_old else 9999.0
    
    comparison_result = "UNKNOWN"
    if mae_new < mae_old:
        comparison_result = "BETTER"
        logger.info(f"🚀 New Model is BETTER ({mae_new:.2f} vs {mae_old:.2f})")
    else:
        comparison_result = "WORSE"
        logger.info(f"📉 New Model is WORSE ({mae_new:.2f} vs {mae_old:.2f})")

    # JSON Report
    report = {
        "regression_metrics": {
            "mae": {"value": mae_new, "standard_deviation": 0.0},
            "mse": {"value": metrics_new["mse"] if metrics_new else 9999.0, "standard_deviation": 0.0}
        },
        "comparison": {
            "new_mae": mae_new,
            "old_mae": mae_old,
            "result": comparison_result
        }
    }

    # 5. Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation.json")
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"✅ Evaluation report saved to: {output_path}")