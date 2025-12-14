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

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- CẤU HÌNH ---
N_STEPS = 288           # Độ dài chuỗi đầu vào (24h)
TIME_STEP_MINUTES = 5   # Bước thời gian
FUTURE_STEP = 12        # Dự đoán cho 60 phút sau
OLD_MODEL_DIR = "/tmp/old_model" # Thư mục tạm để tải model cũ

def extract_model_artifact(model_tar_path, extract_dir):
    """Giải nén model.tar.gz ra thư mục đích"""
    if not os.path.exists(model_tar_path):
        logger.warning(f"⚠️ Không tìm thấy file model tại: {model_tar_path}")
        return False
    
    logger.info(f"Đang giải nén {model_tar_path} vào {extract_dir}...")
    try:
        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        return True
    except Exception as e:
        logger.error(f"Lỗi giải nén: {e}")
        return False

def get_latest_approved_model_artifact(sm_client, model_package_group_name):
    """Tìm và trả về S3 URI của model Approved mới nhất từ Registry."""
    try:
        # Lấy danh sách model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        
        packages = response.get('ModelPackageSummaryList', [])
        if not packages:
            return None
            
        # Lấy cái mới nhất
        latest_pkg = packages[0]
        pkg_arn = latest_pkg['ModelPackageArn']
        
        # Lấy chi tiết để tìm S3 URI
        details = sm_client.describe_model_package(ModelPackageName=pkg_arn)
        s3_uri = details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        
        logger.info(f"🔎 Tìm thấy model Approved mới nhất: {pkg_arn}")
        return s3_uri
        
    except Exception as e:
        logger.warning(f"⚠️ Không thể lấy model cũ từ Registry: {e}")
        return None

def download_from_s3(s3_uri, local_path, region):
    """Tải file từ S3 về local."""
    try:
        s3 = boto3.client('s3', region_name=region)
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        
        logger.info(f"⬇️ Downloading {s3_uri}...")
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi download S3: {e}")
        return False

def preprocess_test_csv(df, scaler_car, scaler_hour):
    """Tiền xử lý dữ liệu test giống như lúc training."""
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
            logger.error(f"Dữ liệu test quá ngắn. Cần tối thiểu {N_STEPS + FUTURE_STEP} dòng.")
            return None, None

        for i in range(limit):
            X.append(data_matrix[i : i + N_STEPS])
            y_true.append(raw_car_counts[i + N_STEPS + FUTURE_STEP]) 
            
        return np.array(X), np.array(y_true)

    except Exception as e:
        logger.error(f"Lỗi Preprocessing Test Data: {e}")
        return None, None

def evaluate_single_model(model_extract_dir, df_test_raw, model_name="Model"):
    """Load model + scaler, preprocess test data và đánh giá."""
    try:
        # 1. Load Scalers (Tìm đệ quy)
        found_scalers = glob.glob(os.path.join(model_extract_dir, "**", "*.pkl"), recursive=True)
        scaler_car = None
        scaler_hour = None
        
        for f in found_scalers:
            if "scaler_car_count" in os.path.basename(f):
                scaler_car = joblib.load(f)
            elif "scaler_hour" in os.path.basename(f):
                scaler_hour = joblib.load(f)
        
        if not scaler_car or not scaler_hour:
            logger.error(f"[{model_name}] Thiếu file Scaler (.pkl) trong artifact model!")
            return None
            
        # 2. Preprocess Data
        logger.info(f"[{model_name}] Preprocessing test CSV...")
        X_test, y_true = preprocess_test_csv(df_test_raw.copy(), scaler_car, scaler_hour)
        
        if X_test is None: return None
        logger.info(f"[{model_name}] Test Set Size: {len(X_test)} mẫu")

        # 3. Load Keras Model (Tìm folder chứa saved_model.pb)
        # Thường là model_dir/1/saved_model.pb hoặc model_dir/saved_model.pb
        subfolders = [f.path for f in os.scandir(model_extract_dir) if f.is_dir() and f.name.isdigit()]
        if subfolders:
            model_path = subfolders[0] # Lấy folder '1'
        else:
            model_path = model_extract_dir # Lấy thư mục gốc

        logger.info(f"[{model_name}] Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)

        # 4. Predict
        logger.info(f"[{model_name}] Predicting...")
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # 5. Inverse Transform
        y_pred_actual = scaler_car.inverse_transform(y_pred_scaled)
        
        y_pred_actual = y_pred_actual.flatten()
        y_true = y_true.flatten()
        
        # 6. Tính Metrics
        mae = mean_absolute_error(y_true, y_pred_actual)
        mse = mean_squared_error(y_true, y_pred_actual)
        
        logger.info(f"[{model_name}] Result -> MAE: {mae:.4f} xe, MSE: {mse:.4f}")
        return {"mae": mae, "mse": mse}

    except Exception as e:
        logger.error(f"Lỗi khi đánh giá {model_name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Tham số từ Pipeline
    parser.add_argument("--new-model-tar", type=str, default="/opt/ml/processing/new_model/model.tar.gz")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test/parking_test.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    # Tham số thêm để tự tải model cũ
    parser.add_argument("--model-package-group-name", type=str, required=False)
    parser.add_argument("--region", type=str, default="ap-southeast-1")
    
    args, _ = parser.parse_known_args()

    # 1. Đọc File CSV Test Gốc
    logger.info(f"📂 Đang đọc file CSV Test từ: {args.test_data}")
    try:
        if os.path.isdir(args.test_data):
            csv_files = glob.glob(os.path.join(args.test_data, "*.csv"))
            if csv_files:
                test_file_path = csv_files[0]
            else:
                raise FileNotFoundError("Không tìm thấy file .csv trong thư mục test input")
        else:
            test_file_path = args.test_data

        df_test_raw = pd.read_csv(test_file_path)
        logger.info(f"📦 Raw Test Data: {len(df_test_raw)} dòng.")
    except Exception as e:
        logger.error(f"❌ Lỗi đọc file Test: {e}")
        exit(1)

    # 2. Đánh giá MODEL MỚI 
    new_model_dir = "/tmp/new_model"
    os.makedirs(new_model_dir, exist_ok=True)
    
    metrics_new = None
    if extract_model_artifact(args.new_model_tar, new_model_dir):
        metrics_new = evaluate_single_model(new_model_dir, df_test_raw, "NEW_MODEL")
    
    # 3. Đánh giá MODEL CŨ 
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
            logger.warning("⚠️ Không tìm thấy model Approved nào trong Registry (Lần chạy đầu tiên?).")
    else:
        logger.warning("⚠️ Không nhận được tham số Model Package Group Name. Bỏ qua so sánh.")

    # Xử lý trường hợp không có metrics cũ
    if metrics_old is None:
        metrics_old = {"mae": 9999.0, "mse": 9999.0}

    # 4. So sánh và Tạo Báo cáo
    mae_new = metrics_new["mae"] if metrics_new else 9999.0
    mae_old = metrics_old["mae"] if metrics_old else 9999.0
    
    comparison_result = "UNKNOWN"
    if mae_new < mae_old:
        comparison_result = "BETTER"
        logger.info(f"🚀 Model Mới TỐT HƠN ({mae_new:.2f} vs {mae_old:.2f})")
    else:
        comparison_result = "WORSE"
        logger.info(f"📉 Model Mới TỆ HƠN ({mae_new:.2f} vs {mae_old:.2f})")

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

    # 5. Lưu Output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation.json")
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"✅ Đã lưu báo cáo đánh giá vào: {output_path}")