#!/usr/bin/env python3
import json
import os
import tarfile
import argparse
import logging
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- CẤU HÌNH ---
N_STEPS = 288           # Độ dài chuỗi đầu vào (24h)
TIME_STEP_MINUTES = 5   # Bước thời gian
FUTURE_STEP = 12        # Dự đoán cho 60 phút sau (12 bước * 5p)
CAR_MAX = 100.0         # Dùng để Inverse Scaling thủ công 

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

def preprocess_test_csv(df, scaler_car, scaler_hour):
    try:
        # 1. Parse Timestamp
        if 'timestamp' in df.columns:
            # File CSV thường lưu DD/MM/YYYY hoặc ISO. Thử dayfirst=True trước.
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.set_index('timestamp').sort_index()
        
        # 2. Resample 
        df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
        
        # Feature Engineering
        df_resampled['hour'] = df_resampled.index.hour
        
        # 3. Scaling 
        # Ép kiểu float để tránh lỗi
        df_resampled['car_count'] = df_resampled['car_count'].astype(float)
        df_resampled['hour'] = df_resampled['hour'].astype(float)

        df_resampled['car_count_scaled'] = scaler_car.transform(df_resampled[['car_count']])
        df_resampled['hour_scaled'] = scaler_hour.transform(df_resampled[['hour']])
        
        # 4. Tạo Sequence
        # Dữ liệu đầu vào cho model: [car_scaled, hour_scaled]
        data_matrix = df_resampled[['car_count_scaled', 'hour_scaled']].values
        # Dữ liệu gốc để so sánh (Ground Truth)
        raw_car_counts = df_resampled['car_count'].values
        
        X, y_true = [], []
        
        # Logic Sliding Window:
        # Input (X): từ i -> i + n_steps
        # Output (y): tại i + n_steps + future_step
        limit = len(data_matrix) - N_STEPS - FUTURE_STEP
        
        if limit <= 0:
            logger.error(f"Dữ liệu test quá ngắn ({len(data_matrix)} dòng). Cần tối thiểu {N_STEPS + FUTURE_STEP} dòng.")
            return None, None

        for i in range(limit):
            X.append(data_matrix[i : i + N_STEPS])
            y_true.append(raw_car_counts[i + N_STEPS + FUTURE_STEP]) 
            
        return np.array(X), np.array(y_true)

    except Exception as e:
        logger.error(f"Lỗi Preprocessing Test Data: {e}")
        return None, None

def evaluate_single_model(model_extract_dir, df_test_raw, model_name="Model"):
    """
    Load model + scaler, preprocess test data và đánh giá.
    """
    try:
        # 1. Load Scalers 
        scaler_car_path = os.path.join(model_extract_dir, "scaler_car_count.pkl")
        scaler_hour_path = os.path.join(model_extract_dir, "scaler_hour.pkl")
        
        if not os.path.exists(scaler_car_path) or not os.path.exists(scaler_hour_path):
            logger.error(f"[{model_name}] Thiếu file Scaler (.pkl) trong artifact model!")
            return None
            
        scaler_car = joblib.load(scaler_car_path)
        scaler_hour = joblib.load(scaler_hour_path)
        
        # 2. Preprocess Data (Tạo X_test, y_test ngay tại đây)
        logger.info(f"[{model_name}] Preprocessing test CSV...")
        X_test, y_true = preprocess_test_csv(df_test_raw.copy(), scaler_car, scaler_hour)
        
        if X_test is None: return None
        logger.info(f"[{model_name}] Test Set Size: {len(X_test)} mẫu")

        # 3. Load Keras Model
        model_path = os.path.join(model_extract_dir, "1")
        if not os.path.exists(model_path):
             logger.error(f"[{model_name}] Không tìm thấy thư mục model/1")
             return None
             
        logger.info(f"[{model_name}] Loading model...")
        model = tf.keras.models.load_model(model_path)

        # 4. Predict
        logger.info(f"[{model_name}] Predicting...")
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # 5. Inverse Transform (Đưa dự đoán về số xe thực tế)
        # Model output shape (N, 1) -> Inverse bằng scaler_car
        y_pred_actual = scaler_car.inverse_transform(y_pred_scaled)
        
        # Flatten về mảng 1 chiều để so sánh
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
    # Các tham số này được truyền từ build_pipeline.py
    parser.add_argument("--new-model-tar", type=str, default="/opt/ml/processing/new_model/model.tar.gz")
    parser.add_argument("--old-model-tar", type=str, default="/opt/ml/processing/old_model/model.tar.gz")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test/parking_test.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    
    args = parser.parse_args()

    # 1. Đọc File CSV Test Gốc
    logger.info(f"📂 Đang đọc file CSV Test từ: {args.test_data}")
    try:
        if os.path.isdir(args.test_data):
            csv_files = [f for f in os.listdir(args.test_data) if f.endswith('.csv')]
            if csv_files:
                test_file_path = os.path.join(args.test_data, csv_files[0])
            else:
                raise FileNotFoundError("Không tìm thấy file .csv trong thư mục test input")
        else:
            test_file_path = args.test_data

        df_test_raw = pd.read_csv(test_file_path)
        logger.info(f"📦 Raw Test Data: {len(df_test_raw)} dòng.")
    except Exception as e:
        logger.error(f"❌ Lỗi đọc file Test: {e}")
        exit(1)

    # 2. Đánh giá MODEL MỚI (Candidate)
    new_model_dir = "/tmp/new_model"
    os.makedirs(new_model_dir, exist_ok=True)
    
    metrics_new = None
    if extract_model_artifact(args.new_model_tar, new_model_dir):
        # Truyền df_test_raw vào, hàm sẽ tự preprocess bằng scaler CỦA MODEL MỚI
        metrics_new = evaluate_single_model(new_model_dir, df_test_raw, "NEW_MODEL")
    
    # 3. Đánh giá MODEL CŨ (Production)
    metrics_old = None
    old_model_dir = "/tmp/old_model"
    os.makedirs(old_model_dir, exist_ok=True)
    
    if os.path.exists(args.old_model_tar):
        if extract_model_artifact(args.old_model_tar, old_model_dir):
            # Truyền df_test_raw vào, hàm sẽ tự preprocess bằng scaler CỦA MODEL CŨ
            metrics_old = evaluate_single_model(old_model_dir, df_test_raw, "OLD_MODEL")
    else:
        logger.warning("⚠️ Không tìm thấy model cũ (Lần chạy đầu tiên?).")
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
            "mae": {
                "value": mae_new,
                "standard_deviation": 0.0
            },
            "mse": {
                "value": metrics_new["mse"] if metrics_new else 9999.0,
                "standard_deviation": 0.0
            }
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