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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

N_STEPS = 288
TIME_STEP_MINUTES = 5

def extract_model_artifact(model_tar_path, extract_dir):
    """Giải nén model.tar.gz ra thư mục đích"""
    if not os.path.exists(model_tar_path):
        raise FileNotFoundError(f"Không tìm thấy file model tại: {model_tar_path}")
    
    logger.info(f"Đang giải nén {model_tar_path} vào {extract_dir}...")
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

def preprocess_test_data(df, scaler_car, scaler_hour, n_steps=N_STEPS):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df = df.set_index('timestamp').sort_index()
    
    # Resample để lấp đầy dữ liệu thiếu (quan trọng cho tập Test rời rạc)
    df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
    
    # Feature Engineering
    df_resampled['hour'] = df_resampled.index.hour
    
    # Scaling (Sử dụng scaler CỦA CHÍNH MODEL ĐÓ)
    df_resampled['car_count_scaled'] = scaler_car.transform(df_resampled[['car_count']])
    df_resampled['hour_scaled'] = scaler_hour.transform(df_resampled[['hour']])
    
    # Tạo Sequences (Sliding Window)
    data_matrix = df_resampled[['car_count_scaled', 'hour_scaled']].values
    
    X, y = [], []
    # Lấy chuỗi quá khứ để dự đoán điểm tương lai (dựa vào FUTURE_STEP=12 trong train)
    # Lưu ý: Logic này phải khớp logic train. Giả sử train dùng future_step=12
    future_step = 12 
    
    for i in range(len(data_matrix) - n_steps - future_step):
        X.append(data_matrix[i : i + n_steps])
        y.append(df_resampled['car_count'].values[i + n_steps + future_step]) # Lấy giá trị thực (chưa scale) để so sánh MAE gốc
        
    return np.array(X), np.array(y)

def evaluate_single_model(model_extract_dir, df_test, model_name="Model"):
    """Load model, scaler và đánh giá trên tập test"""
    try:
        # 1. Load Model (SavedModel trong thư mục '1')
        model_path = os.path.join(model_extract_dir, "1")
        logger.info(f"[{model_name}] Loading model từ {model_path}...")
        model = tf.keras.models.load_model(model_path)

        # 2. Load Scalers (nằm cùng thư mục giải nén)
        scaler_car = joblib.load(os.path.join(model_extract_dir, "scaler_car_count.pkl"))
        scaler_hour = joblib.load(os.path.join(model_extract_dir, "scaler_hour.pkl"))
        
        # 3. Preprocess Test Data
        logger.info(f"[{model_name}] Preprocessing test data...")
        X_test, y_true = preprocess_test_data(df_test.copy(), scaler_car, scaler_hour)
        
        if len(X_test) == 0:
            logger.warning(f"[{model_name}] Tập test quá ngắn để tạo sequence!")
            return None

        # 4. Predict
        logger.info(f"[{model_name}] Predicting...")
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # 5. Inverse Transform (Đưa về số lượng xe thực tế)
        # Model trả về shape (N, 1), cần inverse
        y_pred_actual = scaler_car.inverse_transform(y_pred_scaled)
        
        # 6. Tính Metrics
        mae = mean_absolute_error(y_true, y_pred_actual)
        mse = mean_squared_error(y_true, y_pred_actual)
        
        logger.info(f"[{model_name}] Result -> MAE: {mae:.4f}, MSE: {mse:.4f}")
        return {"mae": mae, "mse": mse}

    except Exception as e:
        logger.error(f"Lỗi khi đánh giá {model_name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-model-tar", type=str, default="/opt/ml/processing/new_model/model.tar.gz")
    parser.add_argument("--old-model-tar", type=str, default="/opt/ml/processing/old_model/model.tar.gz")
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test/updated_test_set.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    # 1. Load Test Data
    logger.info(f"Đọc tập test từ: {args.test_data}")
    df_test = pd.read_csv(args.test_data)
    
    # 2. Đánh giá MODEL MỚI (Candidate)
    new_model_dir = "/tmp/new_model"
    extract_model_artifact(args.new_model_tar, new_model_dir)
    metrics_new = evaluate_single_model(new_model_dir, df_test, "NEW_MODEL")
    
    # 3. Đánh giá MODEL CŨ (Baseline)
    metrics_old = None
    if os.path.exists(args.old_model_tar):
        old_model_dir = "/tmp/old_model"
        extract_model_artifact(args.old_model_tar, old_model_dir)
        metrics_old = evaluate_single_model(old_model_dir, df_test, "OLD_MODEL")
    else:
        logger.warning("Không tìm thấy model cũ (Lần chạy đầu tiên?).")

    # 4. So sánh và Tạo Báo cáo
    report = {
        "regression_metrics": {
            "mae": {"value": metrics_new["mae"] if metrics_new else 9999.0, "standard_deviation": 0},
            "mae_baseline": {"value": metrics_old["mae"] if metrics_old else 9999.0}
        },
        "evaluation_timestamp": datetime.now().isoformat()
    }
    
    # Logic so sánh
    new_mae = metrics_new["mae"] if metrics_new else float('inf')
    old_mae = metrics_old["mae"] if metrics_old else float('inf')
    
    if new_mae < old_mae:
        report["metadata"] = {"comparison_result": "BETTER"}
        logger.info(">>> KẾT QUẢ: Model MỚI tốt hơn!")
    else:
        report["metadata"] = {"comparison_result": "WORSE"}
        logger.info(">>> KẾT QUẢ: Model MỚI tệ hơn (hoặc bằng).")

    # 5. Lưu file JSON
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Đã lưu báo cáo đánh giá tại: {output_path}")