import argparse
import os
import pandas as pd
import numpy as np
import logging
import glob
import joblib 
from sklearn.preprocessing import MinMaxScaler 

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CẤU HÌNH ---
N_STEPS = 288  # Số bước thời gian (5 phút) trong 24 giờ

def create_sequences(data, n_steps):
    """Chuyển đổi dữ liệu 2D thành 3D sequences cho LSTM/GRU."""
    X = []
    y = []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps, 0]) 
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    # Lưu ý: Ta sẽ lưu tất cả (data + scaler) vào thư mục train này để pipeline dễ xử lý
    parser.add_argument("--output-train-dir", type=str, default="/opt/ml/processing/train")
    
    args = parser.parse_args()

    logging.info("--- BẮT ĐẦU: PREPROCESSING (LẤY 70% DỮ LIỆU CUỐI) ---")
    
    # 1. Đọc Master File
    all_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"❌ Không tìm thấy file CSV nào trong {args.input_dir}")
    
    input_file = all_files[0]
    logging.info(f"📂 Đang đọc file: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # 2. Xử lý thời gian
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        raise ValueError("❌ File dữ liệu thiếu cột 'timestamp'")
        
    logging.info(f"📊 Tổng dữ liệu gốc: {len(df)} dòng.")

    # 3. LẤY 70% DỮ LIỆU CUỐI CÙNG
    train_ratio = 0.7
    cut_index = int(len(df) * (1 - train_ratio))
    df_train = df.iloc[cut_index:].copy()
    
    logging.info(f"✂️ Đã cắt lấy 70% dữ liệu cuối: {len(df_train)} dòng.")

    # 4. Scaling bằng MinMaxScaler để tạo ra file .pkl
    # Tạo feature hour
    df_train['hour'] = df_train['timestamp'].dt.hour
    
    # Khởi tạo Scaler
    scaler_car = MinMaxScaler(feature_range=(0, 1))
    scaler_hour = MinMaxScaler(feature_range=(0, 1))
    
    # Fit và Transform
    car_counts_scaled = scaler_car.fit_transform(df_train[['car_count']])
    hours_scaled = scaler_hour.fit_transform(df_train[['hour']])
    
    # Gộp lại thành mảng 2D
    train_data_scaled = np.column_stack((car_counts_scaled, hours_scaled))
    
    # 5. Tạo Sequence
    if len(train_data_scaled) <= N_STEPS:
        raise ValueError(f"❌ Dữ liệu quá ít ({len(train_data_scaled)}) so với N_STEPS ({N_STEPS}).")
        
    X_train, y_train = create_sequences(train_data_scaled, N_STEPS)
    logging.info(f"📦 Kích thước tập Train: X={X_train.shape}, y={y_train.shape}")

    # 6. Lưu Data VÀ Scaler vào cùng thư mục Output
    os.makedirs(args.output_train_dir, exist_ok=True)
    
    # A. Lưu Data .npy
    output_path = os.path.join(args.output_train_dir, "train_data.npy")
    np.save(output_path, {'X': X_train, 'y': y_train})
    logging.info(f"💾 Đã lưu data vào: {output_path}")

    # B. Lưu Scaler .pkl (Để bước Train copy đi, và bước Evaluate dùng lại)
    scaler_car_path = os.path.join(args.output_train_dir, "scaler_car_count.pkl")
    scaler_hour_path = os.path.join(args.output_train_dir, "scaler_hour.pkl")
    
    joblib.dump(scaler_car, scaler_car_path)
    joblib.dump(scaler_hour, scaler_hour_path)
    
    logging.info(f"💾 Đã lưu Scaler vào: {scaler_car_path} và {scaler_hour_path}")
    logging.info("--- HOÀN TẤT PREPROCESSING ---")