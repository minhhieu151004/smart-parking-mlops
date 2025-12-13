import argparse
import os
import pandas as pd
import numpy as np
import logging
import glob

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CẤU HÌNH ---
CAR_MAX = 100.0
HOUR_MAX = 24.0
N_STEPS = 288  # Số bước thời gian (5 phút) trong 24 giờ

def create_sequences(data, n_steps):
    """Chuyển đổi dữ liệu 2D thành 3D sequences cho LSTM/GRU."""
    X = []
    y = []
    # Dữ liệu: [car_count_scaled, hour_scaled]
    for i in range(len(data) - n_steps):
        # Input: n_steps quá khứ
        X.append(data[i : i + n_steps])
        # Output: Bước tiếp theo (chỉ dự đoán car_count - cột index 0)
        y.append(data[i + n_steps, 0]) 
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SageMaker sẽ mount dữ liệu vào các đường dẫn này
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-train-dir", type=str, default="/opt/ml/processing/train")
    parser.add_argument("--output-scaler-dir", type=str, default="/opt/ml/processing/scaler")
    
    args = parser.parse_args()

    logging.info("--- BẮT ĐẦU: PREPROCESSING (LẤY 70% DỮ LIỆU CUỐI) ---")
    
    # 1. Đọc Master File từ Input
    # Master file (parking_data.csv) được pipeline truyền vào /opt/ml/processing/input
    all_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"❌ Không tìm thấy file CSV nào trong {args.input_dir}")
    
    input_file = all_files[0] # Lấy file đầu tiên tìm thấy
    logging.info(f"📂 Đang đọc file: {input_file}")
    
    # Đọc CSV (Master file format: DD/MM/YYYY)
    df = pd.read_csv(input_file)
    
    # 2. Xử lý thời gian và Sắp xếp
    if 'timestamp' in df.columns:
        # dayfirst=True cực kỳ quan trọng với format DD/MM/YYYY
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        raise ValueError("❌ File dữ liệu thiếu cột 'timestamp'")
        
    logging.info(f"📊 Tổng dữ liệu gốc: {len(df)} dòng. Từ {df['timestamp'].min()} đến {df['timestamp'].max()}")

    # 3. LẤY 70% DỮ LIỆU CUỐI CÙNG (MỚI NHẤT)
    train_ratio = 0.7
    total_rows = len(df)
    
    # Tính index cắt: Lấy từ dòng thứ (1 - 0.7) * total trở đi
    cut_index = int(total_rows * (1 - train_ratio))
    
    # Thực hiện cắt
    df_train = df.iloc[cut_index:].copy()
    
    logging.info(f"✂️ Đã cắt lấy 70% dữ liệu cuối: {len(df_train)} dòng.")
    logging.info(f"   -> Dữ liệu train từ: {df_train['timestamp'].min()} đến {df_train['timestamp'].max()}")

    # 4. Feature Engineering & Scaling (Thủ công)
    # Lý do Scaling thủ công: Để khớp 100% với logic Inference trên Lambda/Pi
    df_train['hour'] = df_train['timestamp'].dt.hour
    
    # Ép kiểu float
    car_counts = df_train['car_count'].values.astype(float)
    hours = df_train['hour'].values.astype(float)
    
    # Scale
    car_counts_scaled = car_counts / CAR_MAX
    hours_scaled = hours / HOUR_MAX
    
    # Gộp lại thành mảng 2D: [rows, 2] -> cột 0: car, cột 1: hour
    train_data_scaled = np.column_stack((car_counts_scaled, hours_scaled))
    
    # 5. Tạo Sequence (Sliding Window)
    if len(train_data_scaled) <= N_STEPS:
        raise ValueError(f"❌ Dữ liệu sau khi cắt ({len(train_data_scaled)}) ít hơn N_STEPS ({N_STEPS}). Không thể tạo sequence.")
        
    X_train, y_train = create_sequences(train_data_scaled, N_STEPS)
    
    logging.info(f"📦 Kích thước tập Train: X={X_train.shape}, y={y_train.shape}")

    # 6. Lưu file .npy để bước Train sử dụng
    os.makedirs(args.output_train_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_train_dir, "train_data.npy")
    # Lưu cả X và y vào 1 file cho gọn, hoặc 2 file tuỳ ý. Ở đây lưu 1 file dictionary hoặc array
    # Để đơn giản cho train_pipeline.py đọc, ta lưu dictionary
    np.save(output_path, {'X': X_train, 'y': y_train})
    
    logging.info(f"💾 Đã lưu file processed vào: {output_path}")
    logging.info("--- HOÀN TẤT PREPROCESSING ---")