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

# --- CẤU HÌNH CHO MÔ HÌNH CNN-LSTM (Conv2D) ---
LOOK_BACK = 288       # INPUT
TIME_STEPS = 4        # 
ROWS = 8              # 4 ma trân 8x9
COLS = 9              # 
FUTURE_STEPS = 12     # Dự đoán cho 60 phút sau


def create_sequences_for_conv2d(data, total_look_back, future_steps, n_steps, rows, cols):
    """
    Output X shape: (Samples, 4, 8, 9, 2)
    """
    X = []
    y = []
    
    # Duyệt qua dữ liệu 
    for i in range(len(data) - total_look_back - future_steps):
        # 1. Lấy 288 dòng 
        window = data[i : i + total_look_back]
        
        # 2. Reshape -> (4, 8, 9, 2)
        try:
            # -1 ở cuối để tự động tính Channels (ở đây là 2)
            reshaped_window = window.reshape(n_steps, rows, cols, -1)
            X.append(reshaped_window)
            
            # Label là giá trị car_count tại bước (hiện tại + look_back + tương lai)
            y.append(data[i + total_look_back + future_steps, 0]) 
        except ValueError as e:
            logging.error(f"❌ Lỗi reshape tại index {i}: {e}")
            raise e
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    # lưu tất cả (data + scaler) vào thư mục train để pipeline dễ xử lý
    parser.add_argument("--output-train-dir", type=str, default="/opt/ml/processing/train")
    
    args = parser.parse_args()

    logging.info("--- BẮT ĐẦU: PREPROCESSING ---")
    
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

    # 3. Lấy dữ liệu huấn luyện 
    df_train = df.copy()

    # 4. Scaling bằng MinMaxScaler để tạo ra file .pkl
    # Tạo feature hour
    df_train['hour'] = df_train['timestamp'].dt.hour
    
    # Khởi tạo Scaler
    scaler_car = MinMaxScaler(feature_range=(0, 1))
    scaler_hour = MinMaxScaler(feature_range=(0, 1))
    
    # Fit và Transform
    car_counts_scaled = scaler_car.fit_transform(df_train[['car_count']])
    hours_scaled = scaler_hour.fit_transform(df_train[['hour']])
    
    # Gộp lại thành mảng 2D: (Samples, 2)
    train_data_scaled = np.column_stack((car_counts_scaled, hours_scaled))
    
    # 5. Tạo Sequence 
    logging.info(f"🔄 Đang Reshape data sang 5D ({TIME_STEPS}x{ROWS}x{COLS})...")
    
    # Kiểm tra độ dài dữ liệu đủ cho Lookback + Future 
    if len(train_data_scaled) <= (LOOK_BACK + FUTURE_STEPS):
        raise ValueError(f"❌ Dữ liệu quá ít ({len(train_data_scaled)}) so với yêu cầu ({LOOK_BACK} + {FUTURE_STEPS}).")
        
    X_train, y_train = create_sequences_for_conv2d(
        train_data_scaled, 
        LOOK_BACK, 
        FUTURE_STEPS,
        TIME_STEPS, 
        ROWS, 
        COLS
    )
    
    # Kiểm tra kích thước output
    logging.info(f"📦 Kích thước tập Train: X={X_train.shape}, y={y_train.shape}")

    # 6. Lưu Data VÀ Scaler vào cùng thư mục Output
    os.makedirs(args.output_train_dir, exist_ok=True)
    
    # A. Lưu Data .npy
    output_path = os.path.join(args.output_train_dir, "train_data.npy")
    np.save(output_path, {'X': X_train, 'y': y_train})
    logging.info(f"💾 Đã lưu data vào: {output_path}")

    # B. Lưu Scaler .pkl 
    scaler_car_path = os.path.join(args.output_train_dir, "scaler_car_count.pkl")
    scaler_hour_path = os.path.join(args.output_train_dir, "scaler_hour.pkl")
    
    joblib.dump(scaler_car, scaler_car_path)
    joblib.dump(scaler_hour, scaler_hour_path)
    
    logging.info(f"💾 Đã lưu Scaler vào: {scaler_car_path} và {scaler_hour_path}")
    logging.info("--- HOÀN TẤT PREPROCESSING ---")