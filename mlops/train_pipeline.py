import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# --- THÊM MỚI (Mục 2) ---
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # Để tải model tốt nhất
import json
from datetime import datetime
# ----------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import boto3
from io import StringIO, BytesIO
import argparse
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_s3_client(endpoint, access_key, secret_key):
    return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

def preprocess_data(df):
    # (Giữ nguyên logic tiền xử lý của bạn)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df = df.resample('5min').mean().interpolate()
    df['car_count'] = df['car_count'].round().astype(int)
    df = df.ffill().bfill()
    df['hour'] = df.index.hour
    scaler_car_count = MinMaxScaler()
    df['car_count_scaled'] = scaler_car_count.fit_transform(df[['car_count']])
    scaler_hour = MinMaxScaler()
    df['hour_scaled'] = scaler_hour.fit_transform(df[['hour']])
    return df, scaler_car_count, scaler_hour

def create_sequences(df, n_steps=288, future_step=12):
    # (Giữ nguyên logic tạo chuỗi của bạn)
    car_count = df['car_count_scaled'].values
    hour = df['hour_scaled'].values
    X, y = [], []
    for i in range(len(car_count) - n_steps - future_step):
        X.append(np.column_stack((car_count[i:i + n_steps], hour[i:i + n_steps])))
        y.append(car_count[i + n_steps + future_step])
    return np.array(X), np.array(y)

def run_training(args):
    s3 = get_s3_client(args.minio_endpoint, args.minio_access_key, args.minio_secret_key)
    
    # --- (Mục 2) Lấy phiên bản từ tham số ---
    MODEL_VERSION = args.version
    if not MODEL_VERSION:
        MODEL_VERSION = datetime.now().strftime("%Y%m%d%H%M%S") # Fallback nếu không truyền
    logging.info(f"--- BẮT ĐẦU HUẤN LUYỆN CHO PHIÊN BẢN: {MODEL_VERSION} ---")

    logging.info(f"Loading data from s3://{args.bucket}/{args.data_key}...")
    obj = s3.get_object(Bucket=args.bucket, Key=args.data_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    logging.info("Data loaded successfully.")

    df_processed, scaler_car, scaler_hour = preprocess_data(df)
    X, y = create_sequences(df_processed)
    
    # --- (Mục 2) CẢI TIẾN: Chia train/validation set ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split. Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # (Giữ nguyên kiến trúc model của bạn)
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2]), padding='same'),
        BatchNormalization(), MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(pool_size=2),
        Reshape((-1, 128)),
        LSTM(units=150, return_sequences=True), Dropout(0.4),
        LSTM(units=100), Dropout(0.4),
        Dense(units=50, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') # Giữ nguyên MSE
    logging.info("Model built.")
    logging.info(f"Starting model training for {args.epochs} epochs...")

    # --- (Mục 2) CẢI TIẾN: Theo dõi 'val_loss' thay vì 'loss' ---
    # Điều này đảm bảo chúng ta chọn model tốt nhất trên dữ liệu *chưa thấy* (validation)
    temp_model_file = 'best_model_temp.keras'
    checkpoint = ModelCheckpoint(temp_model_file, monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        X_train, y_train, 
        epochs=args.epochs, 
        batch_size=64, 
        # Cung cấp validation data
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping], 
        verbose=2
    )
    logging.info("Training finished.")

    # --- (Mục 2) CẢI TIẾN: Đánh giá và lưu metrics ---
    # Tải model tốt nhất đã được checkpoint lưu
    best_model = load_model(temp_model_file)
    # Đánh giá trên tập validation
    val_loss = best_model.evaluate(X_val, y_val, verbose=0)
    logging.info(f"Validation Loss (MSE) của model tốt nhất: {val_loss}")

    # Tạo file metrics
    metrics = {
        'val_loss': val_loss,
        'training_loss': history.history['loss'][-1], # Lấy loss cuối
        'epochs_run': len(history.history['loss']),
        'model_version': MODEL_VERSION,
        'training_timestamp': datetime.now().isoformat()
    }
    metrics_json = json.dumps(metrics, indent=4)
    # ----------------------------------------------------

    # --- (Mục 2) CẢI TIẾN: Định nghĩa đường dẫn theo phiên bản ---
    # Tất cả artifacts giờ sẽ nằm trong thư mục của phiên bản đó
    BASE_MODEL_PATH = f'models/{MODEL_VERSION}'
    MODEL_KEY = f'{BASE_MODEL_PATH}/best_cnn_lstm_model.keras'
    SCALER_CAR_KEY = f'{BASE_MODEL_PATH}/scaler_car_count.pkl'
    SCALER_HOUR_KEY = f'{BASE_MODEL_PATH}/scaler_hour.pkl'
    METRICS_KEY = f'{BASE_MODEL_PATH}/metrics.json'
    # ----------------------------------------------------

    # Tải artifact lên MinIO
    logging.info("Uploading artifacts to MinIO...")
    # 1. Tải Model
    with open(temp_model_file, 'rb') as f:
        s3.put_object(Bucket=args.bucket, Key=MODEL_KEY, Body=f)
    logging.info(f"Artifact saved to s3://{args.bucket}/{MODEL_KEY}")

    # 2. Tải Scalers
    for key, obj_to_save in [
        (SCALER_CAR_KEY, scaler_car),
        (SCALER_HOUR_KEY, scaler_hour)
    ]:
        buffer = BytesIO()
        joblib.dump(obj_to_save, buffer)
        s3.put_object(Bucket=args.bucket, Key=key, Body=buffer.getvalue())
        logging.info(f"Artifact saved to s3://{args.bucket}/{key}")

    # 3. Tải Metrics
    s3.put_object(Bucket=args.bucket, Key=METRICS_KEY, Body=metrics_json.encode('utf-8'))
    logging.info(f"Metrics saved to s3://{args.bucket}/{METRICS_KEY}")
    
    os.remove(temp_model_file) # Xóa file model tạm
    logging.info(f"--- HUẤN LUYỆN PHIÊN BẢN {MODEL_VERSION} HOÀN TẤT ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training pipeline for parking prediction model.")
    parser.add_argument('--minio-endpoint', default=os.getenv('MINIO_ENDPOINT', 'http://minio:9000'))
    parser.add_argument('--minio-access-key', default=os.getenv('MINIO_ACCESS_KEY', 'admin'))
    parser.add_argument('--minio-secret-key', default=os.getenv('MINIO_SECRET_KEY', 'password'))
    parser.add_argument('--bucket', default=os.getenv('S3_BUCKET', 'my-bucket'))
    parser.add_argument('--data-key', default='parking_data/parking_data.csv')
    parser.add_argument('--epochs', type=int, default=50)

    # --- (Mục 2) CẢI TIẾN: Thay đổi tham số, chỉ cần --version ---
    parser.add_argument(
        '--version', 
        type=str, 
        required=True, 
        help="Phiên bản của mô hình (ví dụ: '{{ run_id }}' từ Airflow)"
    )
    # -----------------------------------------------------------
    
    args = parser.parse_args()
    run_training(args)