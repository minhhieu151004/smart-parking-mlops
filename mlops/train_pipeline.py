import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import json
from datetime import datetime
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

def preprocess_data(df):
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
    car_count = df['car_count_scaled'].values
    hour = df['hour_scaled'].values
    X, y = [], []
    for i in range(len(car_count) - n_steps - future_step):
        X.append(np.column_stack((car_count[i:i + n_steps], hour[i:i + n_steps])))
        y.append(car_count[i + n_steps + future_step])
    return np.array(X), np.array(y)

def run_training(args):
    # Khởi tạo boto3 dùng Access Key từ tham số (args)
    # Vì script này chạy trên GHA, nó cần key để kết nối S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=args.aws_access_key,
        aws_secret_access_key=args.aws_secret_key
    )

    MODEL_VERSION = args.version
    if not MODEL_VERSION:
        MODEL_VERSION = datetime.now().strftime("%Y%m%d%H%M%S") # Fallback
    logging.info(f"--- BẮT ĐẦU HUẤN LUYỆN CHO PHIÊN BẢN: {MODEL_VERSION} ---")

    logging.info(f"Loading data from s3://{args.bucket}/{args.data_key}...")
    obj = s3.get_object(Bucket=args.bucket, Key=args.data_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    logging.info("Data loaded successfully.")

    df_processed, scaler_car, scaler_hour = preprocess_data(df)
    X, y = create_sequences(df_processed)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Data split. Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    logging.info("Model built.")
    logging.info(f"Starting model training for {args.epochs} epochs...")

    temp_model_file = 'best_model_temp.keras'
    checkpoint = ModelCheckpoint(temp_model_file, monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=2
    )
    logging.info("Training finished.")

    best_model = load_model(temp_model_file)
    val_loss = best_model.evaluate(X_val, y_val, verbose=0)
    logging.info(f"Validation Loss (MSE) của model tốt nhất: {val_loss}")

    metrics = {
        'val_loss': val_loss,
        'training_loss': history.history['loss'][-1],
        'epochs_run': len(history.history['loss']),
        'model_version': MODEL_VERSION,
        'training_timestamp': datetime.now().isoformat()
    }
    metrics_json = json.dumps(metrics, indent=4)

    BASE_MODEL_PATH = f'models/{MODEL_VERSION}'
    MODEL_KEY = f'{BASE_MODEL_PATH}/best_cnn_lstm_model.keras'
    SCALER_CAR_KEY = f'{BASE_MODEL_PATH}/scaler_car_count.pkl'
    SCALER_HOUR_KEY = f'{BASE_MODEL_PATH}/scaler_hour.pkl'
    METRICS_KEY = f'{BASE_MODEL_PATH}/metrics.json'

    # SỬA 3: Sửa log từ MinIO thành S3
    logging.info("Uploading artifacts to S3...")
    with open(temp_model_file, 'rb') as f:
        s3.put_object(Bucket=args.bucket, Key=MODEL_KEY, Body=f)
    logging.info(f"Artifact saved to s3://{args.bucket}/{MODEL_KEY}")

    for key, obj_to_save in [
        (SCALER_CAR_KEY, scaler_car),
        (SCALER_HOUR_KEY, scaler_hour)
    ]:
        buffer = BytesIO()
        joblib.dump(obj_to_save, buffer)
        s3.put_object(Bucket=args.bucket, Key=key, Body=buffer.getvalue())
        logging.info(f"Artifact saved to s3://{args.bucket}/{key}")

    s3.put_object(Bucket=args.bucket, Key=METRICS_KEY, Body=metrics_json.encode('utf-8'))
    logging.info(f"Metrics saved to s3://{args.bucket}/{METRICS_KEY}")

    os.remove(temp_model_file)
    logging.info(f"--- HUẤN LUYỆN PHIÊN BẢN {MODEL_VERSION} HOÀN TẤT ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training pipeline for parking prediction model.")

    parser.add_argument('--aws-access-key', default=os.getenv('AWS_ACCESS_KEY_ID'), required=True, help="AWS Access Key ID (from GitHub Secrets)")
    parser.add_argument('--aws-secret-key', default=os.getenv('AWS_SECRET_ACCESS_KEY'), required=True, help="AWS Secret Access Key (from GitHub Secrets)")
    
    parser.add_argument('--bucket', default=os.getenv('S3_BUCKET'), required=True, help="S3 Bucket name (from GitHub Secrets)")
    parser.add_argument('--data-key', default='parking_data/parking_data.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        help="Model version (e.g., from Airflow run_id via GHA client_payload)"
    )

    args = parser.parse_args()
    run_training(args)
