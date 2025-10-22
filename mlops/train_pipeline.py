import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
    s3 = get_s3_client(args.minio_endpoint, args.minio_access_key, args.minio_secret_key)

    logging.info(f"Loading data from s3://{args.bucket}/{args.data_key}...")
    obj = s3.get_object(Bucket=args.bucket, Key=args.data_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    logging.info("Data loaded successfully.")

    df_processed, scaler_car, scaler_hour = preprocess_data(df)
    X, y = create_sequences(df_processed)
    logging.info(f"Data preprocessed. Shape X: {X.shape}, y: {y.shape}")

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
    checkpoint = ModelCheckpoint('best_model_temp.keras', monitor='loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.fit(X, y, epochs=args.epochs, batch_size=64, callbacks=[checkpoint, early_stopping], verbose=2)
    logging.info("Training finished.")

    # Tải artifact lên MinIO
    for key, obj_to_save in [
        (args.model_key, 'best_model_temp.keras'),
        (args.scaler_car_key, scaler_car),
        (args.scaler_hour_key, scaler_hour)
    ]:
        if isinstance(obj_to_save, str): # Model file
            with open(obj_to_save, 'rb') as f:
                s3.put_object(Bucket=args.bucket, Key=key, Body=f)
        else: # Scaler object
            buffer = BytesIO()
            joblib.dump(obj_to_save, buffer)
            s3.put_object(Bucket=args.bucket, Key=key, Body=buffer.getvalue())
        logging.info(f"Artifact saved to s3://{args.bucket}/{key}")
    
    os.remove('best_model_temp.keras')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training pipeline for parking prediction model.")
    parser.add_argument('--minio-endpoint', default=os.getenv('MINIO_ENDPOINT', 'http://minio:9000'))
    parser.add_argument('--minio-access-key', default=os.getenv('MINIO_ACCESS_KEY', 'admin'))
    parser.add_argument('--minio-secret-key', default=os.getenv('MINIO_SECRET_KEY', 'password'))
    parser.add_argument('--bucket', default=os.getenv('S3_BUCKET', 'my-bucket'))
    parser.add_argument('--data-key', default='parking_data/parking_data.csv')
    parser.add_argument('--model-key', default='models/best_cnn_lstm_model.keras')
    parser.add_argument('--scaler-car-key', default='models/scaler_car_count.pkl')
    parser.add_argument('--scaler-hour-key', default='models/scaler_hour.pkl')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    run_training(args)

