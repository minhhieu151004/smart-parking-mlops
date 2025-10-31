import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import boto3
from io import StringIO, BytesIO
import argparse
import os
import logging
import json
from datetime import datetime
import sys

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Các hàm preprocess_data và create_sequences ---
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

# --- Hàm chính ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Đọc Hyperparameters từ SageMaker ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--n-steps', type=int, default=288)
    parser.add_argument('--future-step', type=int, default=12)
    parser.add_argument('--model-version', type=str, required=True)

    # --- Đọc đường dẫn dữ liệu và model từ biến môi trường SageMaker ---
    # Kênh dữ liệu input (SageMaker tự động tải S3 về đây)
    input_data_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    # Thư mục output model
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    # Thư mục output khác 
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data') # Nơi lưu metrics.json

    args = parser.parse_args()

    logging.info(f"--- BẮT ĐẦU HUẤN LUYỆN SAGEMAKER CHO PHIÊN BẢN: {args.model_version} ---")
    logging.info(f"Hyperparameters: epochs={args.epochs}, lr={args.learning_rate}")
    logging.info(f"Input data path: {input_data_path}")
    logging.info(f"Model output path: {model_dir}")
    logging.info(f"Metrics output path: {output_dir}")

    try:
        data_file_path = os.path.join(input_data_path, 'parking_data.csv') 
        if not os.path.exists(data_file_path):
             data_file_path = input_data_path 
        
        logging.info(f"Loading data from {data_file_path}...")
        # Tìm file csv trong thư mục input
        csv_files = [f for f in os.listdir(input_data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file .csv nào trong {input_data_path}")
        
        # Lấy file csv đầu tiên tìm thấy
        data_file_path = os.path.join(input_data_path, csv_files[0])
        logging.info(f"Reading data from {data_file_path}")
        df = pd.read_csv(data_file_path)
        logging.info("Data loaded successfully.")

        df_processed, scaler_car, scaler_hour = preprocess_data(df)
        X, y = create_sequences(df_processed, n_steps=args.n_steps, future_step=args.future_step)
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
        model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='mean_squared_error')
        logging.info("Model built.")
        logging.info(f"Starting model training for {args.epochs} epochs...")

        # Lưu model tạm thời trong container
        temp_model_file = 'best_model_temp.keras'
        checkpoint = ModelCheckpoint(temp_model_file, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(
            X_train, y_train, epochs=args.epochs, batch_size=64,
            validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping], verbose=2
        )
        logging.info("Training finished.")

        best_model = load_model(temp_model_file)
        val_loss = best_model.evaluate(X_val, y_val, verbose=0)
        logging.info(f"Validation Loss (MSE) của model tốt nhất: {val_loss}")

        metrics = {
            'val_loss': val_loss,
            'training_loss': history.history['loss'][-1], 
            'epochs_run': len(history.history['loss']),
            'model_version': args.model_version,
            'training_timestamp': datetime.now().isoformat()
        }

        # === LƯU ARTIFACTS VÀO ĐÚNG THƯ MỤC SAGEMAKER ===
        # SageMaker sẽ nén toàn bộ thư mục 'model_dir' thành 'model.tar.gz'
        
        logging.info(f"Saving model artifacts to {model_dir}")
        # 1. Lưu model vào thư mục model_dir
        model_save_path = os.path.join(model_dir, 'best_cnn_lstm_model.keras')
        best_model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        os.remove(temp_model_file) # Xóa file tạm

        # 2. Lưu scaler vào thư mục model_dir
        scaler_car_path = os.path.join(model_dir, 'scaler_car_count.pkl')
        joblib.dump(scaler_car, scaler_car_path)
        logging.info(f"Scaler car saved to {scaler_car_path}")

        scaler_hour_path = os.path.join(model_dir, 'scaler_hour.pkl')
        joblib.dump(scaler_hour, scaler_hour_path)
        logging.info(f"Scaler hour saved to {scaler_hour_path}")

        # 3. Lưu metrics vào thư mục output_dir (KHÔNG phải model_dir)
        # SageMaker sẽ KHÔNG nén thư mục này, giúp Lambda dễ đọc hơn
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")

        logging.info(f"--- HUẤN LUYỆN SAGEMAKER PHIÊN BẢN {args.model_version} HOÀN TẤT ---")

    except Exception as e:
        logging.error(f"Lỗi nghiêm trọng trong quá trình training: {e}", exc_info=True)
        # Ghi file lỗi vào output nếu cần
        os.makedirs(output_dir, exist_ok=True)
        error_path = os.path.join(output_dir, 'failure_reason.txt')
        with open(error_path, 'w') as f:
            f.write(str(e))
        sys.exit(1)

