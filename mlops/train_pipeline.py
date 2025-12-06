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
import shutil
import tarfile # Cần thiết cho Warm Start
import tensorflow as tf

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

# --- HÀM TẢI MODEL CŨ TỪ S3 ---
def download_and_extract_model(s3_uri, extract_path):
    """Tải model.tar.gz từ S3 và giải nén."""
    try:
        logging.info(f"Đang tải model cũ từ: {s3_uri}")
        s3 = boto3.client('s3')
        
        # Parse S3 URI (s3://bucket/key)
        bucket_name = s3_uri.split('/')[2]
        key = '/'.join(s3_uri.split('/')[3:])
        
        local_tar_path = '/tmp/old_model.tar.gz'
        s3.download_file(bucket_name, key, local_tar_path)
        
        logging.info("Đang giải nén model cũ...")
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
            
        logging.info(f"Đã giải nén model cũ vào {extract_path}")
        return True
    except Exception as e:
        logging.error(f"Lỗi khi tải model cũ: {e}")
        return False

# --- Hàm chính ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--n-steps', type=int, default=288)
    parser.add_argument('--future-step', type=int, default=12)
    parser.add_argument('--model-version', type=str, required=True)
    
    # --- THAM SỐ WARM START MỚI ---
    parser.add_argument('--warm-start-model-uri', type=str, default=None) 
    # ------------------------------

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    input_data_path = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')

    args = parser.parse_args()

    logging.info(f"--- BẮT ĐẦU HUẤN LUYỆN (Version: {args.model_version}) ---")
    
    try:
        # 1. Load Data 
        data_file_path = os.path.join(input_data_path, 'parking_data.csv') 
        if not os.path.exists(data_file_path):
            # Nếu input là thư mục, lấy file csv đầu tiên
            csv_files = [f for f in os.listdir(input_data_path) if f.endswith('.csv')]
            if csv_files:
                 data_file_path = os.path.join(input_data_path, csv_files[0])
            else:
                 data_file_path = input_data_path # Fallback

        logging.info(f"Reading data from {data_file_path}")
        df = pd.read_csv(data_file_path, parse_dates=['timestamp'], dayfirst=True)
        
        # 2. Preprocess
        df_processed, scaler_car, scaler_hour = preprocess_data(df)
        
        if df_processed.empty or len(df_processed) < (args.n_steps + args.future_step):
            raise ValueError(f"Không đủ dữ liệu. Cần ít nhất {args.n_steps + args.future_step} dòng.")

        X, y = create_sequences(df_processed, n_steps=args.n_steps, future_step=args.future_step)
        
        if X.shape[0] == 0:
            raise ValueError("Dữ liệu quá ngắn để tạo sequence.")
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Build Model (Khởi tạo kiến trúc)
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
        logging.info("Model architecture built.")

        # 4. --- LOGIC WARM START (FINE-TUNING) ---
        epochs_to_run = args.epochs
        
        if args.warm_start_model_uri and args.warm_start_model_uri != "None":
            logging.info(">>> PHÁT HIỆN YÊU CẦU WARM START <<<")
            old_model_dir = "/tmp/old_model"
            
            if download_and_extract_model(args.warm_start_model_uri, old_model_dir):
                # Model SavedModel nằm trong thư mục con '1' (theo chuẩn của chúng ta)
                old_saved_model_path = os.path.join(old_model_dir, "1")
                
                try:
                    logging.info(f"Loading weights từ {old_saved_model_path}...")
                    old_model = load_model(old_saved_model_path)
                    
                    # Chép trọng số từ model cũ sang model mới
                    model.set_weights(old_model.get_weights())
                    logging.info("✅ Đã nạp thành công trọng số từ model cũ.")
                    
                    # Giảm số epochs xuống còn 2 để tránh catastrophic forgetting
                    epochs_to_run = 2
                    logging.info(f"Chuyển sang chế độ Fine-tuning: Giảm epochs xuống {epochs_to_run}")
                    
                except Exception as load_err:
                    logging.error(f"Lỗi khi load model cũ: {load_err}. Sẽ train từ đầu.")
            else:
                 logging.warning("Không tải được model cũ. Sẽ train từ đầu.")
        else:
            logging.info("Training from scratch (Huấn luyện từ đầu).")
        # ----------------------------------------

        # 5. Training
        temp_model_file = 'best_model_temp.h5'
        checkpoint = ModelCheckpoint(temp_model_file, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(
            X_train, y_train, epochs=epochs_to_run, batch_size=64,
            validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping], verbose=2
        )
        
        # Load best weights
        best_model = load_model(temp_model_file)
        val_loss = best_model.evaluate(X_val, y_val, verbose=0)
        
        # 6. Save Artifacts
        logging.info(f"Saving model artifacts to {model_dir}")

        # Lưu model SavedModel (Folder '1')
        export_path = os.path.join(model_dir, '1')
        best_model.save(export_path) 
        
        if os.path.exists(temp_model_file): os.remove(temp_model_file) 

        # Lưu Scalers
        joblib.dump(scaler_car, os.path.join(model_dir, 'scaler_car_count.pkl'))
        joblib.dump(scaler_hour, os.path.join(model_dir, 'scaler_hour.pkl'))

        # --- COPY INFERENCE CODE ---
        source_dir = os.path.dirname(os.path.realpath(__file__))
        code_output_dir = os.path.join(model_dir, "code") # Tạo thư mục code/ bên trong model
        os.makedirs(code_output_dir, exist_ok=True)
        
        INFERENCE_SCRIPT = os.path.join(source_dir, "code", "inference.py")
        REQUIREMENTS_FILE = os.path.join(source_dir, "code", "requirements.txt")
        
        if os.path.exists(INFERENCE_SCRIPT):
            shutil.copy(INFERENCE_SCRIPT, os.path.join(code_output_dir, "inference.py"))
            logging.info("✅ Copied inference.py")
            
        if os.path.exists(REQUIREMENTS_FILE):
            shutil.copy(REQUIREMENTS_FILE, os.path.join(code_output_dir, "requirements.txt"))
            logging.info("✅ Copied requirements.txt")
        # ------------------------------------------
        
        # Lưu Metrics
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'val_loss': val_loss,
                'training_loss': history.history['loss'][-1], 
                'model_version': args.model_version
            }, f, indent=4)

        logging.info("--- TRAINING HOÀN TẤT ---")

    except Exception as e:
        logging.error(f"Critical Error: {e}", exc_info=True)
        # Ghi file lỗi để debug
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'failure_reason.txt'), 'w') as f:
            f.write(str(e))
        sys.exit(1)