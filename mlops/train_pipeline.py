import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import shutil  
import sys
import glob  

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_model(input_shape):
    """Xây dựng kiến trúc mô hình Hybrid (CNN-LSTM)."""
    model = Sequential([
        # --- CNN Block ---
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Reshape
        Reshape((-1, 128)),
        
        # --- LSTM Block ---
        LSTM(units=150, return_sequences=True),
        Dropout(0.3),
        LSTM(units=100),
        Dropout(0.3),
        
        # --- Output Layer ---
        Dense(units=50, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    
    # SageMaker Paths
    parser.add_argument('--model_dir', type=str) 
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    
    args, _ = parser.parse_known_args()

    args.model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    
    logging.info("--- BẮT ĐẦU TRAINING ---")
    
    try:
        # 1. LOAD DỮ LIỆU ĐÃ PREPROCESS
        data_path = os.path.join(args.train, 'train_data.npy')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {data_path}")
            
        logging.info(f"📂 Đang tải dữ liệu từ {data_path}...")
        data = np.load(data_path, allow_pickle=True).item()
        X_all = data['X']
        y_all = data['y']
        
        # 2. CHIA DATA TRAIN/VAL
        total_len = len(X_all)
        valid_len = int(total_len * 0.1)
        valid_start = int(total_len * 0.8)
        valid_end = int(total_len * 0.9)
        
        X_val = X_all[valid_start : valid_end]
        y_val = y_all[valid_start : valid_end]
        
        X_train = np.concatenate((X_all[:valid_start], X_all[valid_end:]), axis=0)
        y_train = np.concatenate((y_all[:valid_start], y_all[valid_end:]), axis=0)
        
        logging.info(f"   -> Train size: {len(X_train)}")
        logging.info(f"   -> Valid size: {len(X_val)}")

        # 3. BUILD MODEL & TRAINING
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model(input_shape)
        model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='mean_squared_error')
        
        checkpoint_path = os.path.join(args.model_dir, 'best_model_checkpoint.h5')
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        ]

        logging.info("🚀 Bắt đầu quá trình Fit...")
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2
        )
        
        # 4. LƯU ARTIFACTS
        logging.info(f"💾 Đang lưu model artifacts vào {args.model_dir}...")

        # A. Lưu Model Format SavedModel
        export_path = os.path.join(args.model_dir, '1')
        model.save(export_path) 
        logging.info(f"✅ Đã lưu TensorFlow SavedModel vào {export_path}")
        
        # ---------------------------------------------------------------------
        logging.info("📥 Đang tìm và copy các file Scaler (.pkl)...")
        
        # Tìm tất cả file .pkl trong thư mục input data (nơi preprocessing gửi đến)
        scaler_files = glob.glob(os.path.join(args.train, "*.pkl"))
        
        if not scaler_files:
            logging.warning("⚠️ CẢNH BÁO: Không tìm thấy file .pkl nào trong input data! Pipeline Evaluate sẽ bị lỗi.")
        else:
            for file in scaler_files:
                shutil.copy(file, args.model_dir)
                logging.info(f"   ✅ Đã copy kèm: {os.path.basename(file)}")
        # ---------------------------------------------------------------------
        
        # C. COPY INFERENCE CODE 
        current_dir = os.path.dirname(os.path.realpath(__file__))
        code_output_dir = os.path.join(args.model_dir, "code")
        os.makedirs(code_output_dir, exist_ok=True)
        
        inference_src = os.path.join(current_dir, "inference.py") 
        requirements_src = os.path.join(current_dir, "requirements.txt")
        
        if os.path.exists(inference_src):
            shutil.copy(inference_src, os.path.join(code_output_dir, "inference.py"))
        
        if os.path.exists(requirements_src):
            shutil.copy(requirements_src, os.path.join(code_output_dir, "requirements.txt"))

        logging.info("--- TRAINING HOÀN TẤT THÀNH CÔNG ---")

    except Exception as e:
        logging.error(f"❌ Training Thất Bại: {e}", exc_info=True)
        sys.exit(1)