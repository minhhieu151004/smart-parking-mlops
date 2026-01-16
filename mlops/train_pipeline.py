import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import shutil
import sys
import glob

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_cnn_lstm_optimized(time_steps, rows, cols, channels):
    """
    Input Shape: (Batch_Size, TimeSteps, Rows, Cols, Channels)
    """
    model = Sequential()
    
    # --- KHỐI CNN  ---
    # Lớp 1
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'),
                              input_shape=(time_steps, rows, cols, channels)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    # Lớp 2
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    # Flatten
    model.add(TimeDistributed(Flatten()))
    
    # --- KHỐI LSTM ---
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.4))
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.4))
    
    # --- OUTPUT LAYER ---
    model.add(Dense(1, activation='sigmoid'))
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=60) #epoch
    parser.add_argument('--learning-rate', type=float, default=0.005) #learning rate
    parser.add_argument('--batch-size', type=int, default=32) #batch size
    
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
        valid_len = int(total_len * 0.2)
        valid_start = int(total_len * 0.6)
        valid_end = int(total_len * 0.8)
        
        X_val = X_all[valid_start : valid_end]
        y_val = y_all[valid_start : valid_end]
        
        X_train = np.concatenate((X_all[:valid_start], X_all[valid_end:]), axis=0)
        y_train = np.concatenate((y_all[:valid_start], y_all[valid_end:]), axis=0)
        
        logging.info(f"   -> Train size: {len(X_train)}")
        logging.info(f"   -> Valid size: {len(X_val)}")

        # 3. BUILD MODEL & TRAINING
        # X_train shape mong đợi: (Samples, 4, 8, 9, 2)
        if len(X_train.shape) != 5:
             raise ValueError(f"❌ Shape dữ liệu không đúng chuẩn 5D: {X_train.shape}. Kiểm tra lại Preprocessing.")

        # Lấy kích thước input động từ dữ liệu
        _, time_steps, rows, cols, channels = X_train.shape
        
        logging.info(f"🏗️ Building Model input: ({time_steps}, {rows}, {cols}, {channels})")
        
        model = build_cnn_lstm_optimized(time_steps, rows, cols, channels)
        
        # Sử dụng MAE loss 
        model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='mae')
        
        checkpoint_path = os.path.join(args.model_dir, 'best_model.h5') 
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
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
        
        # Tìm file .pkl trong thư mục input data
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