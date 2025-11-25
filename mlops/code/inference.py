import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import json
from io import StringIO
from datetime import datetime, timedelta 

# === Hằng số ===
N_STEPS = 288
FUTURE_STEP = 12 
TIME_STEP_MINUTES = 5

def model_fn(model_dir):
    """
    Hàm này được SageMaker gọi MỘT LẦN khi Endpoint khởi động.
    """
    print(f"Bắt đầu tải model từ thư mục: {model_dir}")
    
    # Trỏ vào thư mục '1' chứa SavedModel
    model_path = os.path.join(model_dir, '1')

    scaler_car_path = os.path.join(model_dir, 'scaler_car_count.pkl')
    scaler_hour_path = os.path.join(model_dir, 'scaler_hour.pkl')
    
    print(f"Loading model from: {model_path}")
    # TensorFlow tự động nhận diện SavedModel từ thư mục
    model = tf.keras.models.load_model(model_path)
    
    scaler_car = joblib.load(scaler_car_path)
    scaler_hour = joblib.load(scaler_hour_path)
    
    print("Tải model và scalers thành công.")
    
    return {
        'model': model,
        'scaler_car': scaler_car,
        'scaler_hour': scaler_hour
    }

def input_fn(request_body, request_content_type):
    """
    Hàm này nhận dữ liệu thô từ Lambda (CSV) và xử lý triệt để các lỗi kiểu dữ liệu.
    """
    if request_content_type == 'text/csv':
        try:
            # 1. Đọc CSV
            df = pd.read_csv(StringIO(request_body.decode('utf-8')))
            print(f"Đã nhận {len(df)} dòng dữ liệu.")

            # 2. ÉP KIỂU DỮ LIỆU 
            if 'car_count' in df.columns:
                df['car_count'] = pd.to_numeric(df['car_count'], errors='coerce')
                
                df = df.dropna(subset=['car_count'])
                
                # Chuyển sang float32 (định dạng ưa thích của TensorFlow)
                df['car_count'] = df['car_count'].astype('float32')
            else:
                raise ValueError("Lỗi: File CSV thiếu cột 'car_count'")
            
            # 3. Xử lý timestamp 
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

            return df
            
        except Exception as e:
            print(f"LỖI TRONG INPUT_FN: {e}")
            raise ValueError(f"Lỗi khi parse CSV: {e}")
    else:
        raise ValueError(f"Content type không được hỗ trợ: {request_content_type}")

def _preprocess_for_prediction(df, scaler_car, scaler_hour, n_steps=N_STEPS):
    """
    Logic tiền xử lý (resample, scale, lấy N bước cuối).
    """
    print("Đang tiền xử lý dữ liệu (resample, scale, lấy bước cuối)...")
    
    # Đảm bảo timestamp là index và đã được sort
    if 'timestamp' in df.columns: # Nếu chưa set index
         df = df.set_index('timestamp')
    
    df = df.sort_index()
    
    # Resample và Interpolate để điền dữ liệu thiếu
    df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
    
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['car_count_scaled'] = scaler_car.transform(df_resampled[['car_count']])
    df_resampled['hour_scaled'] = scaler_hour.transform(df_resampled[['hour']])
    
    sequence = df_resampled[['car_count_scaled', 'hour_scaled']].values[-n_steps:]
    
    if len(sequence) < n_steps:
        raise ValueError(f"Không đủ dữ liệu, cần ít nhất {n_steps} điểm dữ liệu ({n_steps * TIME_STEP_MINUTES} phút).")
        
    return sequence.reshape(1, n_steps, 2)

def predict_fn(input_data_df, model_artifacts):
    """
    Hàm này chạy logic dự đoán VÀ tính toán timestamp.
    """
    print("Bắt đầu hàm predict_fn...")
    
    # Lấy lại model và scalers
    model = model_artifacts['model']
    scaler_car = model_artifacts['scaler_car']
    scaler_hour = model_artifacts['scaler_hour']

    # 1. Tiền xử lý dữ liệu (lấy N bước cuối, chuẩn hóa)
    input_sequence = _preprocess_for_prediction(input_data_df, scaler_car, scaler_hour)
    
    # 2. Chạy dự đoán
    print("Đang chạy model.predict()...")
    prediction_scaled = model.predict(input_sequence)[0][0]
    
    # 3. Hậu xử lý (inverse transform)
    prediction_actual = scaler_car.inverse_transform([[prediction_scaled]])[0][0]
    prediction = int(round(prediction_actual))
    
    print(f"Dự đoán (số lượng): {prediction}")

    # === 4. Tính toán Timestamp cho Dự đoán ===
    try:        
        # Lấy mốc thời gian cuối cùng trong dữ liệu đầu vào
        if isinstance(input_data_df.index, pd.DatetimeIndex):
             last_timestamp = input_data_df.index[-1]
        else:
             last_timestamp = input_data_df['timestamp'].iloc[-1]

        # Resample lần cuối để lấy mốc 5 phút tròn (ví dụ: 09:01 -> 09:00)
        last_aligned_timestamp = pd.Timestamp(last_timestamp).floor(f'{TIME_STEP_MINUTES}T')
        
        # Tính thời gian tương lai
        future_timedelta_minutes = TIME_STEP_MINUTES * FUTURE_STEP
        prediction_timestamp = last_aligned_timestamp + timedelta(minutes=future_timedelta_minutes)
        
        prediction_timestamp_str = prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Timestamp cuối cùng: {last_timestamp}, Timestamp dự đoán: {prediction_timestamp_str}")

    except Exception as e:
        print(f"Lỗi khi tính toán timestamp: {e}. Dùng timestamp hiện tại làm dự phòng.")
        prediction_timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    # 5. Trả về cả hai
    return {
        "predicted_car_count": prediction,
        "for_timestamp": prediction_timestamp_str 
    }

def output_fn(prediction_result, response_content_type):
    """
    Hàm này định dạng kết quả trả về cho Lambda.
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction_result)
    else:
        raise ValueError(f"Content type không được hỗ trợ: {response_content_type}")