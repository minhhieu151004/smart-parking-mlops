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
FUTURE_STEP = 12 # Dự đoán 12*5 = 60 phút trong tương lai
TIME_STEP_MINUTES = 5

def model_fn(model_dir):
    """
    Hàm này được SageMaker gọi MỘT LẦN khi Endpoint khởi động.
    """
    print(f"Bắt đầu tải model từ thư mục: {model_dir}")
    
    model_path = os.path.join(model_dir, 'best_cnn_lstm_model.keras')
    scaler_car_path = os.path.join(model_dir, 'scaler_car_count.pkl')
    scaler_hour_path = os.path.join(model_dir, 'scaler_hour.pkl')
    
    model = load_model(model_path, compile=False)
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
    Hàm này nhận dữ liệu thô từ Lambda (CSV đã được combine).
    """
    if request_content_type == 'text/csv':
        try:
            df = pd.read_csv(StringIO(request_body.decode('utf-8'))) # Decode bytes
            print(f"Đã nhận và parse {len(df)} dòng dữ liệu CSV.")
            return df
        except Exception as e:
            raise ValueError(f"Lỗi khi parse CSV: {e}")
    else:
        raise ValueError(f"Content type không được hỗ trợ: {request_content_type}")

def _preprocess_for_prediction(df, scaler_car, scaler_hour, n_steps=N_STEPS):
    """
    Logic tiền xử lý (resample, scale, lấy N bước cuối).
    """
    print("Đang tiền xử lý dữ liệu (resample, scale, lấy bước cuối)...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values('timestamp').set_index('timestamp')
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
        # Lấy timestamp cuối cùng từ dữ liệu gốc đã nhận
        last_timestamp_str = input_data_df['timestamp'].iloc[-1]
        last_timestamp = pd.to_datetime(last_timestamp_str, dayfirst=True)
        
        # Resample lần cuối để lấy mốc 5 phút cuối cùng (ví dụ: 09:01 -> 09:00)
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