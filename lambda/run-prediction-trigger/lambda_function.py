import boto3
import json
import os
import pandas as pd
import numpy as np
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from decimal import Decimal 
import sys

# --- CẤU HÌNH ---
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
TABLE_RAW = 'SmartParkingRawData'
TABLE_PRED = 'SmartParkingPredictions'
SENSOR_ID = 'camera-01' 

CAR_MAX = 100.0
HOUR_MAX = 24.0

# Hằng số mô hình
N_STEPS = 288 
TIME_STEP_MINUTES = 5 
PREDICTION_WINDOW_MINUTES = 60 

# --- KHỞI TẠO GLOBAL ---
dynamodb = boto3.resource('dynamodb')
table_raw = dynamodb.Table(TABLE_RAW)
table_pred = dynamodb.Table(TABLE_PRED)
runtime = boto3.client('sagemaker-runtime')


# --- HÀM TIỀN XỬ LÝ DỮ LIỆU ---
def _preprocess_and_scale(df, n_steps=N_STEPS):
    """Thực hiện toàn bộ logic tiền xử lý và SCALING thủ công."""
    
    df = df[['car_count', 'timestamp']].copy() 

    # 1. Ép kiểu  
    df['car_count'] = pd.to_numeric(df['car_count'], errors='coerce').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['car_count', 'timestamp'])
    
    if len(df) < n_steps:
         raise ValueError(f"Không đủ dữ liệu, cần {n_steps} điểm.")

    # 2. Resample và Interpolate 
    df = df.set_index('timestamp').sort_index()
    df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
    
    # 3. Feature Engineering và SCALING 
    df_resampled['hour'] = df_resampled.index.hour
    
    df_resampled['car_count_scaled'] = df_resampled['car_count'] / CAR_MAX
    df_resampled['hour_scaled'] = df_resampled['hour'] / HOUR_MAX
    
    # 4. Tạo Sequence
    sequence = df_resampled[['car_count_scaled', 'hour_scaled']].values[-n_steps:]
    last_valid_ts = df_resampled.index[-1]
    
    # 5. Trả về mảng 3D chuẩn
    return sequence.reshape(1, n_steps, 2), last_valid_ts

def lambda_handler(event, context):
    try:
        # --- BƯỚC 1: TIẾP NHẬN DỮ LIỆU THÔ TỪ PI5 & GHI DB ---
        
        request_data = json.loads(event['body'])
        pi_timestamp_str = request_data['timestamp']
        pi_car_count = request_data['car_count']
        
        pi_timestamp_dt = datetime.strptime(pi_timestamp_str, '%d/%m/%Y %H:%M:%S')
        iso_timestamp = pi_timestamp_dt.isoformat()
        
        table_raw.put_item(
            Item={'sensor_id': SENSOR_ID, 'timestamp': iso_timestamp, 'car_count': Decimal(str(pi_car_count))}
        )
        print(f"✅ Ghi dữ liệu Pi vào DB thành công: {pi_car_count} xe.")
        
        # --- BƯỚC 2: LẤY LỊCH SỬ VÀ GỌI ENDPOINT ---
        
        # 1. Lấy 288 dòng lịch sử 
        response = table_raw.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID),
            Limit=N_STEPS, ScanIndexForward=False 
        )
        items = response['Items']
        
        if len(items) < N_STEPS:
            return {'statusCode': 202, 'body': json.dumps({"status": "COLD_START", "message": "Collecting more data..."})}

        items.reverse()
        df = pd.DataFrame(items)
        
        # 2. Tiền xử lý (Tạo Tensor chuẩn [1, 288, 2])
        input_tensor, last_valid_ts = _preprocess_and_scale(df) 
        
        # 3. Tính Timestamp cho dự đoán
        floored_ts = last_valid_ts.floor(f'{TIME_STEP_MINUTES}min') 
        pred_ts = floored_ts + timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        
        # 4. Gói vào format "instances"
        payload_data = {"instances": input_tensor.tolist()}
        json_payload = json.dumps(payload_data)
        
        print(f"📤 Đang gửi Tensor 3D tới Endpoint: {ENDPOINT_NAME}")

        # 5. Gọi SageMaker Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=json_payload
        )
        result = json.loads(response['Body'].read().decode())
        
        # 6. Hậu xử lý
        scaled_pred_value = result['predictions'][0][0] 
        actual_pred_value = scaled_pred_value * CAR_MAX
        final_prediction = int(round(actual_pred_value))

        # 7. Lưu kết quả dự đoán
        table_pred.put_item(
            Item={
                'sensor_id': SENSOR_ID,
                'timestamp': last_valid_ts.isoformat(),
                'prediction': final_prediction,
                'prediction_for': pred_ts.isoformat(), 
                'created_at': datetime.now().isoformat()
            }
        )
        
        # 8. Trả về phản hồi cho Pi5
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                "prediction": final_prediction, 
                "timestamp_for": pred_ts.strftime('%Y-%m-%d %H:%M:%S')
            })
        }

    except Exception as e:
        print(f"❌ LỖI KHÔNG XỬ LÝ: {e}")
        return {'statusCode': 500, 'headers': {'Content-Type': 'application/json'}, 'body': json.dumps({"error": str(e)})}