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
    """
    Thực hiện toàn bộ logic tiền xử lý và SCALING thủ công.
    Chỉ giữ lại car_count và timestamp để đưa vào mô hình.
    """
    
    # 1. Lọc chỉ lấy các cột cần thiết cho Model
    df = df[['car_count', 'timestamp']].copy() 

    # 2. Ép kiểu dữ liệu an toàn
    df['car_count'] = pd.to_numeric(df['car_count'], errors='coerce').astype(float)
    
    # Dữ liệu từ DynamoDB là ISO string (YYYY-MM-DD...), KHÔNG dùng dayfirst=True
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
    
    df = df.dropna(subset=['car_count', 'timestamp'])
    
    # Kiểm tra độ dài dữ liệu
    if len(df) < n_steps:
         raise ValueError(f"Không đủ dữ liệu lịch sử, cần {n_steps} điểm, hiện có {len(df)}.")

    # 3. Resample và Interpolate
    df = df.set_index('timestamp').sort_index()
    df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
    
    # 4. Feature Engineering và SCALING
    df_resampled['hour'] = df_resampled.index.hour
    
    df_resampled['car_count_scaled'] = df_resampled['car_count'] / CAR_MAX
    df_resampled['hour_scaled'] = df_resampled['hour'] / HOUR_MAX
    
    # 5. Tạo Sequence
    sequence = df_resampled[['car_count_scaled', 'hour_scaled']].values[-n_steps:]
    last_valid_ts = df_resampled.index[-1]
    
    # 6. Trả về mảng 3D chuẩn
    return sequence.reshape(1, n_steps, 2), last_valid_ts

def lambda_handler(event, context):
    try:
        # --- BƯỚC 1: TIẾP NHẬN DỮ LIỆU TỪ PI5 & GHI VÀO DB ---
        
        request_data = json.loads(event['body'])
        pi_timestamp_str = request_data['timestamp']
        pi_car_count = request_data['car_count']
        
        # Lấy danh sách chỗ trống
        pi_free_spots = request_data.get('free_spots', [])
        
        # Chuyển đổi timestamp sang ISO format để lưu DB Raw
        # Input Pi: "08/12/2025..." -> strptime -> ISO: "2025-12-08..."
        pi_timestamp_dt = datetime.strptime(pi_timestamp_str, '%d/%m/%Y %H:%M:%S')
        iso_timestamp = pi_timestamp_dt.isoformat()
        
        # Tạo Item để lưu vào DynamoDB
        item_to_save = {
            'sensor_id': SENSOR_ID, 
            'timestamp': iso_timestamp, 
            'car_count': Decimal(str(pi_car_count)),
            'free_spots': [int(x) for x in pi_free_spots] 
        }

        # Thực hiện ghi Raw Data
        table_raw.put_item(Item=item_to_save)
        print(f"✅ Đã ghi Raw Data: {pi_car_count} xe, Time: {iso_timestamp}")
        
        # --- BƯỚC 2: LẤY LỊCH SỬ VÀ GỌI SAGEMAKER ENDPOINT ---
        
        # 1. Query lấy 288 dòng dữ liệu lịch sử gần nhất
        response = table_raw.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID),
            Limit=N_STEPS, 
            ScanIndexForward=False 
        )
        items = response['Items']
        
        if len(items) < N_STEPS:
            msg = f"COLD START: Đang thu thập dữ liệu ({len(items)}/{N_STEPS})..."
            print(msg)
            return {'statusCode': 202, 'body': json.dumps({"status": "COLD_START", "message": msg})}

        items.reverse()
        df = pd.DataFrame(items)
        
        # 2. Tiền xử lý & Tạo Tensor
        input_tensor, last_valid_ts = _preprocess_and_scale(df) 
        
        # 3. Tính Timestamp cho thời điểm dự đoán 
        floored_ts = last_valid_ts.floor(f'{TIME_STEP_MINUTES}min') 
        pred_ts = floored_ts + timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        
        # 4. Gọi SageMaker Endpoint
        payload_data = {"instances": input_tensor.tolist()}
        json_payload = json.dumps(payload_data)
        
        print(f"📤 Đang gọi Endpoint: {ENDPOINT_NAME}")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, 
            ContentType='application/json', 
            Body=json_payload
        )
        result = json.loads(response['Body'].read().decode())
        
        # 5. Hậu xử lý
        scaled_pred_value = result['predictions'][0][0] 
        actual_pred_value = scaled_pred_value * CAR_MAX
        final_prediction = int(round(actual_pred_value))

        # 6. Lưu kết quả dự đoán vào bảng Predictions
        # last_valid_ts bây giờ sẽ đúng là tháng 12 nhờ sửa lỗi ở bước preprocess
        pred_timestamp_iso = last_valid_ts.isoformat()
        
        table_pred.put_item(
            Item={
                'sensor_id': SENSOR_ID,
                'timestamp': pred_timestamp_iso, 
                'prediction': final_prediction,
                'prediction_for': pred_ts.isoformat(), 
                'created_at': datetime.now().isoformat()
            }
        )
        print(f"✅ Dự đoán thành công: {final_prediction} xe (Time: {pred_timestamp_iso})")
        
        # 7. Trả về phản hồi
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                "prediction": final_prediction, 
                "timestamp_for": pred_ts.strftime('%Y-%m-%d %H:%M:%S'),
                "message": "Success"
            })
        }

    except Exception as e:
        print(f"❌ LỖI SYSTEM: {e}")
        return {
            'statusCode': 500, 
            'headers': {'Content-Type': 'application/json'}, 
            'body': json.dumps({"error": str(e)})
        }