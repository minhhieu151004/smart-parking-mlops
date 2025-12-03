import boto3
import json
import os
import pandas as pd
import numpy as np
import joblib
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from decimal import Decimal 
import tempfile
import sys

# --- CẤU HÌNH ---
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
TABLE_RAW = 'SmartParkingRawData'
TABLE_PRED = 'SmartParkingPredictions'
SENSOR_ID = 'camera-01' 
S3_BUCKET = os.environ.get('S3_BUCKET', 'kltn-smart-parking-data') 

# --- ĐƯỜNG DẪN SCALER ---
SCALER_PREFIX = 'models/production' 
SCALER_KEY_CAR = f'{SCALER_PREFIX}/scaler_car_count.pkl' 
SCALER_KEY_HOUR = f'{SCALER_PREFIX}/scaler_hour.pkl'

# Hằng số mô hình
N_STEPS = 288 
TIME_STEP_MINUTES = 5 
PREDICTION_WINDOW_MINUTES = 60 

# --- KHỞI TẠO GLOBAL & CACHING ---
SCALER_ARTIFACTS = {} 

dynamodb = boto3.resource('dynamodb')
table_raw = dynamodb.Table(TABLE_RAW)
table_pred = dynamodb.Table(TABLE_PRED)
runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

# --- HÀM TẢI SCALERS ---
def load_scalers_from_s3():
    """Tải scaler từ S3 và load chúng vào bộ nhớ."""
    global SCALER_ARTIFACTS
    
    if SCALER_ARTIFACTS:
        return SCALER_ARTIFACTS

    temp_dir = tempfile.gettempdir()
    local_car_path = os.path.join(temp_dir, 'scaler_car.pkl')
    local_hour_path = os.path.join(temp_dir, 'scaler_hour.pkl')

    try:
        s3_client.download_file(S3_BUCKET, SCALER_KEY_CAR, local_car_path)
        s3_client.download_file(S3_BUCKET, SCALER_KEY_HOUR, local_hour_path)
        
        SCALER_ARTIFACTS = {
            'scaler_car': joblib.load(local_car_path),
            'scaler_hour': joblib.load(local_car_path)
        }
        return SCALER_ARTIFACTS
    
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không thể tải hoặc load scalers từ S3. {e}")
        raise RuntimeError("Missing model artifacts for scaling.")

# --- HÀM TIỀN XỬ LÝ DỮ LIỆU ---
def _preprocess_and_scale(df, artifacts, n_steps=N_STEPS):
    df['car_count'] = pd.to_numeric(df['car_count'], errors='coerce').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['car_count', 'timestamp'])
    
    if len(df) < n_steps:
         raise ValueError(f"Không đủ dữ liệu, cần {n_steps} điểm.")

    scaler_car = artifacts['scaler_car']
    scaler_hour = artifacts['scaler_hour']

    df = df.set_index('timestamp').sort_index()
    df_resampled = df.resample(f'{TIME_STEP_MINUTES}T').mean().interpolate(method='time')
    
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['car_count_scaled'] = scaler_car.transform(df_resampled[['car_count']])
    df_resampled['hour_scaled'] = scaler_hour.transform(df_resampled[['hour']])
    
    sequence = df_resampled[['car_count_scaled', 'hour_scaled']].values[-n_steps:]
    
    return sequence.reshape(1, n_steps, 2), df_resampled.index[-1]


def lambda_handler(event, context):
    try:
        # --- BƯỚC 1: TIẾP NHẬN DỮ LIỆU THÔ TỪ PI5 (API Gateway) ---
        print("🔗 Đang xử lý HTTP POST từ Pi5...")
        
        # 1. Parse body (Dữ liệu Pi gửi lên: {"car_count": 45, "timestamp": "dd/mm/yyyy HH:MM:SS"})
        request_data = json.loads(event['body'])
        pi_timestamp_str = request_data['timestamp']
        pi_car_count = request_data['car_count']
        
        # 2. Chuẩn hóa Timestamp Pi gửi sang ISO để lưu DB
        pi_timestamp_dt = datetime.strptime(pi_timestamp_str, '%d/%m/%Y %H:%M:%S')
        iso_timestamp = pi_timestamp_dt.isoformat()
        
        # 3. Ghi dữ liệu Pi vừa gửi vào bảng Raw Data (DB Write 1)
        table_raw.put_item(
            Item={
                'sensor_id': SENSOR_ID,
                'timestamp': iso_timestamp,
                'car_count': Decimal(str(pi_car_count)) 
            }
        )
        print(f"✅ Ghi dữ liệu Pi vào DB thành công: {pi_car_count} xe.")
        
        # --- BƯỚC 2: LẤY LỊCH SỬ VÀ GỌI ENDPOINT ---
        
        # Tải scalers
        artifacts = load_scalers_from_s3()
        
        # 1. Lấy 288 dòng lịch sử (DB Read)
        response = table_raw.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID),
            Limit=N_STEPS, ScanIndexForward=False 
        )
        items = response['Items']
        
        if len(items) < N_STEPS:
            # Đây là lần khởi động hệ thống, chưa đủ 24h dữ liệu
            return {'statusCode': 202, 'body': json.dumps({"status": "COLD_START", "message": "Collecting more data..."})}

        items.reverse()
        df = pd.DataFrame(items)
        
        # 2. Tiền xử lý (Tạo Tensor chuẩn [1, 288, 2])
        input_tensor, last_valid_ts = _preprocess_and_scale(df, artifacts) 
        
        # 3. Lấy Timestamp cho dự đoán
        pred_ts = last_valid_ts.floor(f'{TIME_STEP_MINUTES}min') + timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        
        # 4. Gói vào format "instances"
        payload_data = {"instances": input_tensor.tolist()}
        json_payload = json.dumps(payload_data)
        
        # 5. Gọi SageMaker Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json_payload
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # 6. Hậu xử lý (Inverse Transform)
        scaled_pred_value = result['predictions'][0][0] 
        actual_pred_value = artifacts['scaler_car'].inverse_transform([[scaled_pred_value]])[0][0]
        final_prediction = int(round(actual_pred_value))

        # 7. Lưu kết quả dự đoán (DB Write 2)
        table_pred.put_item(
            Item={
                'sensor_id': SENSOR_ID,
                'timestamp': last_valid_ts.isoformat(),
                'prediction': final_prediction,
                'prediction_for': pred_ts.isoformat(), 
                'created_at': datetime.now().isoformat()
            }
        )
        
        # --- BƯỚC 3: TRẢ VỀ PHẢN HỒI CHO PI5 ---
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
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": str(e)})
        }