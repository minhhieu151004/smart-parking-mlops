import boto3
import json
import os
import pandas as pd
from boto3.dynamodb.conditions import Key
from datetime import datetime

# --- CẤU HÌNH ---
ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
TABLE_RAW = 'SmartParkingRawData'
TABLE_PRED = 'SmartParkingPredictions'
SENSOR_ID = 'camera-01'

dynamodb = boto3.resource('dynamodb')
table_raw = dynamodb.Table(TABLE_RAW)
table_pred = dynamodb.Table(TABLE_PRED)
runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    print("Có dữ liệu mới từ DynamoDB! Đang lấy ngữ cảnh lịch sử...")

    # 1. Lấy 288 dòng dữ liệu gần nhất
    response = table_raw.query(
        KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID),
        Limit=288, 
        ScanIndexForward=False 
    )
    items = response['Items']
    
    if len(items) < 288:
        print(f"Chưa đủ dữ liệu (Hiện có: {len(items)} dòng). Cần tối thiểu 288 dòng.")
        return {"status": "Not enough data"}

    items.reverse()
    
    # 2. Chuyển đổi sang CSV string
    df = pd.DataFrame(items)

    df['car_count'] = df['car_count'].astype(float)
    
    csv_data = df[['car_count', 'timestamp']].to_csv(index=False)
    
    last_timestamp = items[-1]['timestamp']

    print(f"Đang gửi dữ liệu tới Endpoint: {ENDPOINT_NAME}")

    # 3. Gọi SageMaker Endpoint
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_data
        )
        result = json.loads(response['Body'].read().decode())
        
        pred_value = result['predicted_car_count']
        pred_time = result['for_timestamp']
        
        print(f"Kết quả: {pred_value} xe vào lúc {pred_time}")

        # 4. Lưu kết quả
        table_pred.put_item(
            Item={
                'sensor_id': SENSOR_ID,
                'timestamp': last_timestamp,
                'prediction': int(pred_value),
                'prediction_for': pred_time,
                'created_at': datetime.now().isoformat()
            }
        )
        print("💾 Đã lưu kết quả vào bảng SmartParkingPredictions.")
        return result

    except Exception as e:
        print(f"Lỗi: {e}")
        raise e