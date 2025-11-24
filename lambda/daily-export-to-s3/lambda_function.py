import boto3
import pandas as pd
import os
from datetime import datetime
from boto3.dynamodb.conditions import Key
from io import StringIO

# --- CẤU HÌNH ---
BUCKET_NAME = os.environ.get('S3_BUCKET', 'kltn-smart-parking-data')
SENSOR_ID = 'camera-01'
TABLE_RAW = 'SmartParkingRawData'
TABLE_PRED = 'SmartParkingPredictions'

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def export_table_to_s3(table_name, s3_folder, date_str):
    print(f"--- Đang xử lý bảng {table_name} ---")
    table = dynamodb.Table(table_name)
    
    start_of_day = f"{date_str}T00:00:00"
    end_of_day = f"{date_str}T23:59:59"
    
    try:
        response = table.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_of_day, end_of_day)
        )
        items = response['Items']
        
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_of_day, end_of_day),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response['Items'])
            
        if not items:
            print(f"Không có dữ liệu ngày {date_str} cho bảng {table_name}.")
            return

        df = pd.DataFrame(items)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
            
        df = df.sort_values('timestamp')

        if table_name == TABLE_RAW:
            cols = ['car_count', 'timestamp']
            df = df[cols]
        elif table_name == TABLE_PRED:
            if 'sensor_id' in df.columns:
                df = df.drop(columns=['sensor_id'])

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_key = f"{s3_folder}/{date_str}.csv"
        s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
        print(f"Đã lưu file: s3://{BUCKET_NAME}/{s3_key}")

    except Exception as e:
        print(f"Lỗi khi export bảng {table_name}: {e}")
        raise e

def lambda_handler(event, context):
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    
    print(f"Bắt đầu backup dữ liệu ngày: {today_str}")
    
    export_table_to_s3(TABLE_RAW, "daily_actuals", today_str)
    export_table_to_s3(TABLE_PRED, "daily_predictions", today_str)
    
    return {"statusCode": 200, "body": f"Backup completed for {today_str}"}