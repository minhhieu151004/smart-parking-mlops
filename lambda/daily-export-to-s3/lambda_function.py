import boto3
import pandas as pd
import os
import json
from datetime import datetime, timedelta 
from boto3.dynamodb.conditions import Key
from io import StringIO
import io 

# --- CẤU HÌNH ---
BUCKET_NAME = os.environ.get('S3_BUCKET', 'kltn-smart-parking-data')
SENSOR_ID = 'camera-01'
TABLE_RAW = 'SmartParkingRawData'
TABLE_PRED = 'SmartParkingPredictions'

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# --- HÀM 1: TRUY VẤN DYNAMODB ---
def query_items_for_day(table_name, date_str):
    """Truy vấn tất cả các mục từ DynamoDB cho một ngày cụ thể."""
    table = dynamodb.Table(table_name)
    start_of_day = f"{date_str}T00:00:00"
    end_of_day = f"{date_str}T23:59:59"
    items = []
    
    try:
        print(f"   Querying {table_name} from {start_of_day} to {end_of_day}...")
        response = table.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_of_day, end_of_day)
        )
        items.extend(response['Items'])
        
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_of_day, end_of_day),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response['Items'])
            
        return items
    except Exception as e:
        print(f"Lỗi khi query bảng {table_name}: {e}")
        return []


# --- HÀM 2: LƯU TRỮ ARCHIVE ---
def export_archive_to_s3(df, s3_folder, date_str):
    """Lưu trữ dữ liệu hàng ngày (Actuals/Predictions) vào folder archive."""
    if df.empty:
        return
        
    df = df.sort_values('timestamp')
    
    # Format lại timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')

    if 'sensor_id' in df.columns: 
         df = df.drop(columns=['sensor_id'], errors='ignore')

    if s3_folder == "daily_actuals":
         cols = ['car_count', 'timestamp']
         # Chỉ giữ lại các cột có trong df
         valid_cols = [c for c in cols if c in df.columns]
         df = df[valid_cols]
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_key = f"{s3_folder}/{date_str}.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"✅ Đã lưu file Archive: {s3_key}")


# --- HÀM 3: NỐI VÀ GHI ĐÈ MASTER FILE ---
def consolidate_master_file(s3_client, df_new_day):
    """Tải file parking_data.csv gốc, nối dữ liệu mới và ghi đè."""
    MASTER_KEY = "parking_data/parking_data.csv"
    
    # 1. Tải Master File 
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        master_csv_string = response['Body'].read().decode('utf-8')
        df_master = pd.read_csv(io.StringIO(master_csv_string), parse_dates=['timestamp'], dayfirst=True)
        print(f"   Đã tải Master File: {len(df_master)} dòng.")
    except s3_client.exceptions.NoSuchKey:
        print("   Master file chưa tồn tại. Tạo file mới.")
        df_master = pd.DataFrame() 

    # 2. Chuẩn bị dữ liệu mới
    cols_needed = ['timestamp', 'car_count']
    valid_cols = [c for c in cols_needed if c in df_new_day.columns]
    df_new = df_new_day[valid_cols].copy()
    
    # 3. Format timestamp thống nhất
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
    if not df_master.empty:
        df_master['timestamp'] = pd.to_datetime(df_master['timestamp'], dayfirst=True).dt.strftime('%d/%m/%Y %H:%M:%S')

    # 4. Nối và Ghi đè
    df_merged = pd.concat([df_master, df_new]).drop_duplicates(subset=['timestamp'], keep='last')
    df_merged['temp_ts'] = pd.to_datetime(df_merged['timestamp'], dayfirst=True)
    df_merged = df_merged.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    csv_buffer = io.StringIO()
    df_merged.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=MASTER_KEY,
        Body=csv_buffer.getvalue()
    )
    print(f"✅ Đã ghi đè Master File tổng: {len(df_merged)} dòng.")


# --- HÀM HANDLER ---
def lambda_handler(event, context):
    now = datetime.now()
    
    yesterday = now - timedelta(days=1)
    target_date_str = yesterday.strftime('%Y-%m-%d')
    
    print(f"--- BẮT ĐẦU BACKUP DỮ LIỆU NGÀY HÔM QUA: {target_date_str} ---")
    
    # 1. TRUY VẤN DỮ LIỆU CẦN THIẾT 
    raw_items = query_items_for_day(TABLE_RAW, target_date_str)
    pred_items = query_items_for_day(TABLE_PRED, target_date_str)

    if not raw_items:
         print(f"⚠️ Không có dữ liệu Pi/thực tế nào ngày {target_date_str}. Kết thúc.")
         return {"statusCode": 200, "body": f"No data for {target_date_str}"}
    
    # Chuyển raw items sang DataFrame
    df_raw_yesterday = pd.DataFrame(raw_items)
    
    # 2. TASK 3: MASTER CONSOLIDATION (Nối dữ liệu hôm qua vào Master File)
    consolidate_master_file(s3, df_raw_yesterday)
    
    # 3. TASK 1: ARCHIVAL (Lưu file thực tế ngày hôm qua)
    export_archive_to_s3(df_raw_yesterday, "daily_actuals", target_date_str)
    
    # 4. TASK 2: ARCHIVAL (Lưu file dự đoán ngày hôm qua)
    if pred_items:
        df_pred_yesterday = pd.DataFrame(pred_items)
        export_archive_to_s3(df_pred_yesterday, "daily_predictions", target_date_str)
    
    print("--- HOÀN TẤT ARCHIVAL VÀ CONSOLIDATION ---")
    return {"statusCode": 200, "body": f"Backup completed for {target_date_str}"}