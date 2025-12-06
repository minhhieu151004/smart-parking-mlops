import boto3
import pandas as pd
import os
import json
from datetime import datetime
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

# --- HÀM 1: TRUY VẤN DYNAMODB (Tách logic ra ngoài) ---
def query_items_for_day(table_name, date_str):
    """Truy vấn tất cả các mục từ DynamoDB cho một ngày cụ thể."""
    table = dynamodb.Table(table_name)
    start_of_day = f"{date_str}T00:00:00"
    end_of_day = f"{date_str}T23:59:59"
    items = []
    
    try:
        response = table.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_of_day, end_of_day)
        )
        items.extend(response['Items'])
        
        while 'LastEvaluatedKey' in response:
             # Xử lý phân trang
            response = table.query(
                KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_of_day, end_of_day),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response['Items'])
            
        return items
    except Exception as e:
        print(f"Lỗi khi query bảng {table_name}: {e}")
        return []


# --- HÀM 2: LƯU TRỮ ARCHIVE (Task 1 & 2) ---
def export_archive_to_s3(df, s3_folder, date_str):
    """Lưu trữ dữ liệu hàng ngày (Actuals/Predictions) vào folder archive."""
    
    df = df.sort_values('timestamp')
    
    # Định dạng lại timestamp cho chuẩn train_pipeline (DD/MM/YYYY)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')

    if 'sensor_id' in df.columns: 
         df = df.drop(columns=['sensor_id'], errors='ignore')

    # Định nghĩa cột cuối cùng
    if s3_folder == "daily_actuals":
         cols = ['car_count', 'timestamp']
         df = df[cols]
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_key = f"{s3_folder}/{date_str}.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"✅ Đã lưu file Archive: {s3_key}")
    return df

# --- HÀM 3: NỐI VÀ GHI ĐÈ MASTER FILE (Task 3) ---
def consolidate_master_file(s3_client, df_new_day):
    """
    Tải file parking_data.csv gốc, nối dữ liệu mới và ghi đè (Master Consolidation).
    """
    MASTER_KEY = "parking_data/parking_data.csv"
    
    # 1. Tải Master File 
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        master_csv_string = response['Body'].read().decode('utf-8')
        # Parse với dayfirst=True để đảm bảo đọc đúng dữ liệu cũ đã lưu
        df_master = pd.read_csv(io.StringIO(master_csv_string), parse_dates=['timestamp'], dayfirst=True)
        print(f"   Đã tải Master File: {len(df_master)} dòng.")
    except s3_client.exceptions.NoSuchKey:
        print("   Master file chưa tồn tại. Tạo file mới.")
        df_master = pd.DataFrame() # Tạo DataFrame rỗng

    # 2. Chuyển đổi format cho DF mới 
    df_new = df_new_day[['timestamp', 'car_count']].copy()
    
    # 3. Nối dữ liệu và loại bỏ trùng lặp
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
    df_master['timestamp'] = df_master['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')

    # Nối Master cũ và dữ liệu mới, loại bỏ trùng lặp dựa trên timestamp
    df_merged = pd.concat([df_master, df_new]).drop_duplicates(subset=['timestamp'], keep='last')
    
    # 4. Ghi đè file Master (Master Consolidation)
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
    today_str = now.strftime('%Y-%m-%d')
    
    print(f"--- BẮT ĐẦU CHU TRÌNH CONSOLIDATE & ARCHIVE {today_str} ---")
    
    # 1. TRUY VẤN DỮ LIỆU CẦN THIẾT
    raw_items = query_items_for_day(TABLE_RAW, today_str)
    pred_items = query_items_for_day(TABLE_PRED, today_str)

    if not raw_items:
         print("⚠️ Không có dữ liệu Pi/thực tế nào để backup. Kết thúc.")
         return {"statusCode": 200, "body": "No data today"}
    
    # Chuyển raw items sang DataFrame (dùng cho cả Consolidate và Archival)
    df_raw_today = pd.DataFrame(raw_items)
    
    # 2. TASK 3: MASTER CONSOLIDATION (Nối dữ liệu vào Master File)
    consolidate_master_file(s3_client, df_raw_today)
    
    # 3. TASK 1: ARCHIVAL (Lưu trữ file thực tế theo ngày)
    export_archive_to_s3(df_raw_today, "daily_actuals", today_str)
    
    # 4. TASK 2: ARCHIVAL (Lưu trữ file dự đoán theo ngày)
    if pred_items:
        df_pred_today = pd.DataFrame(pred_items)
        export_archive_to_s3(df_pred_today, "daily_predictions", today_str)
    
    print("--- HOÀN TẤT ARCHIVAL VÀ CONSOLIDATION ---")
    return {"statusCode": 200, "body": f"Backup completed for {today_str}"}