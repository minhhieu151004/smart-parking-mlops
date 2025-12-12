import boto3
import pandas as pd
import os
import io
from datetime import datetime, timedelta
from boto3.dynamodb.conditions import Key
from io import StringIO

# --- CẤU HÌNH ---
BUCKET_NAME = os.environ.get('S3_BUCKET', 'kltn-smart-parking-data')
SENSOR_ID = 'camera-01'
TABLE_RAW = 'SmartParkingRawData'
TABLE_PRED = 'SmartParkingPredictions'

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def query_items_by_range(table_name, start_time_str, end_time_str):
    """Query DynamoDB (Giữ nguyên logic cũ của bạn)."""
    table = dynamodb.Table(table_name)
    items = []
    try:
        print(f"   Querying {table_name}: {start_time_str} -> {end_time_str}")
        response = table.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_time_str, end_time_str)
        )
        items.extend(response['Items'])
        
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_time_str, end_time_str),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response['Items'])
        return items
    except Exception as e:
        print(f"Error querying {table_name}: {e}")
        return []

def consolidate_master_file(s3_client, df_new_input):
    """Nối dữ liệu mới vào Master File (parking_data.csv)."""
    MASTER_KEY = "parking_data/parking_data.csv"
    
    # 1. Tạo bản sao và chuẩn bị dữ liệu 
    df_new = df_new_input.copy()
    if 'free_spots' in df_new.columns:
        df_new = df_new.drop(columns=['free_spots'])
    
    # Chỉ lấy 2 cột cần thiết
    df_new = df_new[['timestamp', 'car_count']]
    
    # Format sang DD/MM/YYYY để khớp với Master File
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')

    # 2. Tải Master File hiện tại
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        master_csv_string = response['Body'].read().decode('utf-8')
        # Đọc file Master (dayfirst=True vì nó đang lưu dạng DD/MM/YYYY)
        df_master = pd.read_csv(io.StringIO(master_csv_string), parse_dates=['timestamp'], dayfirst=True)
        print(f"   Đã tải Master File: {len(df_master)} dòng.")
        
        # Format lại timestamp Master về string để nối
        df_master['timestamp'] = df_master['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
        
    except s3_client.exceptions.NoSuchKey:
        print("   Master file chưa tồn tại. Tạo file mới.")
        df_master = pd.DataFrame(columns=['timestamp', 'car_count'])

    # 3. Nối và Ghi đè
    df_merged = pd.concat([df_master, df_new])
    
    # Loại bỏ trùng lặp
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sắp xếp lại (cần convert tạm sang datetime để sort đúng)
    df_merged['temp_ts'] = pd.to_datetime(df_merged['timestamp'], dayfirst=True)
    df_merged = df_merged.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    # Lưu lên S3
    csv_buffer = io.StringIO()
    df_merged.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(Bucket=BUCKET_NAME, Key=MASTER_KEY, Body=csv_buffer.getvalue())
    print(f"✅ Đã cập nhật Master File tổng: {len(df_merged)} dòng.")

def export_archive_to_s3(df_input, s3_folder, target_date_str):
    """Lưu file Archive (Giữ logic cũ của bạn + Lọc cột)."""
    if df_input.empty:
        return

    # Tạo bản sao để không ảnh hưởng biến bên ngoài
    df = df_input.copy()

    # 1. Sắp xếp và Format (Logic cũ của bạn)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
        
        if 'prediction_for' in df.columns:
             df['prediction_for'] = pd.to_datetime(df['prediction_for']).dt.strftime('%d/%m/%Y %H:%M:%S')

    # 2. Bỏ cột không cần thiết (sensor_id, free_spots)
    if s3_folder == "daily_actuals":
        # Chỉ giữ lại timestamp và car_count -> Tự động loại bỏ free_spots
        df = df[['timestamp', 'car_count']]
    elif s3_folder == "daily_predictions":
        # Với prediction thì giữ lại các cột này
        cols = ['timestamp', 'prediction', 'prediction_for']
        df = df[[c for c in cols if c in df.columns]]

    # 3. Lưu lên S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_key = f"{s3_folder}/{target_date_str}.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"✅ Đã lưu file Archive {s3_key}: {len(df)} dòng")

# --- HÀM HANDLER ---
def lambda_handler(event, context):
    now = datetime.now()
    
    # Lấy ngày hôm qua
    target_date = (now - timedelta(days=1)).date()
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"--- BẮT ĐẦU TỔNG HỢP DỮ LIỆU CHO NGÀY: {target_date_str} ---")

    # ==============================================================================
    # 1. RAW DATA (ACTUALS)
    # ==============================================================================
    raw_start = f"{target_date_str}T00:00:00"
    raw_end = f"{target_date_str}T23:59:59"
    
    raw_items = query_items_by_range(TABLE_RAW, raw_start, raw_end)
    if raw_items:
        df_raw = pd.DataFrame(raw_items)
        
        # TASK 1: Lưu file daily_actuals 
        export_archive_to_s3(df_raw.copy(), "daily_actuals", target_date_str)
        
        # TASK 3: Append vào Master File
        consolidate_master_file(s3, df_raw.copy())
        
    else:
        print(f"⚠️ Không có Raw Data cho ngày {target_date_str}")

    # ==============================================================================
    # 2. PREDICTIONS (Dữ liệu dự báo)
    # ==============================================================================
    # Buffer 2 tiếng để lấy dự đoán vắt ngày
    buffer_hours = 3
    query_start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=buffer_hours)
    query_end_dt = datetime.combine(target_date, datetime.max.time())
    
    pred_query_start = query_start_dt.isoformat()
    pred_query_end = query_end_dt.isoformat()
    
    pred_items = query_items_by_range(TABLE_PRED, pred_query_start, pred_query_end)

    if pred_items:
        df_pred = pd.DataFrame(pred_items)
        
        # Lọc theo prediction_for
        df_pred['prediction_for_dt'] = pd.to_datetime(df_pred['prediction_for'])
        df_pred_filtered = df_pred[df_pred['prediction_for_dt'].dt.date == target_date].copy()
        
        # Xóa cột tạm
        df_pred_filtered = df_pred_filtered.drop(columns=['prediction_for_dt'])
        
        # TASK 2: Lưu file daily_predictions
        export_archive_to_s3(df_pred_filtered, "daily_predictions", target_date_str)
    else:
        print(f"⚠️ Không tìm thấy Prediction nào.")

    return {"statusCode": 200, "body": "Success"}