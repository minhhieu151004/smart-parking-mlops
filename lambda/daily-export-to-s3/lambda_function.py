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
    """Query DynamoDB theo khoảng thời gian tùy chỉnh."""
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

def consolidate_master_file(s3_client, df_new_day):
    """Tải file Master, nối dữ liệu mới và ghi đè."""
    MASTER_KEY = "parking_data/parking_data.csv"
    
    cols_needed = ['timestamp', 'car_count']
    valid_cols = [c for c in cols_needed if c in df_new_day.columns]
    df_new = df_new_day[valid_cols].copy()

    # 1. Tải Master File hiện tại
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        master_csv_string = response['Body'].read().decode('utf-8')
        # Đọc file Master (Master lưu DD/MM/YYYY nên cần dayfirst=True)
        df_master = pd.read_csv(io.StringIO(master_csv_string), parse_dates=['timestamp'], dayfirst=True)
        print(f"   Đã tải Master File: {len(df_master)} dòng.")
        
        # Format lại timestamp Master để thống nhất string
        df_master['timestamp'] = df_master['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
        
    except s3_client.exceptions.NoSuchKey:
        print("   Master file chưa tồn tại. Tạo file mới.")
        df_master = pd.DataFrame(columns=cols_needed)

    # 2. Nối và Ghi đè
    df_merged = pd.concat([df_master, df_new])
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sắp xếp lại (Tạo cột tạm datetime để sort)
    df_merged['temp_ts'] = pd.to_datetime(df_merged['timestamp'], dayfirst=True)
    df_merged = df_merged.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    # Ghi lên S3
    csv_buffer = io.StringIO()
    df_merged.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=MASTER_KEY,
        Body=csv_buffer.getvalue()
    )
    print(f"✅ Đã cập nhật Master File tổng: {len(df_merged)} dòng.")

def export_archive_to_s3(df, s3_folder, target_date_str):
    """Lưu trữ dữ liệu hàng ngày (Chỉ lưu, không format lại)."""
    if df.empty:
        return

    
    # Chỉ giữ các cột quan trọng
    if s3_folder == "daily_actuals":
        keep_cols = ['timestamp', 'car_count']
    else:
        keep_cols = ['timestamp', 'prediction', 'prediction_for']
        
    final_cols = [c for c in keep_cols if c in df.columns]
    df_to_save = df[final_cols].copy()

    # Lưu lên S3
    csv_buffer = StringIO()
    df_to_save.to_csv(csv_buffer, index=False)
    
    s3_key = f"{s3_folder}/{target_date_str}.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"✅ Đã lưu file Archive {s3_key}: {len(df_to_save)} dòng")

# --- HÀM HANDLER ---
def lambda_handler(event, context):
    now = datetime.now()
    target_date = (now - timedelta(days=1)).date()
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"--- BẮT ĐẦU TỔNG HỢP DỮ LIỆU CHO NGÀY: {target_date_str} ---")

    # ==============================================================================
    # 1. XỬ LÝ RAW DATA (ACTUALS)
    # ==============================================================================
    raw_start = f"{target_date_str}T00:00:00"
    raw_end = f"{target_date_str}T23:59:59"
    
    raw_items = query_items_by_range(TABLE_RAW, raw_start, raw_end)
    if raw_items:
        df_raw = pd.DataFrame(raw_items)
        
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
        
        # TASK 1: Lưu file daily_actuals (Truyền df đã chuẩn)
        export_archive_to_s3(df_raw, "daily_actuals", target_date_str)
        
        # TASK 3: Append vào Master File (Truyền df đã chuẩn)
        consolidate_master_file(s3, df_raw)
        
    else:
        print(f"⚠️ Không có Raw Data cho ngày {target_date_str}")

    # ==============================================================================
    # 2. XỬ LÝ PREDICTIONS
    # ==============================================================================
    buffer_hours = 3
    query_start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=buffer_hours)
    query_end_dt = datetime.combine(target_date, datetime.max.time())
    
    pred_items = query_items_by_range(TABLE_PRED, query_start_dt.isoformat(), query_end_dt.isoformat())

    if pred_items:
        df_pred = pd.DataFrame(pred_items)
        
        # 1. Timestamp (Lúc tạo)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
        df_pred['prediction_for_dt'] = pd.to_datetime(df_pred['prediction_for'])
        df_pred['prediction_for'] = df_pred['prediction_for_dt'].dt.strftime('%d/%m/%Y %H:%M:%S')
        # --------------------------------------------------------
        
        # Lọc theo prediction_for (ngày mục tiêu)
        df_pred_filtered = df_pred[df_pred['prediction_for_dt'].dt.date == target_date].copy()
        
        # Xóa cột tạm
        df_pred_filtered = df_pred_filtered.drop(columns=['prediction_for_dt'])
        
        # TASK 2: Lưu file daily_predictions
        export_archive_to_s3(df_pred_filtered, "daily_predictions", target_date_str)
    else:
        print(f"⚠️ Không tìm thấy Prediction nào.")

    return {"statusCode": 200, "body": "Success"}