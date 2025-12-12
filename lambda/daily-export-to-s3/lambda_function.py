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
    """Query DynamoDB theo khoảng thời gian."""
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

def consolidate_master_file(s3_client, df_new_day_obj):
    """
    Tải Master, nối dữ liệu mới và lưu.
    Input: df_new_day_obj có cột 'timestamp' là datetime objects.
    """
    MASTER_KEY = "parking_data/parking_data.csv"
    
    # Chỉ lấy cột cần thiết và copy để không ảnh hưởng bên ngoài
    cols_needed = ['timestamp', 'car_count']
    valid_cols = [c for c in cols_needed if c in df_new_day_obj.columns]
    df_new = df_new_day_obj[valid_cols].copy()

    # 1. Tải Master File (Đang lưu dạng string dd/mm/yyyy)
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        master_csv_string = response['Body'].read().decode('utf-8')
        
        # Đọc và Parse ngày tháng ngay lập tức
        df_master = pd.read_csv(io.StringIO(master_csv_string))
        if 'timestamp' in df_master.columns:
            # Parse DD/MM/YYYY
            df_master['timestamp'] = pd.to_datetime(df_master['timestamp'], dayfirst=True)
            
        print(f"   Đã tải Master File: {len(df_master)} dòng.")
        
    except s3_client.exceptions.NoSuchKey:
        print("   Master file chưa tồn tại. Tạo file mới.")
        df_master = pd.DataFrame(columns=cols_needed)

    # 2. Nối 2 Dataframe (Đều đang là Datetime Objects -> Sort chuẩn)
    df_merged = pd.concat([df_master, df_new])
    
    # Loại bỏ trùng lặp
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sắp xếp theo thời gian
    df_merged = df_merged.sort_values('timestamp')
    
    # 3. Format sang String DD/MM/YYYY để lưu
    df_merged['timestamp'] = df_merged['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
    
    # Ghi lên S3
    csv_buffer = io.StringIO()
    df_merged.to_csv(csv_buffer, index=False)
    
    s3_client.put_object(Bucket=BUCKET_NAME, Key=MASTER_KEY, Body=csv_buffer.getvalue())
    print(f"✅ Đã cập nhật Master File tổng: {len(df_merged)} dòng.")

def export_archive_to_s3(df_obj, s3_folder, target_date_str):
    """
    Lưu file daily.
    Input: df_obj có cột 'timestamp' là datetime objects.
    """
    if df_obj.empty: return

    # Copy để không làm hỏng biến gốc
    df = df_obj.copy()

    # Sắp xếp (Dựa trên Object)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        # Format sang String để lưu
        df['timestamp'] = df['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
        
    # Xử lý riêng cho cột prediction_for
    if 'prediction_for' in df.columns:
         # Chuyển về datetime nếu chưa phải (đề phòng)
         df['prediction_for'] = pd.to_datetime(df['prediction_for'])
         df['prediction_for'] = df['prediction_for'].dt.strftime('%d/%m/%Y %H:%M:%S')

    # Lọc cột
    if s3_folder == "daily_actuals":
        keep_cols = ['timestamp', 'car_count']
    else:
        keep_cols = ['timestamp', 'prediction', 'prediction_for']
        
    final_cols = [c for c in keep_cols if c in df.columns]
    df = df[final_cols]

    # Lưu S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_key = f"{s3_folder}/{target_date_str}.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"✅ Đã lưu file Archive {s3_key}: {len(df)} dòng")

# --- HÀM HANDLER ---
def lambda_handler(event, context):
    now = datetime.now()
    
    # Xác định ngày hôm qua
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
        
        # [QUAN TRỌNG] CHUẨN HÓA SANG DATETIME OBJECT NGAY TỪ ĐẦU
        # Raw Data từ DynamoDB là ISO string (YYYY-MM-DD...), Pandas tự hiểu
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        
        # TASK 1: Lưu file daily_actuals
        export_archive_to_s3(df_raw, "daily_actuals", target_date_str)
        
        # TASK 3: Append vào Master File
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
        
        # Convert timestamp sang Object
        if 'timestamp' in df_pred.columns:
            df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
            
        # Convert prediction_for sang Object để lọc
        df_pred['prediction_for'] = pd.to_datetime(df_pred['prediction_for'])
        
        # Lọc theo prediction_for (ngày mục tiêu)
        df_pred_filtered = df_pred[df_pred['prediction_for'].dt.date == target_date].copy()
        
        print(f"   -> Lọc còn {len(df_pred_filtered)} dòng dự đoán cho ngày {target_date_str}")
        
        # TASK 2: Lưu file daily_predictions
        export_archive_to_s3(df_pred_filtered, "daily_predictions", target_date_str)
    else:
        print(f"⚠️ Không tìm thấy Prediction nào.")

    return {"statusCode": 200, "body": "Success"}