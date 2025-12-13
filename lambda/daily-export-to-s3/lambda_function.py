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
    """Query DynamoDB."""
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

def process_and_save_actuals(s3_client, df_raw, target_date_str):
    """
    Xử lý Actuals: Lưu car_count trước, timestamp sau.
    """
    MASTER_KEY = "parking_data/parking_data.csv"
    DAILY_KEY = f"daily_actuals/{target_date_str}.csv"
    
    # 1. Chuẩn bị dữ liệu
    df_new = df_raw.copy()
    if 'free_spots' in df_new.columns:
        df_new = df_new.drop(columns=['free_spots'])
    
    # Thứ tự: car_count, timestamp
    df_new = df_new[['car_count', 'timestamp']]
    
    # Format timestamp
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
    
    # Sort
    df_new['temp_ts'] = pd.to_datetime(df_new['timestamp'], dayfirst=True)
    df_new = df_new.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    # --- LƯU DAILY ACTUALS ---
    csv_buffer_daily = StringIO()
    df_new.to_csv(csv_buffer_daily, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=DAILY_KEY, Body=csv_buffer_daily.getvalue())
    print(f"✅ Daily Actuals saved: {DAILY_KEY}")
    
    # --- APPEND MASTER FILE ---
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        master_csv_string = response['Body'].read().decode('utf-8')
        df_master = pd.read_csv(io.StringIO(master_csv_string), parse_dates=['timestamp'], dayfirst=True)
        df_master['timestamp'] = df_master['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
    except s3_client.exceptions.NoSuchKey:
        print("   Master file chưa tồn tại. Tạo file mới.")
        df_master = pd.DataFrame(columns=['car_count', 'timestamp'])
    
    # Merge & Sort
    df_merged = pd.concat([df_master, df_new])
    df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
    
    df_merged['temp_ts'] = pd.to_datetime(df_merged['timestamp'], dayfirst=True)
    df_merged = df_merged.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    # Đảm bảo thứ tự cột Master
    df_merged = df_merged[['car_count', 'timestamp']]
    
    # Lưu Master
    csv_buffer_master = StringIO()
    df_merged.to_csv(csv_buffer_master, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=MASTER_KEY, Body=csv_buffer_master.getvalue())
    print(f"✅ Master File updated: {len(df_merged)} rows.")

def save_daily_predictions(s3_client, df_pred, target_date_str):

    DAILY_PRED_KEY = f"daily_predictions/{target_date_str}.csv"
    
    if df_pred.empty: return

    df = df_pred.copy()
    
    # Format prediction_for
    if 'prediction_for' in df.columns:
         df['prediction_for'] = pd.to_datetime(df['prediction_for']).dt.strftime('%d/%m/%Y %H:%M:%S')

    # prediction: Giá trị dự báo (số xe)
    # prediction_for: Thời điểm dự báo (để so khớp với actual)
    cols = ['prediction', 'prediction_for']
    
    # Lọc cột an toàn (chỉ lấy nếu tồn tại)
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]

    # Lưu S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=DAILY_PRED_KEY, Body=csv_buffer.getvalue())
    print(f"✅ Daily Predictions saved: {DAILY_PRED_KEY} (Cols: {list(df.columns)})")

# --- HÀM HANDLER ---
def lambda_handler(event, context):
    now = datetime.now()
    target_date = (now - timedelta(days=1)).date()
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"--- START EXPORT: {target_date_str} ---")

    # 1. RAW DATA
    raw_start = f"{target_date_str}T00:00:00"
    raw_end = f"{target_date_str}T23:59:59"
    raw_items = query_items_by_range(TABLE_RAW, raw_start, raw_end)
    
    if raw_items:
        df_raw = pd.DataFrame(raw_items)
        process_and_save_actuals(s3, df_raw, target_date_str)
    else:
        print(f"⚠️ No Raw Data for {target_date_str}")

    # 2. PREDICTIONS
    buffer_hours = 2
    query_start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=buffer_hours)
    query_end_dt = datetime.combine(target_date, datetime.max.time())
    
    pred_items = query_items_by_range(TABLE_PRED, query_start_dt.isoformat(), query_end_dt.isoformat())

    if pred_items:
        df_pred = pd.DataFrame(pred_items)
        
        # Filter by prediction_for date
        df_pred['prediction_for_dt'] = pd.to_datetime(df_pred['prediction_for'])
        df_pred_filtered = df_pred[df_pred['prediction_for_dt'].dt.date == target_date].copy()
        
        # Cleanup temp col
        df_pred_filtered = df_pred_filtered.drop(columns=['prediction_for_dt'])
        
        # Save
        save_daily_predictions(s3, df_pred_filtered, target_date_str)
    else:
        print(f"⚠️ No Predictions found.")

    return {"statusCode": 200, "body": "Success"}