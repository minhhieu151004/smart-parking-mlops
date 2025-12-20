import boto3
import pandas as pd
import os
import io
import numpy as np
from datetime import datetime, timedelta
from boto3.dynamodb.conditions import Key
from io import StringIO

# --- CẤU HÌNH BIẾN MÔI TRƯỜNG ---
BUCKET_NAME = os.environ.get('S3_BUCKET', 'kltn-smart-parking-data')
SENSOR_ID = 'camera-01'
TABLE_RAW_NAME = 'SmartParkingRawData'
TABLE_PRED_NAME = 'SmartParkingPredictions'
TABLE_MAE_NAME = 'SmartParkingMAE'

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def query_items_by_range(table_name, start_time_str, end_time_str):
    """Truy vấn DynamoDB với hỗ trợ phân trang (Pagination)."""
    table = dynamodb.Table(table_name)
    items = []
    try:
        print(f"   Querying {table_name}: {start_time_str} -> {end_time_str}")
        response = table.query(
            KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_time_str, end_time_str)
        )
        items.extend(response.get('Items', []))
        
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('sensor_id').eq(SENSOR_ID) & Key('timestamp').between(start_time_str, end_time_str),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))
        return items
    except Exception as e:
        print(f"Error querying {table_name}: {e}")
        return []

def calculate_and_save_mae(df_raw, df_pred, target_date_str):
    """
    Tính toán MAE bằng cách khớp dữ liệu thực tế (Actual) và dự báo (Prediction).
    """
    try:
        if df_raw.empty or df_pred.empty:
            print("⚠️ Thiếu dữ liệu để tính toán MAE.")
            return

        # 1. Xử lý dữ liệu Actual (Raw)
        df_act = df_raw.copy()
        # Chuyển đổi timestamp sang Datetime và ép kiểu Index chuẩn
        df_act['timestamp'] = pd.to_datetime(df_act['timestamp'])
        df_act = df_act.set_index('timestamp')
        df_act.index = pd.to_datetime(df_act.index) # ÉP KIỂU DATETIMEINDEX Ở ĐÂY
        
        # Chuyển đổi car_count sang kiểu số trước khi resample
        df_act['car_count'] = pd.to_numeric(df_act['car_count'], errors='coerce')
        df_act = df_act[['car_count']].resample('5min').mean().dropna()

        # 2. Xử lý dữ liệu Prediction
        df_p = df_pred.copy()
        df_p['prediction_for'] = pd.to_datetime(df_p['prediction_for'])
        df_p = df_p.rename(columns={'prediction_for': 'timestamp'})
        df_p = df_p.set_index('timestamp')
        df_p.index = pd.to_datetime(df_p.index) # ÉP KIỂU DATETIMEINDEX Ở ĐÂY
        
        # Chuyển đổi prediction sang kiểu số
        df_p['prediction'] = pd.to_numeric(df_p['prediction'], errors='coerce')
        df_p = df_p[['prediction']].resample('5min').mean().dropna()

        # 3. Khớp (Inner Join) hai bảng dữ liệu
        df_merged = pd.merge(df_act, df_p, left_index=True, right_index=True, how='inner')

        if df_merged.empty:
            print(f"⚠️ Không tìm thấy mốc thời gian khớp nhau giữa Actual và Pred ngày {target_date_str}")
            return

        # 4. Tính toán MAE
        df_merged['abs_error'] = (df_merged['car_count'] - df_merged['prediction']).abs()
        mae_val = float(df_merged['abs_error'].mean())

        # 5. Lưu vào DynamoDB
        table_mae = dynamodb.Table(TABLE_MAE_NAME)
        table_mae.put_item(Item={
            'date': target_date_str,
            'mae': round(mae_val, 4),
            'samples_count': len(df_merged),
            'calculated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        print(f"✅ MAE Calculated: {mae_val:.4f} (Dựa trên {len(df_merged)} mẫu khớp)")

    except Exception as e:
        print(f"❌ Lỗi tính MAE chi tiết: {str(e)}")

def process_and_save_actuals(df_raw, target_date_str):
    """Lưu Actuals hàng ngày và cập nhật Master File trên S3."""
    MASTER_KEY = "parking_data/parking_data.csv"
    DAILY_KEY = f"daily_actuals/{target_date_str}.csv"
    
    df_new = df_raw.copy()
    df_new = df_new[['car_count', 'timestamp']]
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
    
    # Sắp xếp
    df_new['temp_ts'] = pd.to_datetime(df_new['timestamp'], dayfirst=True)
    df_new = df_new.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    # 1. Lưu Daily CSV
    csv_buf = StringIO()
    df_new.to_csv(csv_buf, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=DAILY_KEY, Body=csv_buf.getvalue())
    
    # 2. Cập nhật Master File (Append)
    try:
        resp = s3.get_object(Bucket=BUCKET_NAME, Key=MASTER_KEY)
        df_master = pd.read_csv(io.BytesIO(resp['Body'].read()), parse_dates=['timestamp'], dayfirst=True)
        df_master['timestamp'] = df_master['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')
    except s3.exceptions.NoSuchKey:
        df_master = pd.DataFrame(columns=['car_count', 'timestamp'])
    
    df_merged = pd.concat([df_master, df_new]).drop_duplicates(subset=['timestamp'], keep='last')
    
    df_merged['temp_ts'] = pd.to_datetime(df_merged['timestamp'], dayfirst=True)
    df_merged = df_merged.sort_values('temp_ts').drop(columns=['temp_ts'])
    
    csv_master_buf = StringIO()
    df_merged[['car_count', 'timestamp']].to_csv(csv_master_buf, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=MASTER_KEY, Body=csv_master_buf.getvalue())
    print(f"✅ Master file updated: {len(df_merged)} rows.")

def save_daily_predictions(df_pred, target_date_str):
    """Lưu kết quả dự báo của ngày target vào S3."""
    KEY = f"daily_predictions/{target_date_str}.csv"
    df = df_pred.copy()
    df['prediction_for'] = pd.to_datetime(df['prediction_for']).dt.strftime('%d/%m/%Y %H:%M:%S')
    
    csv_buf = StringIO()
    df[['prediction', 'prediction_for']].to_csv(csv_buf, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=KEY, Body=csv_buf.getvalue())
    print(f"✅ Daily Predictions saved: {KEY}")

def lambda_handler(event, context):
    # Xác định ngày hôm qua (Target date)
    now = datetime.now() + timedelta(hours=7) # Giả sử cộng 7 cho giờ VN
    target_date = (now - timedelta(days=1)).date()
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU NGÀY: {target_date_str}")

    # 1. Lấy dữ liệu RAW
    raw_start = f"{target_date_str}T00:00:00"
    raw_end = f"{target_date_str}T23:59:59"
    raw_items = query_items_by_range(TABLE_RAW_NAME, raw_start, raw_end)
    df_raw = pd.DataFrame(raw_items) if raw_items else pd.DataFrame()

    # 2. Lấy dữ liệu PREDICTIONS
    # Truy vấn rộng hơn 2h để đảm bảo lấy đủ các dự báo "cho" ngày hôm qua
    query_start = (datetime.combine(target_date, datetime.min.time()) - timedelta(hours=2)).isoformat()
    query_end = (datetime.combine(target_date, datetime.max.time()) + timedelta(hours=2)).isoformat()
    pred_items = query_items_by_range(TABLE_PRED_NAME, query_start, query_end)
    
    if pred_items:
        df_p_all = pd.DataFrame(pred_items)
        df_p_all['prediction_for_dt'] = pd.to_datetime(df_p_all['prediction_for'])
        df_pred_filtered = df_p_all[df_p_all['prediction_for_dt'].dt.date == target_date].copy()
    else:
        df_pred_filtered = pd.DataFrame()

    # 3. Thực thi các nhiệm vụ
    if not df_raw.empty:
        process_and_save_actuals(df_raw, target_date_str)
    
    if not df_pred_filtered.empty:
        save_daily_predictions(df_pred_filtered, target_date_str)

    # 4. Tính toán MAE (Phần quan trọng bạn yêu cầu)
    if not df_raw.empty and not df_pred_filtered.empty:
        calculate_and_save_mae(df_raw, df_pred_filtered, target_date_str)
    else:
        print("⚠️ Không đủ dữ liệu đối soát để tính MAE.")

    return {
        "statusCode": 200,
        "body": f"Successfully processed data for {target_date_str}"
    }