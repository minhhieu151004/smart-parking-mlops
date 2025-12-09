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
PREDICTION_WINDOW_MINUTES = 60  # Mô hình dự đoán trước 60'

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

def export_archive_to_s3(df, s3_folder, target_date_str):
    if df.empty:
        return

    # Sắp xếp lại cho chuẩn
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        # Format lại timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
        
        # Nếu là file prediction, format luôn cột prediction_for
        if 'prediction_for' in df.columns:
             df['prediction_for'] = pd.to_datetime(df['prediction_for']).dt.strftime('%d/%m/%Y %H:%M:%S')

    if 'sensor_id' in df.columns:
         df = df.drop(columns=['sensor_id'], errors='ignore')

    # Lưu lên S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_key = f"{s3_folder}/{target_date_str}.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"✅ Đã lưu file {s3_key} ({len(df)} dòng)")

# --- HÀM HANDLER ---
def lambda_handler(event, context):
    now = datetime.now()
    
    # 1. Xác định ngày mục tiêu (Hôm qua)
    target_date = (now - timedelta(days=1)).date()
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"--- BẮT ĐẦU TỔNG HỢP DỮ LIỆU CHO NGÀY: {target_date_str} ---")

    # ==============================================================================
    # XỬ LÝ 1: RAW DATA (Dữ liệu thực tế) - Logic: Dựa vào timestamp (lúc gửi)
    # ==============================================================================
    # Raw data thì timestamp chính là thời điểm thực tế, nên query đúng ngày là được
    raw_start = f"{target_date_str}T00:00:00"
    raw_end = f"{target_date_str}T23:59:59"
    
    raw_items = query_items_by_range(TABLE_RAW, raw_start, raw_end)
    if raw_items:
        df_raw = pd.DataFrame(raw_items)
        # Lưu Actuals
        export_archive_to_s3(df_raw, "daily_actuals", target_date_str)
        # Gộp Master (Giả sử bạn đã có hàm consolidate_master_file ở trên)
        # consolidate_master_file(s3, df_raw) 
    else:
        print(f"⚠️ Không có Raw Data cho ngày {target_date_str}")

    # ==============================================================================
    # XỬ LÝ 2: PREDICTIONS (Dữ liệu dự báo) - Logic: Dựa vào prediction_for
    # ==============================================================================
    # Để lấy đủ dự đoán CHO ngày 8/12, ta cần lấy các dự đoán được TẠO RA từ:
    # 22:00 ngày 7/12 (dự đoán cho 23:00 -> 00:00 ngày 8/12)
    # đến 23:00 ngày 8/12 (dự đoán cho 00:00 ngày 9/12 - cái này sẽ bị lọc bỏ sau)
    
    # Mở rộng cửa sổ query lùi về quá khứ (Buffer thêm 2 tiếng cho chắc)
    buffer_hours = 2 
    query_start_dt = datetime.combine(target_date, datetime.min.time()) - timedelta(hours=buffer_hours)
    query_end_dt = datetime.combine(target_date, datetime.max.time())
    
    pred_query_start = query_start_dt.isoformat()
    pred_query_end = query_end_dt.isoformat()
    
    print(f"🔍 Quét bảng Prediction rộng hơn: từ {pred_query_start} đến {pred_query_end}")
    pred_items = query_items_by_range(TABLE_PRED, pred_query_start, pred_query_end)

    if pred_items:
        df_pred = pd.DataFrame(pred_items)
        
        # --- LOGIC LỌC QUAN TRỌNG ---
        # Chuyển đổi prediction_for sang datetime để so sánh
        df_pred['prediction_for_dt'] = pd.to_datetime(df_pred['prediction_for'])
        
        # Chỉ giữ lại những dòng mà prediction_for thuộc đúng ngày mục tiêu
        df_pred_filtered = df_pred[df_pred['prediction_for_dt'].dt.date == target_date].copy()
        
        print(f"   -> Tìm thấy {len(df_pred)} dự đoán trong khoảng query.")
        print(f"   -> Sau khi lọc theo 'prediction_for' == {target_date_str}: Còn {len(df_pred_filtered)} dòng.")
        
        # Xóa cột tạm
        df_pred_filtered = df_pred_filtered.drop(columns=['prediction_for_dt'])
        
        # Lưu file Archive
        export_archive_to_s3(df_pred_filtered, "daily_predictions", target_date_str)
    else:
        print(f"⚠️ Không tìm thấy Prediction nào.")

    return {"statusCode": 200, "body": "Success"}