import pandas as pd
import boto3
from io import StringIO
from sklearn.metrics import mean_absolute_error
import argparse
import os
from datetime import datetime, timedelta
import logging
import json
import sys

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_csv_from_s3(s3_client, bucket, key):
    """Đọc file CSV từ S3, trả về DataFrame rỗng nếu lỗi"""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    except Exception as e:
        logger.warning(f"⚠️ Không tìm thấy hoặc lỗi đọc file {key}: {e}")
        return pd.DataFrame()

def align_dataframe_by_time(df, value_col, time_col='timestamp'):
    """
    Chuẩn hóa DataFrame về index 5 phút (00:00-23:55).
    """
    # 1. Tạo index 24 giờ chuẩn (288 điểm)
    full_time_index = pd.date_range("00:00", "23:55", freq="5T").time
    
    if df.empty:
        return pd.Series(index=full_time_index, dtype=float)
        
    # Đảm bảo cột thời gian tồn tại và convert sang datetime
    if time_col in df.columns:
        # Lưu ý: File CSV hàng ngày của bạn lưu dạng DD/MM/YYYY nên cần dayfirst=True
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[time_col])
    else:
        logging.warning(f"Không tìm thấy cột thời gian '{time_col}' trong dữ liệu.")
        return pd.Series(index=full_time_index, dtype=float)
    
    # 2. Đặt cột thời gian làm index 
    df = df.set_index(time_col)
    
    # 3. Resample về 5 phút
    try:
        # Nếu trùng index, lấy trung bình
        profile_resampled = df[value_col].resample('5T').mean()
    except TypeError:
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        profile_resampled = df[value_col].resample('5T').mean()
    
    # 4. Nội suy (Interpolate) để lấp lỗ hổng
    profile_interpolated = profile_resampled.interpolate(method='time')
    
    # 5. Nhóm theo giờ trong ngày (để chuẩn hóa về 1 ngày duy nhất)
    profile_grouped = profile_interpolated.groupby(profile_interpolated.index.time).mean()
    
    # 6. Căn chỉnh theo index chuẩn (00:00 -> 23:55)
    profile_aligned = profile_grouped.reindex(full_time_index)
    
    # 7. Lấp đầy lỗ hổng (FFill/BFill)
    profile_final = profile_aligned.ffill().bfill() 
    
    return profile_final

def check_drift(args):
    s3 = boto3.client('s3')
    
    # Logic: Check 7 ngày gần nhất tính từ hôm nay
    today = datetime.now().date()
    
    drift_days_count = 0
    drift_threshold = args.mae_threshold
    limit_days = 3 # Ngưỡng: Nếu >= 3 ngày lỗi thì báo Drift
    
    report_details = {}

    print(f"--- BẮT ĐẦU CHECK DRIFT (Window 7 ngày) ---")
    print(f"Ngưỡng MAE cho phép: {drift_threshold}")
    
    # Duyệt qua 7 ngày trước đó
    for i in range(1, 8): 
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Đường dẫn file
        act_key = f"{args.actual_prefix}{date_str}.csv"
        pred_key = f"{args.prediction_prefix}{date_str}.csv"
        
        # Tải dữ liệu
        df_act = get_csv_from_s3(s3, args.bucket, act_key)
        df_pred = get_csv_from_s3(s3, args.bucket, pred_key)
        
        status = "MISSING_DATA"
        mae = None

        if not df_act.empty and not df_pred.empty:
            try:
                # --- ALIGN DATA (QUAN TRỌNG) ---
                # 1. Actuals: Căn chỉnh theo cột 'timestamp'
                series_act = align_dataframe_by_time(
                    df_act, 
                    value_col='car_count', 
                    time_col='timestamp'
                )
                
                # 2. Predictions: Căn chỉnh theo cột 'prediction_for'
                series_pred = align_dataframe_by_time(
                    df_pred, 
                    value_col='prediction', 
                    time_col='prediction_for'
                )
                
                # 3. Tính MAE bằng Sklearn
                # fillna(0) để đảm bảo không lỗi nếu interpolate thất bại (dù hiếm)
                y_true = series_act.fillna(0).values
                y_pred = series_pred.fillna(0).values
                
                mae = mean_absolute_error(y_true, y_pred)
                
                if mae > drift_threshold:
                    drift_days_count += 1
                    status = "DRIFT"
                else:
                    status = "OK"
                    
            except Exception as e:
                logger.error(f"Lỗi tính toán ngày {date_str}: {e}")
                status = "ERROR"
        
        report_details[date_str] = {"mae": round(mae, 2) if mae is not None else None, "status": status}
        print(f"📅 {date_str}: {status} (MAE={mae})")

    # --- KẾT LUẬN ---
    is_drift_detected = drift_days_count >= limit_days
    
    print(f"--- KẾT QUẢ: {drift_days_count}/7 ngày bị Drift (Ngưỡng kích hoạt: {limit_days}) ---")
    if is_drift_detected:
        print("--- QUYẾT ĐỊNH: 🔴 RETRAIN ---")
        print("ALERT_MONITOR: DRIFT_DETECTED_TRUE") 
    else:
        print("--- QUYẾT ĐỊNH: 🟢 NO RETRAIN ---")
        print("ALERT_MONITOR: DRIFT_DETECTED_FALSE")

    # Xuất file kết quả JSON
    result = {
        "drift_detected": is_drift_detected,
        "drift_count": drift_days_count,
        "details": report_details,
        "check_timestamp": datetime.now().isoformat()
    }
    
    os.makedirs(args.output_path, exist_ok=True)
    output_file_path = os.path.join(args.output_path, 'drift_check_result.json')
    
    with open(output_file_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"✅ Đã lưu kết quả vào: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--mae-threshold', type=float, default=15.0)
    parser.add_argument('--actual-prefix', type=str, default="daily_actuals/")
    parser.add_argument('--prediction-prefix', type=str, default="daily_predictions/")
    parser.add_argument('--output-path', type=str, default="/opt/ml/processing/output")
    
    args = parser.parse_args()
    check_drift(args)