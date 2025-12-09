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

def align_dataframe_by_time(df, value_col, time_col='timestamp'):
    """
    Chuẩn hóa DataFrame về index 5 phút (00:00-23:55).
    Thêm tham số time_col để chỉ định cột thời gian cần dùng.
    """
    # 1. Tạo index 24 giờ chuẩn (288 điểm)
    full_time_index = pd.date_range("00:00", "23:55", freq="5T").time
    
    if df.empty:
        return pd.Series(index=full_time_index, dtype=float)
        
    # Đảm bảo cột thời gian tồn tại và convert sang datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[time_col])
    else:
        # Nếu không tìm thấy cột, trả về rỗng để tránh lỗi
        logging.warning(f"Không tìm thấy cột thời gian '{time_col}' trong dữ liệu.")
        return pd.Series(index=full_time_index, dtype=float)
    
    # 2. Đặt cột thời gian làm index 
    df = df.set_index(time_col)
    
    # 3. Resample về 5 phút
    try:
        profile_resampled = df[value_col].resample('5T').mean()
    except TypeError:
        # Fallback nếu value_col chưa phải dạng số
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        profile_resampled = df[value_col].resample('5T').mean()
    
    # 4. Nội suy (Interpolate)
    profile_interpolated = profile_resampled.interpolate(method='time')
    
    # 5. Nhóm theo giờ trong ngày
    profile_grouped = profile_interpolated.groupby(profile_interpolated.index.time).mean()
    
    # 6. Căn chỉnh theo index chuẩn
    profile_aligned = profile_grouped.reindex(full_time_index)
    
    # 7. Lấp đầy lỗ hổng
    profile_final = profile_aligned.ffill().bfill() 
    
    return profile_final

def check_drift(args):
    """
    CHỈ Kiểm tra Model Drift (Profile MAE).
    """
    try:
        s3 = boto3.client('s3')

        # === 1. Tải Dữ liệu Thực tế và Dự đoán ===
        # Mặc định lấy ngày hôm qua
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        actual_key = f"{args.actual_prefix}{yesterday_str}.csv"
        pred_key = f"{args.prediction_prefix}{yesterday_str}.csv"
        
        # Tải Actuals
        logging.info(f"Loading actual data from s3://{args.data_bucket}/{actual_key}...")
        try:
            obj_actual = s3.get_object(Bucket=args.data_bucket, Key=actual_key)
            df_actual = pd.read_csv(StringIO(obj_actual['Body'].read().decode('utf-8')))
        except s3.exceptions.NoSuchKey:
            logging.warning(f"Không tìm thấy file Actual: {actual_key}. Không thể tính MAE.")
            df_actual = pd.DataFrame()

        # Tải Predictions
        logging.info(f"Loading prediction data from s3://{args.data_bucket}/{pred_key}...")
        try:
            obj_pred = s3.get_object(Bucket=args.data_bucket, Key=pred_key)
            df_pred = pd.read_csv(StringIO(obj_pred['Body'].read().decode('utf-8')))
        except s3.exceptions.NoSuchKey:
            logging.warning(f"Không tìm thấy file Prediction: {pred_key}. Không thể tính MAE.")
            df_pred = pd.DataFrame()
        
        logging.info("Data loading process completed.")

        # === 2. Tính Model Drift (Actual vs Prediction MAE) ===
        logging.info("Calculating Performance Drift (MAE)...")
        mae_model = float('inf') 
        model_drift_detected = False
        
        if df_actual.empty or df_pred.empty:
            logging.warning("Dữ liệu Actual hoặc Prediction bị thiếu/rỗng. Bỏ qua Drift Check.")
            model_drift_detected = False 
        else:
            try:
                # Căn chỉnh Actuals dựa trên cột 'timestamp'
                df_actual_aligned = align_dataframe_by_time(
                    df_actual, 
                    value_col='car_count', 
                    time_col='timestamp' # <-- Actual dùng timestamp
                )
                
                # Căn chỉnh Predictions dựa trên cột 'prediction_for'
                df_pred_aligned = align_dataframe_by_time(
                    df_pred, 
                    value_col='prediction', 
                    time_col='prediction_for' # <-- Prediction dùng prediction_for
                )
                # ----------------------------
                
                # Tính MAE trực tiếp
                mae_model = mean_absolute_error(df_actual_aligned, df_pred_aligned)
                
                logging.info(f"Calculated MAE: {mae_model:.4f} (Threshold: {args.model_mae_threshold})")
                
                if mae_model > args.model_mae_threshold:
                    model_drift_detected = True
                    logging.warning(f"🔴 MODEL DRIFT DETECTED (MAE {mae_model:.4f} > {args.model_mae_threshold}).")
                else:
                    logging.info("🟢 Model Performance is good. No drift.")

            except Exception as mae_error:
                logging.error(f"Lỗi khi tính MAE: {mae_error}. Giả định không drift.", exc_info=True)
                model_drift_detected = False

        # === 3. Kết quả cuối cùng ===
        final_drift_decision = model_drift_detected
        
        if final_drift_decision:
            logging.warning(">>> FINAL DECISION: TRIGGER RETRAINING <<<")
        else:
            logging.info(">>> FINAL DECISION: SKIP RETRAINING <<<")
            
        # Ghi kết quả JSON
        result_data = {
            "drift_detected": final_drift_decision,
            "data_drift": {"detected": False},
            "model_drift": {
                "detected": model_drift_detected,
                "mae": mae_model if mae_model != float('inf') else None,
                "threshold": args.model_mae_threshold
            },
            "check_timestamp": datetime.now().isoformat()
        }

        output_path = os.path.join(args.output_path, 'drift_check_result.json')
        os.makedirs(args.output_path, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
            
        logging.info(f"Result saved to {output_path}")

    except Exception as e:
        logging.error(f"FATAL Error during drift check: {e}", exc_info=True)
        result_data = { "error": str(e), "drift_detected": False } 
        output_path = os.path.join(args.output_path, 'drift_check_result.json')
        os.makedirs(args.output_path, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Các tham số giữ nguyên
    parser.add_argument('--baseline-data-uri', type=str, default="") 
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--actual-prefix', type=str, required=True)
    parser.add_argument('--prediction-prefix', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--p-value-threshold', type=float, default=0.05)
    parser.add_argument('--model-mae-threshold', type=float, default=10.0) 
    
    args = parser.parse_args()
    args.baseline_bucket = ""
    args.baseline_key = ""

    check_drift(args)