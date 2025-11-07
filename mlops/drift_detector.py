import pandas as pd
import boto3
from io import StringIO
from sklearn.metrics import mean_absolute_error # Dùng MAE
from scipy.stats import ks_2samp # Vẫn dùng cho Data Drift
import argparse
import os
from datetime import datetime, timedelta
import logging
import json
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def align_dataframe_by_time(df, value_col):
    """
    Chuẩn hóa DataFrame về index 5 phút (00:00-23:55)
    sử dụng resample và nội suy tuyến tính (linear interpolation).
    """
    # 1. Tạo index 24 giờ chuẩn (288 điểm)
    full_time_index = pd.date_range("00:00", "23:55", freq="5T").time
    
    if df.empty:
        # Trả về một Series rỗng (NaN) với index chuẩn
        return pd.Series(index=full_time_index, dtype=float)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    
    # 2. Đặt timestamp làm index
    df.set_index('timestamp', inplace=True)
    
    # 3. Resample (Lấy mẫu lại) về 5T (tạo ra các mốc 00:00, 00:05...)
    #    Điều này tạo ra các `NaN` ở những nơi bị thiếu (ví dụ 1:05)
    profile_resampled = df[value_col].resample('5T').mean()
    
    # 4. Nội suy (Interpolate) theo thời gian 
    profile_interpolated = profile_resampled.interpolate(method='time')
    
    # 5. Nhóm theo giờ trong ngày (nếu dữ liệu kéo dài nhiều ngày) và tính trung bình
    profile_grouped = profile_interpolated.groupby(profile_interpolated.index.time).mean()
    
    # 6. Căn chỉnh (reindex) theo index chuẩn 288 điểm
    profile_aligned = profile_grouped.reindex(full_time_index)
    
    # 7. Lấp đầy các lỗ hổng còn lại (ví dụ 00:00 nếu Pi bắt đầu lúc 00:01)
    profile_final = profile_aligned.ffill().bfill() 
    
    return profile_final

def check_drift(args):
    """
    Kiểm tra Data Drift (KS-Test) và Model Drift (Profile MAE với Nội suy).
    """
    try:
        s3 = boto3.client('s3')

        # === 1. Tải Dữ liệu ===
        logging.info(f"Loading baseline data from {args.baseline_data_uri}...")
        obj_baseline = s3.get_object(Bucket=args.baseline_bucket, Key=args.baseline_key)
        df_baseline = pd.read_csv(StringIO(obj_baseline['Body'].read().decode('utf-8')), parse_dates=['timestamp'], dayfirst=True)

        yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        actual_key = f"{args.actual_prefix}{yesterday_str}.csv"
        pred_key = f"{args.prediction_prefix}{yesterday_str}.csv"
        
        logging.info(f"Loading actual data from s3://{args.data_bucket}/{actual_key}...")
        obj_actual = s3.get_object(Bucket=args.data_bucket, Key=actual_key)
        df_actual = pd.read_csv(StringIO(obj_actual['Body'].read().decode('utf-8')), parse_dates=['timestamp'], dayfirst=True)
        
        logging.info(f"Loading prediction data from s3://{args.data_bucket}/{pred_key}...")
        obj_pred = s3.get_object(Bucket=args.data_bucket, Key=pred_key)
        df_pred = pd.read_csv(StringIO(obj_pred['Body'].read().decode('utf-8')), parse_dates=['timestamp'], dayfirst=True)
        
        logging.info("All data loaded successfully.")

        # === 2. Tính Data Drift (KS-Test) ===
        logging.info("Calculating Data Drift (Baseline vs Actual) using KS-Test...")
        
        df_actual_data = df_actual['car_count'].dropna()
        
        if df_actual_data.empty:
            logging.warning("Actual data file is empty (seed run). Forcing data drift detection.")
            data_drift_detected = True
            p_value = 0.0
            ks_statistic = float('inf')
        else:
            ks_statistic, p_value = ks_2samp(df_baseline['car_count'].dropna(), df_actual_data)
            data_drift_detected = p_value < args.p_value_threshold
        
        logging.info(f"Data Drift (KS-Test): KS Statistic={ks_statistic:.4f}, P-value={p_value:.4f}")
        if data_drift_detected:
            logging.warning(f"DATA DRIFT DETECTED (p-value {p_value:.4f} < {args.p_value_threshold}).")
        else:
            logging.info("No significant Data Drift detected.")

        # === 3. Tính Model Drift (Actual vs Prediction MAE) ===
        logging.info("Calculating Model Drift (Actual Profile vs Prediction Profile)...")
        mae_model = float('inf') 
        model_drift_detected = True 
        
        try:
            # Căn chỉnh và NỘI SUY dữ liệu thực tế và dự đoán
            df_actual_aligned = align_dataframe_by_time(df_actual, 'car_count')
            df_pred_aligned = align_dataframe_by_time(df_pred, 'predicted_car_count')
            
            # Kiểm tra nếu file trống (dẫn đến toàn NaN)
            if df_actual_aligned.isnull().all() or df_pred_aligned.isnull().all():
                logging.warning("Actual or Prediction data is completely empty (seed run). Assuming model drift.")
                mae_model = float('inf')
                model_drift_detected = True
            else:
                # Tính MAE trực tiếp
                mae_model = mean_absolute_error(df_actual_aligned, df_pred_aligned)
                model_drift_detected = mae_model > args.model_mae_threshold

        except Exception as mae_error:
            logging.error(f"Error during Model MAE calculation: {mae_error}. Assuming model drift.")
            
        logging.info(f"Model Drift (MAE): MAE = {mae_model:.4f}")
        if model_drift_detected:
            logging.warning(f"MODEL DRIFT DETECTED (MAE {mae_model:.4f} > {args.model_mae_threshold}).")
        else:
            logging.info("No significant Model Drift detected.")

        # === 4. Quyết định cuối cùng ===
        final_drift_decision = data_drift_detected or model_drift_detected
        if final_drift_decision:
            logging.warning("FINAL DECISION: DRIFT DETECTED. Triggering retrain.")
        else:
            logging.info("FINAL DECISION: No drift detected.")
            
        # Ghi kết quả
        result_data = {
            "drift_detected": final_drift_decision,
            "data_drift": {
                "detected": data_drift_detected,
                "p_value": p_value, 
                "ks_statistic": ks_statistic
            },
            "model_drift": {
                "detected": model_drift_detected,
                "mae": mae_model if mae_model != float('inf') else None,
                "threshold": args.model_mae_threshold
            },
            "check_timestamp": datetime.now().isoformat()
        }

        output_path = os.path.join(args.output_path, 'drift_check_result.json')
        logging.info(f"Writing drift check result to {output_path}")
        os.makedirs(args.output_path, exist_ok=True) # (Thêm os.makedirs để đảm bảo)
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)

    except Exception as e:
        logging.error(f"FATAL Error during drift check: {e}", exc_info=True) # (Thêm exc_info=True để log chi tiết)
        result_data = { "error": str(e), "drift_detected": True } # (Buộc drift nếu có lỗi)
        output_path = os.path.join(args.output_path, 'drift_check_result.json')
        os.makedirs(args.output_path, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-data-uri', type=str, required=True)
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--actual-prefix', type=str, required=True)
    parser.add_argument('--prediction-prefix', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    
    # Ngưỡng
    parser.add_argument('--p-value-threshold', type=float, default=0.05) 
    parser.add_argument('--model-mae-threshold', type=float, default=10.0) 
    
    args = parser.parse_args()
    
    args.baseline_bucket = args.baseline_data_uri.split('/')[2]
    args.baseline_key = '/'.join(args.baseline_data_uri.split('/')[3:])

    check_drift(args)