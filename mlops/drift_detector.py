import pandas as pd
import boto3
from io import StringIO
from scipy.stats import ks_2samp
import argparse
import os
from datetime import datetime, timedelta
import logging
import json 

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_drift(args):
    """
    Kiểm tra drift và ghi kết quả vào file JSON output.
    """
    try:
        s3 = boto3.client('s3')

        # Đường dẫn S3 do SageMaker cung cấp qua ProcessingInput
        input_data_uri = args.input_data 
        bucket = input_data_uri.split('/')[2]
        key = '/'.join(input_data_uri.split('/')[3:])

        logging.info(f"Loading historical data for drift check from {input_data_uri}...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

        yesterday = datetime.now() - timedelta(days=1)
        start_of_yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_yesterday = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

        df_new = df[(df['timestamp'] >= start_of_yesterday) & (df['timestamp'] <= end_of_yesterday)]
        df_baseline = df[df['timestamp'] < start_of_yesterday]

        drift_detected = False # Mặc định là không có drift
        p_value_result = 1.0 # Mặc định p-value
        ks_statistic = None # Khởi tạo

        if df_new.empty or df_baseline.empty:
            logging.warning("Not enough data to perform drift check. Assuming no drift.")
        else:
            ks_statistic, p_value = ks_2samp(df_baseline['car_count'].dropna(), df_new['car_count'].dropna())
            logging.info(f"Kolmogorov-Smirnov test: KS Statistic={ks_statistic:.4f}, P-value={p_value:.4f}")
            p_value_result = p_value
            if p_value < args.p_value_threshold:
                logging.warning(f"Drift DETECTED (p-value {p_value:.4f} < {args.p_value_threshold}).")
                drift_detected = True
            else:
                logging.info("No significant drift detected.")

        # Chuẩn bị dữ liệu kết quả
        result_data = {
            "drift_detected": drift_detected,
            "p_value": p_value_result,
            "ks_statistic": ks_statistic,
            "check_timestamp": datetime.now().isoformat()
        }

        # Ghi kết quả vào file JSON tại đường dẫn output
        # SageMaker sẽ tự động map /opt/ml/processing/output -> S3
        output_path = os.path.join(args.output_path, 'drift_check_result.json')
        logging.info(f"Writing drift check result to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)

    except Exception as e:
        logging.error(f"Error during drift check: {e}")
        # Ghi file kết quả lỗi để pipeline biết
        result_data = { "error": str(e) }
        output_path = os.path.join(args.output_path, 'drift_check_result.json')
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        raise # Raise lỗi để SageMaker biết job thất bại

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Các tham số này sẽ được SageMaker Pipeline truyền vào
    # Đường dẫn S3 (sagemaker.inputs.ProcessingInput)
    parser.add_argument('--input-data', type=str, required=True)
    # Đường dẫn local trong container (sagemaker.outputs.ProcessingOutput)
    parser.add_argument('--output-path', type=str, required=True, default='/opt/ml/processing/output') 
    parser.add_argument('--p-value-threshold', type=float, default=0.05)
    args = parser.parse_args()
    check_drift(args)

