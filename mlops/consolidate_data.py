import pandas as pd
import boto3
from io import StringIO
import argparse
import os
from datetime import datetime, timedelta
import logging
import sys

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def consolidate(args):
    """
    Tải file baseline cũ và file actual của ngày hôm qua,
    nối chúng lại, và lưu file baseline mới.
    """
    try:
        s3 = boto3.client('s3')

        # === 1. Tải Dữ liệu ===
        # Tải file baseline (chứa dữ liệu đến hết hôm kia)
        logging.info(f"Loading baseline data from s3://{args.baseline_bucket}/{args.baseline_key}...")
        obj_baseline = s3.get_object(Bucket=args.baseline_bucket, Key=args.baseline_key)
        df_baseline = pd.read_csv(StringIO(obj_baseline['Body'].read().decode('utf-8')))

        # Tải file actual (chỉ chứa dữ liệu hôm qua)
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        actual_key = f"{args.actual_prefix}{yesterday_str}.csv"
        
        logging.info(f"Loading actual data from s3://{args.data_bucket}/{actual_key}...")
        obj_actual = s3.get_object(Bucket=args.data_bucket, Key=actual_key)
        df_actual = pd.read_csv(StringIO(obj_actual['Body'].read().decode('utf-8')))
        
        logging.info("Data loaded successfully.")

        # === 2. Nối (Append/Concat) Dữ liệu ===
        logging.info(f"Appending yesterday's data ({len(df_actual)} rows) to baseline ({len(df_baseline)} rows)...")
        # Đảm bảo cột nhất quán (nếu cần)
        # df_actual = df_actual[df_baseline.columns] 
        df_new_master = pd.concat([df_baseline, df_actual], ignore_index=True)
        logging.info(f"New master file has {len(df_new_master)} rows.")

        # === 3. Ghi kết quả ra Output Path của SageMaker ===
        # SageMaker sẽ tự động upload file này lên S3 (ghi đè file baseline cũ)
        output_path = os.path.join(args.output_path, 'parking_data.csv')
        logging.info(f"Saving new master data to {output_path}")
        df_new_master.to_csv(output_path, index=False)
        logging.info("Consolidation complete.")

    except Exception as e:
        logging.error(f"Error during consolidation: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Các đường dẫn này do SageMaker Pipeline (build_pipeline.py) cung cấp
    parser.add_argument('--baseline-data-uri', type=str, required=True)
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--actual-prefix', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True) # /opt/ml/processing/output_consolidated
    
    args = parser.parse_args()
    
    # Tách bucket và key từ baseline_data_uri
    args.baseline_bucket = args.baseline_data_uri.split('/')[2]
    args.baseline_key = '/'.join(args.baseline_data_uri.split('/')[3:])

    consolidate(args)
