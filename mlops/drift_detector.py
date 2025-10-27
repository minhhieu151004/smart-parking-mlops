import pandas as pd
import boto3
from io import StringIO
from scipy.stats import ks_2samp
import argparse
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_drift(args):
    # Khởi tạo boto3 
    s3 = boto3.client('s3')

    logging.info("Loading historical data for drift check...")
    
    obj = s3.get_object(Bucket=args.bucket, Key=args.data_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

    yesterday = datetime.now() - timedelta(days=1)
    start_of_yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_yesterday = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

    df_new = df[(df['timestamp'] >= start_of_yesterday) & (df['timestamp'] <= end_of_yesterday)]
    df_baseline = df[df['timestamp'] < start_of_yesterday]

    if df_new.empty or df_baseline.empty:
        logging.warning("Not enough data to perform drift check. Skipping.")
        print('no_drift')
        return

    ks_statistic, p_value = ks_2samp(df_baseline['car_count'].dropna(), df_new['car_count'].dropna())
    logging.info(f"Kolmogorov-Smirnov test: KS Statistic={ks_statistic:.4f}, P-value={p_value:.4f}")

    if p_value < args.p_value_threshold:
        logging.warning(f"Drift DETECTED (p-value {p_value:.4f} < {args.p_value_threshold}).")
        print('trigger_retrain')
    else:
        logging.info("No significant drift detected.")
        print('no_drift')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data drift detector for parking data.")

    parser.add_argument('--bucket', default=os.getenv('S3_BUCKET', 'kltn-smart-parking-data'))
    parser.add_argument('--data-key', default='parking_data/parking_data.csv')
    parser.add_argument('--p-value-threshold', type=float, default=0.05)
    
    args = parser.parse_args()
    check_drift(args)
