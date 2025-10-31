import json
import os
import boto3
import logging
import pandas as pd
import numpy as np
import joblib
import tarfile
import tempfile
import time
from io import StringIO, BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# --- Cấu hình ---
app = Flask(__name__)

# Cấu hình Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Khởi tạo Clients ---
s3_client = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

# --- Lấy Biến Môi trường ---
S3_BUCKET = os.getenv('S3_BUCKET', 'kltn-smart-parking-data') 
DATA_KEY = 'parking_data/parking_data.csv'
PRODUCTION_MODEL_PATH = 'models/production/'
METRIC_NAMESPACE = 'SmartParkingApp' # Namespace cho CloudWatch Metrics

# --- Hàm tải artifact từ S3 ---
def load_artifact_from_s3(key, artifact_type):
    """
    Tải artifact (model, scaler) từ thư mục production S3.
    """
    s3_key = f"{PRODUCTION_MODEL_PATH}{key}"
    try:
        # Tạo một file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=key) as tmp_file:
            logger.info(f"Downloading {artifact_type} from s3://{S3_BUCKET}/{s3_key}...")
            s3_client.download_file(S3_BUCKET, s3_key, tmp_file.name)
            tmp_file_path = tmp_file.name
        
        # Tải artifact từ file tạm
        if artifact_type == 'model':
            artifact = load_model(tmp_file_path, compile=False)
        else: # scaler
            artifact = joblib.load(tmp_file_path)
            
        os.unlink(tmp_file_path) # Xóa file tạm
        logger.info(f"Loaded {artifact_type} successfully.")
        return artifact
        
    except Exception as e:
        logger.error(f"Error loading {artifact_type} from S3 (s3://{S3_BUCKET}/{s3_key}): {e}")
        logger.critical(f"FATAL: Could not load production {artifact_type}. Exiting.")
        raise

# --- Tải Model khi ứng dụng khởi động ---
logger.info("--- Tải mô hình 'PRODUCTION' khi khởi động ---")
model = load_artifact_from_s3('best_cnn_lstm_model.keras', 'model')
scaler_car_count = load_artifact_from_s3('scaler_car_count.pkl', 'scaler_car')
scaler_hour = load_artifact_from_s3('scaler_hour.pkl', 'scaler_hour')
logger.info("--- Tải mô hình 'PRODUCTION' hoàn tất. Service sẵn sàng. ---")

def preprocess_for_prediction(df, n_steps=288):
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    df_resampled = df.resample('5min').mean().interpolate()
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['car_count_scaled'] = scaler_car_count.transform(df_resampled[['car_count']])
    df_resampled['hour_scaled'] = scaler_hour.transform(df_resampled[['hour']])
    sequence = df_resampled[['car_count_scaled', 'hour_scaled']].values[-n_steps:]
    if len(sequence) < n_steps:
        raise ValueError(f"Không đủ dữ liệu, cần ít nhất {n_steps} điểm dữ liệu (5-phút).")
    return sequence.reshape(1, n_steps, 2)

def send_cloudwatch_metric(metric_name, value, unit):
    """
    Gửi một custom metric đến CloudWatch.
    """
    try:
        cloudwatch.put_metric_data(
            Namespace=METRIC_NAMESPACE,
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit
                },
            ]
        )
    except Exception as e:
        # Ghi log lỗi 
        logger.warning(f"Failed to send metric '{metric_name}' to CloudWatch: {e}")

@app.route('/trigger_predict', methods=['POST'])
def trigger_predict():
    start_time = time.time() # Bắt đầu đo thời gian
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=DATA_KEY)
        df_history = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        
        input_sequence = preprocess_for_prediction(df_history)
        
        prediction_scaled = model.predict(input_sequence, verbose=0)[0][0]
        prediction_actual = scaler_car_count.inverse_transform([[prediction_scaled]])[0][0]
        prediction = int(round(prediction_actual))
        
        pred_timestamp = pd.to_datetime(df_history['timestamp'].max(), dayfirst=True) + timedelta(hours=1)
        logger.info(f"Predicted car_count for {pred_timestamp}: {prediction}")
        
        # --- Gửi Metrics lên CloudWatch ---
        latency_ms = (time.time() - start_time) * 1000
        send_cloudwatch_metric(metric_name='PredictedCarCount', value=prediction, unit='Count')
        send_cloudwatch_metric(metric_name='PredictionLatency', value=latency_ms, unit='Milliseconds')
        
        return jsonify({
            "predicted_car_count": prediction,
            "for_timestamp": pred_timestamp.strftime('%d/%m/%Y %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        # Gửi metric lỗi
        send_cloudwatch_metric(metric_name='PredictionError', value=1, unit='Count')
        return jsonify({"error": str(e)}), 500

# Endpoint kiểm tra sức khỏe (health check)
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

