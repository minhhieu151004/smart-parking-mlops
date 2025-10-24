from flask import Flask, request, jsonify
import boto3
import pandas as pd
from io import StringIO
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import logging
from datetime import timedelta
import os
import tempfile
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)
prediction_gauge = metrics.info('parking_prediction_value', 'Giá trị dự đoán số lượng xe')
prediction_latency = metrics.histogram('parking_prediction_latency_seconds', 'Độ trễ của API dự đoán')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cấu hình kết nối MinIO ---
S3_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
S3_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'admin')
S3_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'password')
S3_BUCKET = os.getenv('S3_BUCKET', 'my-bucket')
DATA_KEY = 'parking_data/parking_data.csv'
PRODUCTION_MODEL_PATH = 'models/production'
MODEL_KEY = f'{PRODUCTION_MODEL_PATH}/best_cnn_lstm_model.keras'
SCALER_CAR_KEY = f'{PRODUCTION_MODEL_PATH}/scaler_car_count.pkl'
SCALER_HOUR_KEY = f'{PRODUCTION_MODEL_PATH}/scaler_hour.pkl'

s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# --- Hàm tải artifact từ MinIO ---
def load_artifact_from_minio(key, artifact_type):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        artifact_data = response['Body'].read()
        suffix = '.keras' if artifact_type == 'model' else '.pkl'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(artifact_data)
            tmp_file_path = tmp_file.name
        if artifact_type == 'model':
            artifact = load_model(tmp_file_path, compile=False)
        else:
            artifact = joblib.load(tmp_file_path)
        os.unlink(tmp_file_path)
        logger.info(f"Loaded {artifact_type} from MinIO: s3://{S3_BUCKET}/{key}")
        return artifact
    except Exception as e:
        logger.error(f"Error loading {artifact_type} from MinIO (s3://{S3_BUCKET}/{key}): {e}")
        logger.critical(f"FATAL: Could not load production {artifact_type}. Exiting.")
        raise

# Tải model và scaler khi ứng dụng khởi động
logger.info("--- Tải mô hình 'PRODUCTION' khi khởi động ---")
model = load_artifact_from_minio(MODEL_KEY, 'model')
scaler_car_count = load_artifact_from_minio(SCALER_CAR_KEY, 'scaler_car')
scaler_hour = load_artifact_from_minio(SCALER_HOUR_KEY, 'scaler_hour')
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

@app.route('/trigger_predict', methods=['POST'])
@prediction_latency  # Sử dụng decorator để đo độ trễ
def trigger_predict():
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=DATA_KEY)
        df_history = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        input_sequence = preprocess_for_prediction(df_history)
        prediction_scaled = model.predict(input_sequence, verbose=0)[0][0]
        prediction_actual = scaler_car_count.inverse_transform([[prediction_scaled]])[0][0]
        prediction = int(round(prediction_actual))
        prediction_gauge.set(prediction)
        pred_timestamp = pd.to_datetime(df_history['timestamp'].max(), dayfirst=True) + timedelta(hours=1)
        logger.info(f"Predicted car_count for {pred_timestamp}: {prediction}")
        return jsonify({
            "predicted_car_count": prediction,
            "for_timestamp": pred_timestamp.strftime('%d/%m/%Y %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #test
