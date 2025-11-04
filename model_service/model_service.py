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
import tarfile  
import time     


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo client CloudWatch
cloudwatch = boto3.client('cloudwatch') 

# --- Cấu hình ---
S3_BUCKET = os.getenv('S3_BUCKET', 'kltn-smart-parking-data') 
DATA_KEY = 'parking_data/parking_data.csv'
# Lấy tên Nhóm Model Registry từ biến môi trường
MODEL_PACKAGE_GROUP_NAME = os.getenv('MODEL_PACKAGE_GROUP_NAME', 'SmartParkingModelGroup')

# Khởi tạo S3 và SageMaker clients
s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')

# --- Tải model từ Model Registry ---
def load_latest_approved_model(model_group_name):
    """
    Truy vấn SageMaker Model Registry, tìm model 'Approved' mới nhất,
    tải file .tar.gz từ S3, giải nén và load model + scalers vào bộ nhớ.
    """
    try:
        logger.info(f"Đang truy vấn Model Registry cho nhóm: {model_group_name}")
        
        # 1. Tìm model package 'Approved' mới nhất
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_group_name,
            ModelApprovalStatus='Approved', 
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        if not response['ModelPackageSummaryList']:
            logger.error(f"KHÔNG TÌM THẤY model 'Approved' nào trong nhóm {model_group_name}!")
            raise ValueError("Không có model 'Approved' nào trong Model Registry.")

        latest_approved_package = response['ModelPackageSummaryList'][0]
        model_package_arn = latest_approved_package['ModelPackageArn']
        version = latest_approved_package['ModelPackageVersion']
        
        logger.info(f"Tìm thấy model 'Approved' mới nhất: Version {version}, ARN: {model_package_arn}")

        # 2. Lấy đường dẫn S3 của artifact (model.tar.gz)
        desc_response = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
        
        # Đường dẫn này là 's3://bucket/path/model.tar.gz'
        s3_artifact_path = desc_response['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        
        if not s3_artifact_path.startswith('s3://'):
             raise ValueError(f"Đường dẫn artifact không hợp lệ: {s3_artifact_path}")
             
        # Tách bucket và key
        s3_path_parts = s3_artifact_path.replace('s3://', '').split('/', 1)
        artifact_bucket = s3_path_parts[0]
        artifact_key = s3_path_parts[1]

        logger.info(f"Đang tải artifact từ: s3://{artifact_bucket}/{artifact_key}")

        # 3. Tải file model.tar.gz về thư mục tạm
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_tar_path = os.path.join(tmp_dir, 'model.tar.gz')
            s3_client.download_file(artifact_bucket, artifact_key, local_tar_path)
            
            logger.info("Tải artifact thành công. Đang giải nén...")
            
            # 4. Giải nén file .tar.gz
            extract_path = os.path.join(tmp_dir, 'model_artifacts')
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(local_tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
                
            logger.info(f"Giải nén vào: {extract_path}")
            
            # 5. Load model và scalers từ các file đã giải nén
            model_path = os.path.join(extract_path, 'best_cnn_lstm_model.keras')
            scaler_car_path = os.path.join(extract_path, 'scaler_car_count.pkl')
            scaler_hour_path = os.path.join(extract_path, 'scaler_hour.pkl')
            
            # Kiểm tra xem các file có tồn tại không
            if not all(os.path.exists(p) for p in [model_path, scaler_car_path, scaler_hour_path]):
                logger.error(f"Một hoặc nhiều file artifact bị thiếu trong model.tar.gz! Các file trong thư mục: {os.listdir(extract_path)}")
                raise FileNotFoundError("File model/scaler bị thiếu trong model.tar.gz")

            model = load_model(model_path, compile=False)
            scaler_car = joblib.load(scaler_car_path)
            scaler_hour = joblib.load(scaler_hour_path)
            
            logger.info(f"Đã load thành công model version {version}.")
            
            return model, scaler_car, scaler_hour
            
    except Exception as e:
        logger.critical(f"LỖI NGHIÊM TRỌNG khi tải model từ Registry: {e}")
        # Nếu không tải được model, service không nên chạy
        raise

# --- Cập nhật logic khởi động ---
# Tải model và scaler khi ứng dụng khởi động
logger.info("--- Tải mô hình 'PRODUCTION' (Approved) từ SageMaker Model Registry ---")
model, scaler_car_count, scaler_hour = load_latest_approved_model(MODEL_PACKAGE_GROUP_NAME)
logger.info("--- Tải mô hình 'PRODUCTION' hoàn tất. Service sẵn sàng. ---")

# --- Hàm preprocess_for_prediction ---
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
        
        # Gửi metrics lên CloudWatch
        latency_ms = (time.time() - start_time) * 1000 
        
        try:
            cloudwatch.put_metric_data(
                Namespace='SmartParkingApp', # Tên Namespace trong CloudWatch
                MetricData=[
                    { 
                        'MetricName': 'PredictedCarCount', 
                        'Value': prediction, 
                        'Unit': 'Count' 
                    },
                    { 
                        'MetricName': 'PredictionLatency', 
                        'Value': latency_ms, 
                        'Unit': 'Milliseconds' 
                    }
                ]
            )
        except Exception as metric_error:
            logger.warning(f"Không thể gửi metrics lên CloudWatch: {metric_error}")

        return jsonify({
            "predicted_car_count": prediction,
            "for_timestamp": pred_timestamp.strftime('%d/%m/%Y %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

