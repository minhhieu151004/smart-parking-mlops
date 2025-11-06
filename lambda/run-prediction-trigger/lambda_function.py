import json
import os
import boto3
import logging
from io import StringIO, BytesIO
from datetime import datetime
import pandas as pd

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
s3 = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# --- Lấy Biến Môi trường ---
try:
    ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
    S3_BUCKET = os.environ['S3_BUCKET']
    PREDICTION_PREFIX = os.environ['PREDICTION_PREFIX'] 
    DATA_KEY = os.environ.get('DATA_KEY', 'parking_data/parking_data.csv') 
except KeyError as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Biến môi trường {e} chưa được set!")
    raise

def lambda_handler(event, context):
    """
    Trigger bởi S3 (khi Pi upload).
    Tải (Baseline + Mới), Nối (Concat), Gọi Endpoint, Ghi (Dự đoán).
    """
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        # === 1. Tải Dữ liệu ===
        
        # 1a. Tải file Baseline (master)
        logger.info(f"Đang tải dữ liệu baseline từ: s3://{S3_BUCKET}/{DATA_KEY}")
        try:
            obj_baseline = s3.get_object(Bucket=S3_BUCKET, Key=DATA_KEY)
            df_baseline = pd.read_csv(StringIO(obj_baseline['Body'].read().decode('utf-8')))
        except Exception as e:
            logger.error(f"Không thể tải file baseline {DATA_KEY}: {e}")
            raise
            
        # 1b. Tải file Mới (từ S3 event)
        try:
            triggering_bucket = event['Records'][0]['s3']['bucket']['name']
            triggering_key = event['Records'][0]['s3']['object']['key'] # (ví dụ: 'daily_actuals/2025-10-31.csv')
            
            if triggering_bucket != S3_BUCKET:
                raise ValueError("Event từ bucket không mong muốn.")
                
            logger.info(f"Đang tải dữ liệu mới từ (trigger): s3://{triggering_bucket}/{triggering_key}")
            obj_new = s3.get_object(Bucket=triggering_bucket, Key=triggering_key)
            df_new = pd.read_csv(StringIO(obj_new['Body'].read().decode('utf-8')))
            
        except (KeyError, IndexError, TypeError, s3.exceptions.NoSuchKey) as e:
            logger.error(f"Không thể lấy file mới từ S3 event: {e}. Hủy bỏ.")
            return {'statusCode': 400, 'body': 'Lỗi xử lý S3 event.'}
            
        # 1c. Nối (Combine) hai file
        logger.info(f"Nối {len(df_baseline)} dòng (baseline) với {len(df_new)} dòng (mới).")
        df_combined = pd.concat([df_baseline, df_new], ignore_index=True)
        
        # Chuyển đổi DataFrame kết hợp về dạng CSV (bytes)
        combined_csv_buffer = StringIO()
        df_combined.to_csv(combined_csv_buffer, index=False)
        combined_csv_bytes = combined_csv_buffer.getvalue().encode('utf-8')

        # === 2. Gọi SageMaker Endpoint ===
        logger.info(f"Đang gọi SageMaker Endpoint: {ENDPOINT_NAME}")
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='text/csv',
                Body=combined_csv_bytes # Gửi CSV đã nối
            )
            result_body = response['Body'].read().decode('utf-8')
            prediction_data = json.loads(result_body)
            # prediction_data có dạng: {"predicted_car_count": 88, "for_timestamp": "..."}
            
            logger.info(f"SageMaker Endpoint trả về: {prediction_data}")

        except Exception as e:
            logger.error(f"Lỗi khi gọi SageMaker Endpoint {ENDPOINT_NAME}: {e}")
            raise

        # === 3. Ghi/Nối (Append) kết quả vào S3 ===
        
        # Quyết định file output (ví dụ: 'daily_predictions/2025-10-31.csv')
        file_name = os.path.basename(triggering_key) 
        prediction_key = f"{PREDICTION_PREFIX}{file_name}"
        
        file_content = ""
        header = "timestamp,predicted_car_count\n"
        
        new_line = f"{prediction_data['for_timestamp']},{prediction_data['predicted_car_count']}\n"

        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=prediction_key)
            file_content = obj['Body'].read().decode('utf-8')
            if not file_content.startswith(header.strip()):
                file_content = header + file_content
        except s3.exceptions.NoSuchKey:
            logger.info(f"Tạo file dự đoán mới: {prediction_key}")
            file_content = header
        
        updated_content = file_content.rstrip('\n') + '\n' + new_line
        
        s3.put_object(Bucket=S3_BUCKET, Key=prediction_key, Body=updated_content.encode('utf-8'))
        
        logger.info(f"Đã ghi dự đoán thành công vào: s3://{S3_BUCKET}/{prediction_key}")

        return {'statusCode': 200}

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi Lambda: {e}")
        raise e
