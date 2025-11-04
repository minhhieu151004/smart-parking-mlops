import json
import os
import boto3
import requests
import logging
from io import StringIO
from datetime import datetime

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
s3 = boto3.client('s3')

# --- Lấy Biến Môi trường (Bạn phải set các biến này trên AWS Lambda) ---
try:
    # URL đầy đủ của service API (ví dụ: http://54.12.34.56:5000/trigger_predict)
    MODEL_SERVICE_URL = os.environ['MODEL_SERVICE_URL']
    # Tên S3 Bucket
    S3_BUCKET = os.environ['S3_BUCKET']
    # Thư mục để ghi dự đoán (ví dụ: 'daily_predictions/')
    PREDICTION_PREFIX = os.environ['PREDICTION_PREFIX'] 
except KeyError as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Biến môi trường {e} chưa được set!")
    raise

def lambda_handler(event, context):
    """
    Hàm Lambda này được trigger bởi S3 Event khi Pi upload file vào 'daily_actuals/'.
    Nó sẽ gọi model service và ghi kết quả dự đoán vào 'daily_predictions/'.
    """
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        # 1. Lấy thông tin file đã trigger sự kiện (file 'actuals' từ Pi)
        # Chúng ta dùng thông tin này để quyết định tên file 'predictions'
        triggering_bucket = event['Records'][0]['s3']['bucket']['name']
        triggering_key = event['Records'][0]['s3']['object']['key'] # (ví dụ: 'daily_actuals/2025-10-31.csv')
        
        # Chỉ xử lý nếu S3_BUCKET khớp và là file CSV
        if triggering_bucket != S3_BUCKET or not triggering_key.endswith('.csv'):
            logger.warning("Event không đúng bucket hoặc không phải file CSV. Bỏ qua.")
            return

        file_name = os.path.basename(triggering_key) # (ví dụ: '2025-10-31.csv')
        prediction_key = f"{PREDICTION_PREFIX}{file_name}" # (ví dụ: 'daily_predictions/2025-10-31.csv')

        logger.info(f"Đã phát hiện thay đổi trên {triggering_key}. Đang gọi Model Service...")

        # === 2. Gọi Model Service API (trên EC2) ===
        try:
            response = requests.post(MODEL_SERVICE_URL, timeout=10) # 10 giây timeout
            response.raise_for_status() # Báo lỗi nếu status code là 4xx/5xx
            prediction_data = response.json()
            # prediction_data có dạng: {"predicted_car_count": 123, "for_timestamp": "..."}
            
            logger.info(f"Model Service trả về: {prediction_data}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi gọi Model Service tại {MODEL_SERVICE_URL}: {e}")
            raise Exception("Không thể gọi Model Service.") from e

        # === 3. Ghi/Nối (Append) kết quả vào S3 ===
        # S3 không hỗ trợ "nối" file. Chúng ta phải: Đọc -> Sửa -> Ghi đè.
        
        file_content = ""
        header = "timestamp,predicted_car_count\n"
        new_line = f"{prediction_data['for_timestamp']},{prediction_data['predicted_car_count']}\n"

        try:
            # Thử đọc file dự đoán cũ (nếu đã tồn tại)
            obj = s3.get_object(Bucket=S3_BUCKET, Key=prediction_key)
            file_content = obj['Body'].read().decode('utf-8')
            logger.info(f"Đã tìm thấy file dự đoán cũ: {prediction_key}. Đang nối...")
            
            # Đảm bảo file cũ có header
            if not file_content.startswith(header.strip()):
                file_content = header + file_content # Thêm header nếu thiếu
                
        except s3.exceptions.NoSuchKey:
            # File chưa tồn tại (lần dự đoán đầu tiên trong ngày)
            logger.info(f"Không tìm thấy file dự đoán cũ. Đang tạo file mới: {prediction_key}")
            file_content = header # Bắt đầu file mới với header
        
        # Nối dòng dự đoán mới
        updated_content = file_content + new_line
        
        # Ghi đè file lên S3
        s3.put_object(Bucket=S3_BUCKET, Key=prediction_key, Body=updated_content.encode('utf-8'))
        
        logger.info(f"Đã ghi dự đoán thành công vào: s3://{S3_BUCKET}/{prediction_key}")

        return {
            'statusCode': 200,
            'body': json.dumps(f"Đã xử lý dự đoán và lưu vào {prediction_key}")
        }

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi Lambda: {e}")
        raise e
