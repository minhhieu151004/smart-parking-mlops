import json
import os
import boto3
import logging
import tarfile
import requests 
from io import BytesIO

# --- Cấu hình Logging ---
# (Logger sẽ tự động ghi vào CloudWatch Logs)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients (bên ngoài handler để tái sử dụng) ---
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# --- Lấy Biến Môi trường ---
try:
    S3_BUCKET = os.environ['S3_BUCKET']
    GITHUB_REPO = os.environ['GITHUB_REPO'] 
    GITHUB_PAT = os.environ['GITHUB_PAT']  
except KeyError as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Biến môi trường {e} chưa được set!")
    raise

# --- Hằng số ---
PRODUCTION_METRICS_KEY = "models/production/metrics.json"
PRODUCTION_MODEL_PATH = "models/production/" # Thư mục S3

def lambda_handler(event, context):
    """
    Hàm Lambda chính được trigger bởi EventBridge khi SageMaker Pipeline 'Succeeded'.
    """
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        # 1. Lấy thông tin từ sự kiện (event)
        pipeline_execution_arn = event['detail']['pipelineExecutionArn']
        # Lấy ID thực thi
        execution_id = pipeline_execution_arn.split('/')[-1] 
        
        logger.info(f"Đang xử lý Pipeline Execution ID: {execution_id}")

        # === 2. Kiểm tra kết quả Drift Check ===
        # Dựa theo 'build_pipeline.py', đường dẫn output của drift check là:
        drift_result_key = f"pipeline-outputs/drift-check/{execution_id}/drift_check_result.json"
        
        try:
            drift_result_obj = s3.get_object(Bucket=S3_BUCKET, Key=drift_result_key)
            drift_data = json.loads(drift_result_obj['Body'].read().decode('utf-8'))
            drift_detected = drift_data.get("drift_detected", False)
        except s3.exceptions.NoSuchKey:
            logger.warning(f"Không tìm thấy file drift check: {drift_result_key}. Bỏ qua...")
            return {"statusCode": 200, "body": "Không tìm thấy file drift, bỏ qua."}

        if not drift_detected:
            logger.info("Không phát hiện drift (drift_detected=false). Không cần làm gì thêm.")
            return {"statusCode": 200, "body": "Không có drift, không huấn luyện."}

        logger.info("Phát hiện Drift. Đang tiến hành đánh giá mô hình mới...")

        # === 3. Lấy và So sánh Metrics (Nếu có Drift) ===
        
        # Dựa theo 'build_pipeline.py', model mới (model.tar.gz) được lưu tại:
        model_artifact_key = f"pipeline-outputs/training-output/{execution_id}/output/model.tar.gz"
        
        # Tải và giải nén model.tar.gz từ S3
        logger.info(f"Đang tải model artifact: {model_artifact_key}")
        s3_artifact = s3.get_object(Bucket=S3_BUCKET, Key=model_artifact_key)
        
        # Giải nén file tar.gz trong bộ nhớ (hoặc /tmp)
        # Script train của bạn lưu 'metrics.json' trong file tar.gz này
        new_metrics = None
        extracted_artifacts = {} # Dict để lưu {tên file: nội dung file}

        with tarfile.open(fileobj=BytesIO(s3_artifact['Body'].read()), mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    file_content = tar.extractfile(member).read()
                    file_name = os.path.basename(member.name)
                    extracted_artifacts[file_name] = file_content
                    
                    if file_name == 'metrics.json':
                        new_metrics = json.loads(file_content.decode('utf-8'))
                        
        if new_metrics is None:
            raise ValueError("Không tìm thấy 'metrics.json' trong model.tar.gz")

        new_loss = new_metrics.get('val_loss')
        logger.info(f"Model mới có val_loss: {new_loss}")

        # Lấy metrics của model 'production' hiện tại
        current_prod_loss = float('inf')
        try:
            prod_metrics_obj = s3.get_object(Bucket=S3_BUCKET, Key=PRODUCTION_METRICS_KEY)
            current_prod_loss = json.loads(prod_metrics_obj['Body'].read().decode('utf-8')).get('val_loss', float('inf'))
            logger.info(f"Model 'production' hiện tại có val_loss: {current_prod_loss}")
        except s3.exceptions.NoSuchKey:
            logger.warning(f"Không tìm thấy 'production' metrics. Model mới sẽ được tự động thúc đẩy.")

        # === 4. Quyết định Thúc đẩy (Promote) ===
        if new_loss < current_prod_loss:
            logger.warning(f"THÚC ĐẨY: Model mới (loss={new_loss}) tốt hơn 'production' (loss={current_prod_loss}).")
            
            # Upload tất cả artifact đã giải nén lên thư mục 'production'
            for file_name, file_content in extracted_artifacts.items():
                prod_key = f"{PRODUCTION_MODEL_PATH}{file_name}"
                s3.put_object(Bucket=S3_BUCKET, Key=prod_key, Body=file_content)
                logger.info(f"Đã upload {file_name} lên {prod_key}")
            
            logger.info("Thúc đẩy model hoàn tất.")
            
            # === 5. Trigger GitHub Actions (Restart Service) ===
            trigger_github_action_restart()
            
            return {"statusCode": 200, "body": "Model đã được thúc đẩy và GHA trigger."}
        else:
            logger.info(f"BỎ QUA: Model mới (loss={new_loss}) không tốt hơn 'production' (loss={current_prod_loss}).")
            return {"statusCode": 200, "body": "Model mới không tốt hơn, không thúc đẩy."}

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi Lambda: {e}")
        # Gửi thông báo lỗi (ví dụ: SNS) nếu cần
        raise e # Raise lỗi để EventBridge biết và (có thể) retry

def trigger_github_action_restart():
    """
    Gửi HTTP POST request đến GitHub API để trigger workflow 'mlops-jobs.yml'.
    """
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_PAT}"
    }
    payload = {
        "event_type": "trigger-restart" 
    }
    
    logger.info(f"Đang trigger GitHub Action: {api_url} với event: {payload['event_type']}")
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status() # Raise lỗi 
        logger.info(f"GitHub API response status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi gọi GitHub API: {e}")
        
