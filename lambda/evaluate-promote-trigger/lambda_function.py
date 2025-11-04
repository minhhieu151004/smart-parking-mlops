import json
import os
import boto3
import logging
import requests 

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# --- Lấy Biến Môi trường ---
try:
    S3_BUCKET = os.environ['S3_BUCKET']
    GITHUB_REPO = os.environ['GITHUB_REPO'] 
    GITHUB_PAT = os.environ['GITHUB_PAT']    
    # Tên nhóm model package
    MODEL_PACKAGE_GROUP_NAME = os.environ['MODEL_PACKAGE_GROUP_NAME']
except KeyError as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Biến môi trường {e} chưa được set!")
    raise

def lambda_handler(event, context):
    """
    Được trigger khi SageMaker Pipeline 'Succeeded'.
    Nhiệm vụ: Tìm model 'Pending' mới nhất, 'Approve' nó, và trigger GHA.
    """
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        # 1. Lấy thông tin từ sự kiện (event)
        pipeline_execution_arn = event['detail']['pipelineExecutionArn']
        execution_id = pipeline_execution_arn.split('/')[-1] 
        logger.info(f"Đang xử lý Pipeline Execution ID: {execution_id}")

        # === 2. Kiểm tra kết quả Drift Check ===
        # (Vẫn kiểm tra xem có drift không, nếu không thì không làm gì cả)
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

        logger.info("Phát hiện Drift. Đang tiến hành phê duyệt mô hình mới...")

        # === 3. Tìm và Phê duyệt Model Package mới nhất ===
        
        # Tìm các model package 'Pending' (chờ duyệt) trong nhóm
        response = sagemaker.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus='Pending',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        if not response['ModelPackageSummaryList']:
            logger.error("Không tìm thấy model package nào ở trạng thái 'Pending'. Dừng lại.")
            raise ValueError("Không tìm thấy model package 'Pending' để phê duyệt.")

        # Lấy model package mới nhất
        latest_pending_package = response['ModelPackageSummaryList'][0]
        model_package_arn = latest_pending_package['ModelPackageArn']
        
        logger.info(f"Tìm thấy model package 'Pending' mới nhất: {model_package_arn}")

        # === 4. Phê duyệt (Approve) Model ===
        logger.warning(f"Đang CẬP NHẬT trạng thái thành 'Approved' cho: {model_package_arn}")
        sagemaker.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus='Approved'
        )
        logger.info("Phê duyệt model hoàn tất.")

        # === 5. Trigger GitHub Actions (Restart Service) ===
        trigger_github_action_restart()
        
        return {"statusCode": 200, "body": "Model đã được phê duyệt (Approved) và GHA trigger."}

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi Lambda: {e}")
        raise e

def trigger_github_action_restart():
    """
    Gửi HTTP POST request đến GitHub API để trigger workflow 'mlops-jobs.yml'.
    """
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_PAT}"
    }
    payload = { "event_type": "trigger-restart" }
    
    logger.info(f"Đang trigger GitHub Action: {api_url} với event: {payload['event_type']}")
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        logger.info(f"GitHub API response status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi gọi GitHub API: {e}")

