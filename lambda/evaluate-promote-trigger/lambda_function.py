import json
import os
import boto3
import logging

# --- Cấu hình Logging ---
# (Logger sẽ tự động ghi vào CloudWatch Logs)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
# Boto3 sẽ tự động sử dụng IAM Role của Lambda
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# --- Lấy Biến Môi trường (Bạn phải set các biến này trên AWS Lambda) ---
try:
    S3_BUCKET = os.environ['S3_BUCKET']
    MODEL_PACKAGE_GROUP_NAME = os.environ['MODEL_PACKAGE_GROUP_NAME']
    SAGEMAKER_ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME'] # (smart-parking-predictor')
except KeyError as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Biến môi trường {e} chưa được set!")
    raise

def lambda_handler(event, context):
    """
    Được trigger khi SageMaker Pipeline 'Succeeded'.
    Nhiệm vụ: Tìm model 'Pending', 'Approve' nó, và 'Deploy/Update' Endpoint.
    """
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        # 1. Lấy thông tin từ sự kiện (event)
        pipeline_execution_arn = event['detail']['pipelineExecutionArn']
        # Lấy ID thực thi
        execution_id = pipeline_execution_arn.split('/')[-1] 
        logger.info(f"Đang xử lý Pipeline Execution ID: {execution_id}")

        # === 2. Kiểm tra kết quả Drift Check ===
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
        
        # === 5. Deploy/Update Endpoint ===
        logger.info(f"Bắt đầu quá trình deploy/update cho Endpoint: {SAGEMAKER_ENDPOINT_NAME}")
        
        # 5.1 Lấy thông tin model package vừa được approve
        model_package_desc = sagemaker.describe_model_package(ModelPackageName=model_package_arn)
        model_data_url = model_package_desc['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        image_uri = model_package_desc['InferenceSpecification']['Containers'][0]['Image']
        
        # 5.2 Tạo một Model (từ Model Package)
        model_name = f"model-{execution_id}" # Tên model duy nhất
        logger.info(f"Đang tạo Model: {model_name}")
        
        # Lấy IAM Role của Lambda (để SageMaker dùng)
        # Cần đảm bảo Role này có quyền 'sagemaker:CreateModel'
        lambda_role_arn = context.invoked_function_arn.split(":")[-2] # Lấy role ARN của chính hàm Lambda
        execution_role_arn = f"arn:aws:iam::{lambda_role_arn.split(':')[4]}:role/{os.environ['AWS_LAMBDA_EXECUTION_ROLE']}"
        
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
            },
            ExecutionRoleArn=execution_role_arn 
        )

        # 5.3 Tạo Endpoint Config mới
        endpoint_config_name = f"epc-{execution_id}" # Tên config duy nhất
        logger.info(f"Đang tạo Endpoint Config: {endpoint_config_name}")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'ServerlessConfig': {
                        'MemorySizeInMB': 2048, # Cần đủ RAM cho TensorFlow
                        'MaxConcurrency': 10
                    }
                    # (HOẶC) Sử dụng Real-time
                    # 'InstanceType': 'ml.m5.large' 
                }
            ]
        )

        # 5.4 Kiểm tra xem Endpoint đã tồn tại chưa
        try:
            sagemaker.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
            # Nếu tồn tại -> Cập nhật (Update)
            logger.info(f"Endpoint {SAGEMAKER_ENDPOINT_NAME} đã tồn tại. Đang cập nhật...")
            sagemaker.update_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            logger.info("Update Endpoint thành công.")
            
        except sagemaker.exceptions.ClientError:
            # Nếu chưa tồn tại -> Tạo mới (Create)
            logger.info(f"Endpoint {SAGEMAKER_ENDPOINT_NAME} chưa tồn tại. Đang tạo mới...")
            sagemaker.create_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            logger.info("Create Endpoint thành công.")
        
        return {"statusCode": 200, "body": f"Model đã được 'Approved' và deploy/update endpoint {SAGEMAKER_ENDPOINT_NAME}."}

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi Lambda: {e}")
        raise e
