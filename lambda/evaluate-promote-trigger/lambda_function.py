import json
import os
import boto3
import logging
import time
from botocore.exceptions import ClientError

# --- CẤU HÌNH ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# Lấy biến môi trường
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'smart-parking-endpoint')
ROLE_ARN = os.environ.get('SAGEMAKER_EXECUTION_ROLE_ARN')

# --- LẤY REPORT TỪ MODEL REGISTRY ---
def get_evaluation_report(model_package_arn):
    """
    Tải file evaluation.json từ Model Registry để đọc kết quả so sánh.
    """
    try:
        # 1. Lấy thông tin Model Package
        package_desc = sagemaker.describe_model_package(ModelPackageName=model_package_arn)
        
        # In ra cấu trúc ModelMetrics để kiểm tra nếu lỗi
        metrics_data = package_desc.get('ModelMetrics', {})
        logger.info(f"🔍 Raw ModelMetrics: {json.dumps(metrics_data, default=str)}")

        # 2. Tìm đường dẫn S3 của file metrics
        metrics_s3_uri = None
        
        if 'ModelQuality' in metrics_data:
            metrics_s3_uri = metrics_data['ModelQuality']['Statistics']['S3Uri']
        elif 'ModelDataQuality' in metrics_data:
             metrics_s3_uri = metrics_data['ModelDataQuality']['Statistics']['S3Uri']
        elif 'ModelStatistics' in metrics_data: # Legacy support
             metrics_s3_uri = metrics_data['ModelStatistics']['S3Uri']
             
        if not metrics_s3_uri:
            logger.warning(f"⚠️ Model {model_package_arn} không có ModelMetrics (ModelQuality/ModelDataQuality).")
            return None
        
        # 3. Xử lý đường dẫn Folder -> File
        if not metrics_s3_uri.endswith('.json'):
            metrics_s3_uri = f"{metrics_s3_uri}/evaluation.json"
            
        logger.info(f"🎯 Full S3 URI: {metrics_s3_uri}")

        # 4. Parse S3 URI (s3://bucket/key)
        metrics_s3_path_parts = metrics_s3_uri.replace('s3://', '').split('/', 1)
        bucket = metrics_s3_path_parts[0]
        key = metrics_s3_path_parts[1]
        
        # 5. Tải file
        logger.info(f"📥 Đang tải report từ Bucket: {bucket}, Key: {key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        report_data = json.loads(obj['Body'].read().decode('utf-8'))
        
        return report_data
            
    except Exception as e:
        logger.error(f"❌ Lỗi lấy report từ {model_package_arn}: {e}")
        return None

# --- HÀM DEPLOY ---
def deploy_model_to_endpoint(model_package_arn):
    """Deploy model mới lên Endpoint."""
    try:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        model_name = f"parking-model-{timestamp}"
        endpoint_config_name = f"parking-config-{timestamp}"

        # 1. Tạo Model Object
        logger.info(f"🚀 Deploying: Tạo Model '{model_name}'")
        sagemaker.create_model(
            ModelName=model_name,
            ExecutionRoleArn=ROLE_ARN, 
            Containers=[{'ModelPackageName': model_package_arn}]
        )

        # 2. Tạo Endpoint Config (Serverless)
        logger.info(f"🚀 Deploying: Tạo Config '{endpoint_config_name}'")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'ServerlessConfig': {
                    'MemorySizeInMB': 2048,
                    'MaxConcurrency': 5
                }
            }]
        )

        # 3. Cập nhật Endpoint
        logger.info(f"🚀 Deploying: Update Endpoint '{ENDPOINT_NAME}'")
        try:
            sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            # Nếu tồn tại -> Update
            sagemaker.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
        except ClientError:
            # Nếu chưa tồn tại -> Create
            logger.info("Endpoint chưa tồn tại -> Tạo mới.")
            sagemaker.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            
        return True

    except Exception as e:
        logger.error(f"❌ Lỗi Deploy: {e}")
        raise e

# --- HÀM XỬ LÝ CHÍNH ---
def lambda_handler(event, context):
    logger.info(f"Event Received: {json.dumps(event)}")

    try:
        # 1. Lấy ARN của Model vừa được đăng ký
        model_package_arn = event.get('detail', {}).get('ModelPackageArn')
        
        if not model_package_arn:
             logger.error("Không tìm thấy ModelPackageArn trong event.")
             return {"statusCode": 400, "body": "Invalid Event"}
        
        # 2. Đọc file Evaluation Report 
        report = get_evaluation_report(model_package_arn)
        
        if not report:
            logger.error("Không tìm thấy report hoặc lỗi tải report. Reject model.")
            sagemaker.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription="Error: Report not found or invalid S3 path."
            )
            return {"statusCode": 400, "body": "Report Not Found"}

        # 3. Kiểm tra kết quả so sánh (BETTER / WORSE)
        comparison_result = report.get('comparison', {}).get('result', 'UNKNOWN')
        new_mae = report.get('comparison', {}).get('new_mae', 'N/A')
        old_mae = report.get('comparison', {}).get('old_mae', 'N/A')
        
        logger.info(f"📊 KẾT QUẢ SO SÁNH: {comparison_result} (New: {new_mae} vs Old: {old_mae})")

        # 4. Ra quyết định
        if comparison_result == "BETTER":
            # === APPROVE & DEPLOY ===
            logger.info("✅ Model MỚI TỐT HƠN -> TIẾN HÀNH DEPLOY.")
            print(f"DEPLOY_SIGNAL: APPROVED")

            sagemaker.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus='Approved',
                ApprovalDescription=f"Auto-approved: Better Performance ({new_mae} < {old_mae})"
            )
            
            deploy_model_to_endpoint(model_package_arn)
            
            return {"statusCode": 200, "body": "Model Approved & Deployed"}
            
        else:
            # === REJECT ===
            logger.info("❌ Model MỚI TỆ HƠN (HOẶC BẰNG) -> TỪ CHỐI.")
            print(f"DEPLOY_SIGNAL: REJECTED")

            sagemaker.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription=f"Auto-rejected: Worse Performance ({new_mae} >= {old_mae})"
            )
            
            return {"statusCode": 200, "body": "Model Rejected"}

    except Exception as e:
        logger.error(f"Critical Error: {e}")
        raise e