import json
import os
import boto3
import logging
import time
from botocore.exceptions import ClientError

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# --- LẤY BIẾN MÔI TRƯỜNG ---
try:
    MODEL_PACKAGE_GROUP_NAME = os.environ.get('MODEL_PACKAGE_GROUP_NAME', 'SmartParkingModelGroup')
    
    ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME'] 
    
    ROLE_ARN = os.environ['SAGEMAKER_EXECUTION_ROLE_ARN']
    
    logger.info(f"Cấu hình: Endpoint={ENDPOINT_NAME} | Role={ROLE_ARN}")
    
except KeyError as e:
    logger.error(f"LỖI: Thiếu biến môi trường {e}")
    raise RuntimeError(f"Missing environment variable: {e}")

# --- Hàm Helper: Lấy MAE ---
def get_mae_from_model_package(model_package_arn):
    """Lấy MAE từ file metrics.json trong S3 artifact của Model Package"""
    try:
        package_desc = sagemaker.describe_model_package(ModelPackageName=model_package_arn)
        
        # Tìm đường dẫn S3 của file metrics
        try:
            metrics_s3_uri = package_desc['ModelMetrics']['ModelStatistics']['S3Uri']
        except KeyError:
            logger.warning(f"Model {model_package_arn} không có ModelMetrics.")
            return float('inf')
        
        # Parse S3 URI
        metrics_s3_path_parts = metrics_s3_uri.replace('s3://', '').split('/', 1)
        metrics_bucket = metrics_s3_path_parts[0]
        metrics_key = metrics_s3_path_parts[1]
        
        # Tải và đọc file
        metrics_obj = s3.get_object(Bucket=metrics_bucket, Key=metrics_key)
        metrics_data = json.loads(metrics_obj['Body'].read().decode('utf-8'))
        
        # Lấy giá trị MAE
        try:
            mae = metrics_data['regression_metrics']['mae']['value']
            return float(mae)
        except KeyError:
            return float('inf')
            
    except Exception as e:
        logger.error(f"Lỗi lấy metrics từ {model_package_arn}: {e}")
        return float('inf')

# --- Deploy Model ---
def deploy_model_to_endpoint(model_package_arn):
    """Tạo Model, Config và Update Endpoint 'smart-parking-predictor'"""
    try:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        model_name = f"parking-model-{timestamp}"
        endpoint_config_name = f"parking-config-{timestamp}"

        # 1. Tạo Model Object
        logger.info(f"Deploying: Tạo Model Object '{model_name}' từ {model_package_arn}")
        sagemaker.create_model(
            ModelName=model_name,
            ExecutionRoleArn=ROLE_ARN, 
            Containers=[{'ModelPackageName': model_package_arn}]
        )

        # 2. Tạo Endpoint Config (Serverless - Tiết kiệm chi phí)
        logger.info(f"Deploying: Tạo Config '{endpoint_config_name}'")
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
        logger.info(f"Deploying: Cập nhật Endpoint '{ENDPOINT_NAME}'")
        try:
            sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            
            # Nếu tồn tại -> Update
            sagemaker.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            logger.info("✅ Lệnh Update Endpoint đã được gửi thành công.")
            
        except ClientError:
            # Nếu chưa tồn tại -> Create
            logger.info("Endpoint chưa tồn tại -> Đang tạo mới...")
            sagemaker.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            logger.info("✅ Lệnh Create Endpoint đã được gửi thành công.")
            
        return True

    except Exception as e:
        logger.error(f"❌ Lỗi quá trình Deploy: {e}")
        raise e

# --- Hàm Handler ---
def lambda_handler(event, context):
    logger.info(f"Event Received: {json.dumps(event)}")

    try:
        # 1. Lấy thông tin Model Mới 
        new_package_arn = event['detail']['ModelPackageArn']
        new_mae = get_mae_from_model_package(new_package_arn)
        
        if new_mae == float('inf'):
            logger.error("Không lấy được MAE của model mới. Tự động Reject.")
            sagemaker.update_model_package(
                ModelPackageArn=new_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription="Error: Missing evaluation metrics."
            )
            return {"statusCode": 400, "body": "Missing metrics"}

        # 2. Lấy thông tin Model Cũ (Production/V_old)
        current_prod_mae = float('inf') 
        response_approved = sagemaker.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        if response_approved['ModelPackageSummaryList']:
            latest_approved_arn = response_approved['ModelPackageSummaryList'][0]['ModelPackageArn']
            current_prod_mae = get_mae_from_model_package(latest_approved_arn)
            logger.info(f"Model Cũ (Prod): {latest_approved_arn} | MAE: {current_prod_mae}")
        else:
            logger.info("Chưa có model Approved nào. Model mới sẽ là bản đầu tiên.")

        logger.info(f"--- SO SÁNH: Mới ({new_mae}) vs Cũ ({current_prod_mae}) ---")

        # 3. So sánh và Ra quyết định
        if new_mae < current_prod_mae:
            # === CASE: APPROVED & DEPLOY ===
            logger.info("✅ KẾT QUẢ: PHÊ DUYỆT (Model mới tốt hơn).")
            
            # A. Cập nhật trạng thái Approved
            sagemaker.update_model_package(
                ModelPackageArn=new_package_arn,
                ModelApprovalStatus='Approved',
                ApprovalDescription=f"Auto-approved: MAE {new_mae:.4f} < {current_prod_mae:.4f}"
            )
            
            # B. Gọi hàm Deploy ngay lập tức
            deploy_model_to_endpoint(new_package_arn)
            
            return {"statusCode": 200, "body": "Model Approved and Deployment Triggered."}
            
        else:
            # === CASE: REJECTED ===
            logger.info("❌ KẾT QUẢ: TỪ CHỐI (Model mới tệ hơn).")
            
            sagemaker.update_model_package(
                ModelPackageArn=new_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription=f"Auto-rejected: MAE {new_mae:.4f} >= {current_prod_mae:.4f}"
            )
            return {"statusCode": 200, "body": "Model Rejected."}

    except Exception as e:
        logger.error(f"Lỗi Critical trong Lambda: {e}")
        raise e