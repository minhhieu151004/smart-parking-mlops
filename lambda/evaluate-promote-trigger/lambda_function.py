import json
import os
import boto3
import logging

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# --- Lấy Biến Môi trường ---
try:
    # Bạn cần set biến này trong Configuration của Lambda
    MODEL_PACKAGE_GROUP_NAME = os.environ['MODEL_PACKAGE_GROUP_NAME'] 
except KeyError:
    logger.warning("Chưa set MODEL_PACKAGE_GROUP_NAME, dùng mặc định: SmartParkingModelGroup")
    MODEL_PACKAGE_GROUP_NAME = "SmartParkingModelGroup"

# --- Hàm Helper: Lấy MAE từ Model Package ---
def get_mae_from_model_package(model_package_arn):
    """
    Lấy ARN -> Tìm S3 URI của evaluation.json -> Tải về -> Trả về MAE
    """
    try:
        # 1. Lấy metadata của model package
        logger.info(f"Đang kiểm tra Model: {model_package_arn}")
        package_desc = sagemaker.describe_model_package(
            ModelPackageName=model_package_arn
        )
        
        # 2. Tìm đường dẫn S3 của file metrics (evaluation.json)
        try:
            metrics_s3_uri = package_desc['ModelMetrics']['ModelStatistics']['S3Uri']
        except KeyError:
            logger.warning(f"Model {model_package_arn} không có ModelMetrics. Bỏ qua.")
            return float('inf') # Trả về vô cực nếu không có metrics
        
        # 3. Parse S3 URI (s3://bucket/key)
        metrics_s3_path_parts = metrics_s3_uri.replace('s3://', '').split('/', 1)
        metrics_bucket = metrics_s3_path_parts[0]
        metrics_key = metrics_s3_path_parts[1]
        
        # 4. Tải file evaluation.json
        logger.info(f"Đang tải metrics từ: s3://{metrics_bucket}/{metrics_key}")
        metrics_obj = s3.get_object(Bucket=metrics_bucket, Key=metrics_key)
        metrics_data = json.loads(metrics_obj['Body'].read().decode('utf-8'))
        
        # 5. Trích xuất MAE
        # Cấu trúc JSON: {"regression_metrics": {"mae": {"value": 5.2}}}
        try:
            mae = metrics_data['regression_metrics']['mae']['value']
            logger.info(f"-> Tìm thấy MAE: {mae}")
            return float(mae)
        except KeyError:
            # Fallback nếu cấu trúc file json khác
            logger.warning("Không tìm thấy key 'regression_metrics.mae.value' trong JSON.")
            return float('inf')

    except Exception as e:
        logger.error(f"Lỗi khi lấy metrics từ {model_package_arn}: {e}")
        return float('inf')

# --- Hàm Handler Chính ---
def lambda_handler(event, context):
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        # 1. Lấy ARN của Model Mới (V_new) từ Event
        # EventBridge gửi ARN trong detail -> ModelPackageArn
        new_package_arn = event['detail']['ModelPackageArn']
        logger.info(f"Đang đánh giá Model Mới: {new_package_arn}")

        # 2. Lấy MAE của Model Mới
        new_mae = get_mae_from_model_package(new_package_arn)
        
        # Nếu không lấy được MAE (lỗi hoặc file rỗng), từ chối ngay
        if new_mae == float('inf'):
            logger.error("Không lấy được MAE của model mới. Tự động từ chối.")
            sagemaker.update_model_package(
                ModelPackageArn=new_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription="Missing evaluation metrics."
            )
            return {"statusCode": 400, "body": "Missing metrics"}

        # 3. Tìm Model Cũ (Production/Approved) để so sánh
        current_prod_mae = float('inf') 
        
        # Lấy danh sách các model đã Approved, sắp xếp mới nhất
        response_approved = sagemaker.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        if response_approved['ModelPackageSummaryList']:
            latest_approved_package_arn = response_approved['ModelPackageSummaryList'][0]['ModelPackageArn']
            logger.info(f"Model Production hiện tại: {latest_approved_package_arn}")
            
            # Lấy MAE của model cũ
            # Lưu ý: Chúng ta lấy MAE từ file evaluation.json CỦA MODEL CŨ
            current_prod_mae = get_mae_from_model_package(latest_approved_package_arn)
        else:
            logger.info("Chưa có model Approved nào. Model mới sẽ là model đầu tiên.")

        logger.info(f"SO SÁNH: Mới ({new_mae}) vs Cũ ({current_prod_mae})")

        # 4. Logic Quyết định (MAE càng thấp càng tốt)
        if new_mae < current_prod_mae:
            # --- APPROVED ---
            logger.info("✅ QUYẾT ĐỊNH: PHÊ DUYỆT (Model mới tốt hơn).")
            
            sagemaker.update_model_package(
                ModelPackageArn=new_package_arn,
                ModelApprovalStatus='Approved',
                ApprovalDescription=f"Auto-approved: MAE {new_mae:.4f} < {current_prod_mae:.4f}"
            )
            # Lưu ý: Lambda này KHÔNG deploy. Việc deploy do hệ thống khác lo hoặc manual.
            return {"statusCode": 200, "body": "Model Approved"}
            
        else:
            # --- REJECTED ---
            logger.info("❌ QUYẾT ĐỊNH: TỪ CHỐI (Model mới tệ hơn hoặc bằng).")
            
            sagemaker.update_model_package(
                ModelPackageArn=new_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription=f"Auto-rejected: MAE {new_mae:.4f} >= {current_prod_mae:.4f}"
            )
            return {"statusCode": 200, "body": "Model Rejected"}

    except Exception as e:
        logger.error(f"Lỗi Critical: {e}")
        raise e