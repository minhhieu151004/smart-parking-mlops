import json
import os
import boto3
import logging
import tarfile
from io import BytesIO

# --- Cấu hình Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Khởi tạo Clients ---
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# --- Lấy Biến Môi trường ---
try:
    S3_BUCKET = os.environ['S3_BUCKET']
    MODEL_PACKAGE_GROUP_NAME = os.environ['MODEL_PACKAGE_GROUP_NAME']
    ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
except KeyError as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Biến môi trường {e} chưa được set!")
    raise

# --- Hàm Helper ---
def get_metrics_from_model_package(model_package_arn):
    """
    Hàm này lấy ARN, tìm file metrics.json trên S3 (ngay cả khi nó
    bị nén trong output.tar.gz), tải về và parse nó.
    """
    try:
        # 1. Lấy mô tả chi tiết của model package
        package_desc = sagemaker.describe_model_package(
            ModelPackageName=model_package_arn
        )
        
        # 2. Tìm đường dẫn S3 của file metrics
        metrics_s3_uri = package_desc['ModelMetrics']['ModelQuality']['Statistics']['S3Uri']
        
        # 3. Tách bucket và key
        metrics_s3_path_parts = metrics_s3_uri.replace('s3://', '').split('/', 1)
        metrics_bucket = metrics_s3_path_parts[0]
        metrics_key = metrics_s3_path_parts[1]
        
        # 4. Tải và đọc file metrics.json
        logger.info(f"Đang thử tải metrics trực tiếp từ: s3://{metrics_bucket}/{metrics_key}")
        try:
            # === KỊCH BẢN A: File metrics là một file .json độc lập ===
            metrics_obj = s3.get_object(Bucket=metrics_bucket, Key=metrics_key)
            metrics_data = json.loads(metrics_obj['Body'].read().decode('utf-8'))
            logger.info("Tải file .json trực tiếp thành công.")

        except s3.exceptions.NoSuchKey:
            # === KỊCH BẢN B: Bị 'NoSuchKey'. File metrics nằm trong output.tar.gz ===
            logger.warning(f"Không tìm thấy file {metrics_key}. " \
                           "Giả định metrics nằm trong 'output.tar.gz' ở cùng thư mục.")
            
            # Xây dựng đường dẫn đến file output.tar.gz
            metrics_dir = os.path.dirname(metrics_key)
            tarball_key = os.path.join(metrics_dir, 'output.tar.gz')
            
            logger.info(f"Đang thử tải tarball từ: s3://{metrics_bucket}/{tarball_key}")
            
            try:
                # Tải file .tar.gz
                tar_obj = s3.get_object(Bucket=metrics_bucket, Key=tarball_key)
                tar_data = BytesIO(tar_obj['Body'].read()) # Đọc vào bộ nhớ
                
                # Giải nén file trong bộ nhớ
                with tarfile.open(fileobj=tar_data, mode='r:gz') as tar:
                    # Tìm file 'metrics.json' bên trong tarball
                    try:
                        # Thử tìm 'metrics.json' trước
                        metrics_file_info = tar.getmember('metrics.json')
                    except KeyError:
                        try:
                            metrics_file_info = tar.getmember('metrics')
                        except KeyError:
                             logger.error(f"Không tìm thấy 'metrics' hoặc 'metrics.json' bên trong {tarball_key}")
                             raise ValueError(f"Không tìm thấy file metrics bên trong tarball.")
                    
                    # Trích xuất và đọc file
                    logger.info(f"Đọc file '{metrics_file_info.name}' từ bên trong tarball.")
                    metrics_file = tar.extractfile(metrics_file_info)
                    metrics_data = json.loads(metrics_file.read().decode('utf-8'))

            except s3.exceptions.NoSuchKey:
                logger.error(f"Lỗi nghiêm trọng: Đã thử {metrics_key} (NoSuchKey) " \
                             f"VÀ {tarball_key} (NoSuchKey). Không thể tìm thấy metrics ở đâu.")
                return float('inf')
            except tarfile.TarError as tar_err:
                logger.error(f"Lỗi khi giải nén {tarball_key}: {tar_err}")
                return float('inf')

        # 5. Lấy val_loss từ file metrics đã được parse
        val_loss = metrics_data.get('val_loss')
        
        if val_loss is None:
            raise ValueError(f"Không tìm thấy 'val_loss' trong file metrics: {metrics_s3_uri}")
            
        return val_loss

    except KeyError as e:
        logger.error(f"Lỗi KeyError khi truy cập {e} trong ModelMetrics. " \
                     "Rất có thể bạn đã đăng ký metrics với một tên khác " \
                     "(không phải 'ModelQuality'). Vui lòng kiểm tra lại 'RegisterModel' step.")
        logger.error(f"Cấu trúc ModelMetrics: {package_desc.get('ModelMetrics')}")
        return float('inf')
    except Exception as e:
        logger.error(f"Lỗi khi lấy metrics từ {model_package_arn}: {e}")
        return float('inf') # Trả về lỗi vô cực nếu không đọc được

# --- Hàm Handler Chính (Đã cập nhật) ---
def lambda_handler(event, context):
    """
    Được trigger khi SageMaker Pipeline 'Succeeded'.
    Nhiệm vụ: Tìm, So sánh, Phê duyệt, và Triển khai.
    """
    logger.info(f"Event nhận được: {json.dumps(event)}")

    try:
        pipeline_execution_arn = event['detail']['pipelineExecutionArn']
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
            logger.info("Không phát hiện drift. Không cần làm gì thêm.")
            return {"statusCode": 200, "body": "Không có drift, không huấn luyện."}

        logger.info("Phát hiện Drift. Đang tiến hành đánh giá mô hình mới...")

        # === 3. Tìm Model Package 'Pending' mới nhất ===
        response_pending = sagemaker.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus='PendingManualApproval',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        if not response_pending['ModelPackageSummaryList']:
            raise ValueError("Không tìm thấy model package 'PendingManualApproval' để phê duyệt.")
            
        latest_pending_package = response_pending['ModelPackageSummaryList'][0]
        pending_package_arn = latest_pending_package['ModelPackageArn']
        logger.info(f"Tìm thấy model 'Pending' mới nhất: {pending_package_arn}")

        # === 4. Lấy Metrics của Model MỚI ===
        new_loss = get_metrics_from_model_package(pending_package_arn)
        logger.info(f"Model MỚI (Pending) có val_loss: {new_loss}")

        # === 5. Lấy Metrics của Model 'Production' (CŨ) ===
        current_prod_loss = float('inf') # Mặc định 
        try:
            response_approved = sagemaker.list_model_packages(
                ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
                ModelApprovalStatus='Approved', # Tìm model đang chạy
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=1
            )
            if response_approved['ModelPackageSummaryList']:
                latest_approved_package_arn = response_approved['ModelPackageSummaryList'][0]['ModelPackageArn']
                logger.info(f"Tìm thấy model 'Approved' (Production) hiện tại: {latest_approved_package_arn}")
                current_prod_loss = get_metrics_from_model_package(latest_approved_package_arn)
            else:
                logger.warning("Không tìm thấy model 'Approved' nào. Model mới sẽ được tự động thúc đẩy.")
        except Exception as e:
            logger.warning(f"Lỗi khi tìm model 'Approved': {e}. Model mới sẽ được tự động thúc đẩy.")
            
        logger.info(f"Model CŨ (Production) có val_loss: {current_prod_loss}")

        # === 6. So sánh và Quyết định ===
        if new_loss < current_prod_loss:
            # === 6a. Model Mới TỐT HƠN -> Phê duyệt và Deploy ===
            logger.warning(f"QUYẾT ĐỊNH: THÚC ĐẨY. Model mới (loss={new_loss}) tốt hơn 'production' (loss={current_prod_loss}).")
            
            # Phê duyệt
            logger.info(f"Đang cập nhật trạng thái 'Approved' cho: {pending_package_arn}")
            sagemaker.update_model_package(
                ModelPackageArn=pending_package_arn,
                ModelApprovalStatus='Approved',
                ApprovalDescription=f"Tự động phê duyệt: val_loss {new_loss} tốt hơn {current_prod_loss}."
            )
            
            # Deploy
            logger.info(f"Bắt đầu quá trình deploy/update cho Endpoint: {ENDPOINT_NAME}")
            deploy_sagemaker_endpoint(pending_package_arn, execution_id, context)

            return {"statusCode": 200, "body": f"Model đã được 'Approved' và deploy/update."}

        else:
            # === 6b. Model Mới TỆ HƠN -> Từ chối ===
            logger.warning(f"QUYẾT ĐỊNH: TỪ CHỐI. Model mới (loss={new_loss}) không tốt hơn 'production' (loss={current_prod_loss}).")
            
            # Từ chối
            sagemaker.update_model_package(
                ModelPackageArn=pending_package_arn,
                ModelApprovalStatus='Rejected',
                ApprovalDescription=f"Tự động từ chối: val_loss {new_loss} không tốt hơn {current_prod_loss}."
            )
            return {"statusCode": 200, "body": "Model mới không tốt hơn, đã từ chối."}

    except Exception as e:
        logger.error(f"Lỗi trong quá trình thực thi Lambda: {e}")
        raise e

def deploy_sagemaker_endpoint(model_package_arn, execution_id, context):
    """
    Hàm này nhận ARN của model package đã được 'Approved'
    và tạo/cập nhật SageMaker Endpoint.
    """
    try:
        model_package_desc = sagemaker.describe_model_package(ModelPackageName=model_package_arn)
        model_data_url = model_package_desc['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        image_uri = model_package_desc['InferenceSpecification']['Containers'][0]['Image']
        
        model_name = f"model-{execution_id}" 
        logger.info(f"Đang tạo Model: {model_name}")
        
        # Lấy IAM Role của Lambda (Role này cần quyền sagemaker:CreateModel)
        lambda_role_arn = context.invoked_function_arn
        account_id = lambda_role_arn.split(':')[4]
        role_name = os.environ.get('AWS_LAMBDA_EXECUTION_ROLE_NAME', 'Lambda-SmartParking-ExecutionRol')
        execution_role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
        
        logger.info(f"Sử dụng Execution Role: {execution_role_arn}")
        
        try:
            sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_data_url,
                },
                ExecutionRoleArn=execution_role_arn
            )
        except sagemaker.exceptions.ClientError as e:
            if "Cannot create already existing model" in str(e):
                logger.warning(f"Model {model_name} đã tồn tại. Bỏ qua tạo model.")
            else:
                raise e

        endpoint_config_name = f"epc-{execution_id}" 
        logger.info(f"Đang tạo Endpoint Config: {endpoint_config_name}")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'ServerlessConfig': {
                        'MemorySizeInMB': 2048, 
                        'MaxConcurrency': 10
                    }
                }
            ]
        )

        try:
            sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
            logger.info(f"Endpoint {ENDPOINT_NAME} đã tồn tại. Đang cập nhật...")
            sagemaker.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            logger.info("Update Endpoint thành công.")
            
        except sagemaker.exceptions.ClientError:
            logger.info(f"Endpoint {ENDPOINT_NAME} chưa tồn tại. Đang tạo mới...")
            sagemaker.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name
            )
            logger.info("Create Endpoint thành công.")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình deploy/update Endpoint: {e}")
        raise e