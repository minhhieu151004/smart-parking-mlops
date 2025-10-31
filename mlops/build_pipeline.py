import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ConditionStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor 
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import JsonGet
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.tensorflow import TensorFlow 
from sagemaker.workflow.utilities import ExecuteVariable

# --- Cấu hình ---
S3_BUCKET = "kltn-smart-parking-data" 
PIPELINE_NAME = "SmartParking-MLOps-Pipeline-v1"

# Lấy thông tin AWS
sagemaker_session = sagemaker.Session()
aws_region = sagemaker_session.boto_region_name
# Lấy IAM Role. Bạn PHẢI có một IAM Role cho SageMaker
# Cách 1: Tự động lấy Role (nếu bạn chạy script này trên EC2/SageMaker Notebook đã gắn Role)
try:
    role = sagemaker.get_execution_role()
except ValueError:
    # Cách 2: Chỉ định ARN của Role (nếu chạy từ máy local)
    # Thay 'YOUR_SAGEMAKER_ROLE_ARN' bằng ARN của Role
    # role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE"
    print("Không thể tự động lấy Role. Hãy đảm bảo bạn đã cấu hình Role.")
    # Tạm thời dùng 1 placeholder, bạn cần sửa lại nếu chạy local
    role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-Default"


# --- Định nghĩa các tham số Pipeline ---
# Tham số: Đường dẫn S3 đến file dữ liệu đầy đủ
input_data_uri = ParameterString(
    name="InputDataUrl",
    default_value=f"s3://{S3_BUCKET}/parking_data/parking_data.csv"
)
# Tham số: Phiên bản model 
model_version = ParameterString(
    name="ModelVersion",
    default_value=ExecuteVariable() 
)
# Tham số: Ngưỡng P-value để phát hiện drift
p_value_drift_threshold = ParameterString(name="PValueDriftThreshold", default_value="0.05")
# Tham số: Số Epochs huấn luyện
training_epochs = ParameterInteger(name="TrainingEpochs", default_value=50)
# Tham số: Loại máy chủ (Instance Type)
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")


# --- Định nghĩa Bước 1: Kiểm tra Data Drift (Processing Step) ---

# Nơi lưu output của bước drift check (file JSON)
drift_output_s3_uri = f"s3://{S3_BUCKET}/pipeline-outputs/drift-check/{model_version}/"

# Định nghĩa Processor (môi trường) để chạy script
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", # Phiên bản Scikit-learn
    role=role,
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="drift-check-job",
    sagemaker_session=sagemaker_session,
)

# Định nghĩa output của bước processing
drift_check_output = ProcessingOutput(
    output_name="drift_result", # Tên của output
    source="/opt/ml/processing/output", # Đường dẫn local trong container
    destination=drift_output_s3_uri # Đường dẫn S3
)

# Định nghĩa bước Processing
step_check_drift = ProcessingStep(
    name="CheckDataDrift",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=input_data_uri, # Lấy dữ liệu từ S3
            destination="/opt/ml/processing/input_data" # Tải về đường dẫn này
        )
    ],
    outputs=[drift_check_output],
    # Script này phải nằm CÙNG THƯ MỤC với build_pipeline.py
    code="drift_detector.py", 
    # Truyền tham số cho script drift_detector.py
    job_arguments=[
        "--input-data", f"/opt/ml/processing/input_data/{input_data_uri.default_value.split('/')[-1]}", # Đường dẫn file local
        "--output-path", "/opt/ml/processing/output",
        "--p-value-threshold", p_value_drift_threshold
    ]
)

# --- Định nghĩa Bước 2: Huấn luyện Mô hình (Training Step) ---

# Nơi lưu model.tar.gz
model_output_s3_uri = f"s3://{S3_BUCKET}/pipeline-outputs/training-output/{model_version}/output/"
# Nơi lưu metrics.json (từ 'SM_OUTPUT_DATA_DIR')
metrics_output_s3_uri = f"s3://{S3_BUCKET}/pipeline-outputs/metrics-output/{model_version}/"
# Nơi SageMaker upload code lên (bắt buộc)
code_location_s3_uri = f"s3://{S3_BUCKET}/pipeline-code/training-code/{model_version}/"

# Định nghĩa Estimator (môi trường) TensorFlow
tf_estimator = TensorFlow(
    entry_point="train_pipeline.py", # Script huấn luyện (phải ở cùng thư mục)
    source_dir=".", # Thư mục chứa script
    role=role,
    instance_count=1,
    instance_type=training_instance_type,
    framework_version="2.15", # Phiên bản TensorFlow
    py_version="py310",      # Phiên bản Python
    hyperparameters={ # Truyền hyperparameters vào script train_pipeline.py
        "epochs": training_epochs,
        "learning-rate": 0.001,
        "model-version": model_version # Truyền version ID
    },
    output_path=model_output_s3_uri, # Nơi lưu model.tar.gz
    code_location=code_location_s3_uri,
    sagemaker_session=sagemaker_session,
    # Định nghĩa thư mục output cho metrics (SM_OUTPUT_DATA_DIR)
    environment={"SM_OUTPUT_DATA_DIR": "/opt/ml/output/data"} 
)

# Định nghĩa bước Training
step_train_model = TrainingStep(
    name="TrainParkingModel",
    estimator=tf_estimator,
    inputs={
        "training": TrainingInput( # Kênh input tên là "training"
            s3_data=input_data_uri,
            distribution="FullyReplicated",
            content_type="text/csv"
        )
    }
    # SageMaker sẽ tự động lưu model (từ /opt/ml/model) 
    # và metrics (từ /opt/ml/output/data)
)

# --- Định nghĩa Bước 3: Điều kiện (Condition Step) ---

# Đọc giá trị 'drift_detected' từ file JSON output của bước 1
condition_drift_detected = ConditionEquals(
    left=JsonGet(
        step_name=step_check_drift.name,
        # Lấy đường dẫn file JSON kết quả
        property_file=drift_check_output.output_path + "/drift_check_result.json",
        json_path="drift_detected" # Lấy giá trị của key "drift_detected"
    ),
    right=True # So sánh xem nó có bằng True (Boolean) không
)

# Định nghĩa bước điều kiện
step_conditional_training = ConditionStep(
    name="DriftCheckCondition",
    conditions=[condition_drift_detected], # Điều kiện để kiểm tra
    if_steps=[step_train_model], # Nếu True (có drift) -> Chạy bước huấn luyện
    else_steps=[], # Nếu False (không drift) -> Không làm gì cả
)

# --- Định nghĩa Bước 4: (Tùy chọn) Đăng ký Mô hình ---
# Đây là bước nâng cao, bạn có thể thêm sau:
# Sau khi train, đăng ký model vào SageMaker Model Registry
# (Hiện tại Lambda của bạn đang làm việc này - so sánh và promote)

# --- Tạo đối tượng Pipeline ---
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[
        input_data_uri,
        model_version,
        p_value_drift_threshold,
        training_epochs,
        processing_instance_type,
        training_instance_type,
    ],
    steps=[step_check_drift, step_conditional_training], # Chạy drift check, sau đó là điều kiện
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    print(f"Đang tạo/cập nhật Pipeline: {PIPELINE_NAME}")
    try:
        # Tạo hoặc cập nhật (Upsert) định nghĩa pipeline trên SageMaker
        pipeline.upsert(role_arn=role)
        print(f"Đã gửi định nghĩa Pipeline '{PIPELINE_NAME}' lên SageMaker thành công.")
        print("Bạn có thể vào SageMaker Studio -> Pipelines để xem.")
        print("Bước tiếp theo: Tạo EventBridge Schedule để trigger pipeline này.")
    except Exception as e:
        print(f"Lỗi khi upsert pipeline: {e}")

