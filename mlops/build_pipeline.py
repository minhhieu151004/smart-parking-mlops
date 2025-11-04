import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ConditionStep
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import JsonGet
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.properties import PropertyFile

# --- Cấu hình ---
pipeline_name = "SmartParkingMLOpsPipeline"
aws_region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
default_s3_bucket = "kltn-smart-parking-data" 
role = sagemaker.get_execution_role() 

# --- Định nghĩa các tham số đầu vào cho Pipeline ---
baseline_data_uri = ParameterString(
    name="BaselineDataUrl",
    default_value=f"s3://{default_s3_bucket}/parking_data/parking_data.csv"
)
daily_data_bucket = ParameterString(
    name="DailyDataBucket",
    default_value=default_s3_bucket
)
actual_data_prefix = ParameterString(
    name="ActualDataPrefix",
    default_value="daily_actuals/" 
)
prediction_data_prefix = ParameterString(
    name="PredictionDataPrefix",
    default_value="daily_predictions/" 
)

# (SỬA) Ngưỡng P-value cho Data Drift
p_value_drift_threshold = ParameterString(name="PValueDriftThreshold", default_value="0.05")
# Ngưỡng MAE cho Model Drift
model_mae_drift_threshold = ParameterString(name="ModelMaeDriftThreshold", default_value="10.0")

# (Các tham số khác giữ nguyên)
model_version = ParameterString(
    name="ModelVersion",
    default_value=sagemaker.workflow.utilities.ExecuteVariable()
)
training_epochs = ParameterInteger(name="TrainingEpochs", default_value=50)
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")

# --- Bước 1: Kiểm tra Data Drift (Processing Step) ---
drift_output_s3_uri = f"s3://{default_s3_bucket}/pipeline-outputs/drift-check/{model_version}"

sklearn_processor_drift = SKLearnProcessor(
    framework_version="1.2", 
    role=role,
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="drift-check-job",
    sagemaker_session=sagemaker_session,
)

drift_property_file = PropertyFile(
    name="DriftCheckResult",
    output_name="drift_result",
    path="drift_check_result.json"
)

step_check_drift = ProcessingStep(
    name="CheckDataDrift",
    processor=sklearn_processor_drift,
    inputs=[
        ProcessingInput(source=baseline_data_uri, destination="/opt/ml/processing/input_baseline")
    ],
    outputs=[
        ProcessingOutput(output_name="drift_result", source="/opt/ml/processing/output",
                         destination=drift_output_s3_uri)
    ],
    code="drift_detector.py", 
    
    # (CẬP NHẬT) Sửa tham số (thay data-mae-threshold bằng p-value-threshold)
    job_arguments=[
        "--baseline-data-uri", baseline_data_uri,
        "--data-bucket", daily_data_bucket,
        "--actual-prefix", actual_data_prefix,
        "--prediction-prefix", prediction_data_prefix,
        "--output-path", "/opt/ml/processing/output",
        "--p-value-threshold", p_value_drift_threshold, # <-- Đã sửa
        "--model-mae-threshold", model_mae_drift_threshold
    ],
    property_files=[drift_property_file]
)

# --- (MỚI) Bước 3: Tổng hợp Dữ liệu (Consolidate Data) ---
# (Giữ nguyên như file build_pipeline.py trước)
consolidated_output_s3_uri = f"s3://{default_s3_bucket}/parking_data/" 

sklearn_processor_consolidate = SKLearnProcessor(
    framework_version="1.2", 
    role=role,
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="consolidate-data-job",
    sagemaker_session=sagemaker_session,
)

step_consolidate_data = ProcessingStep(
    name="ConsolidateData",
    processor=sklearn_processor_consolidate,
    inputs=[
         ProcessingInput(source=baseline_data_uri, destination="/opt/ml/processing/input_baseline")
    ],
    outputs=[
        ProcessingOutput(output_name="new_baseline_data", source="/opt/ml/processing/output",
                         destination=consolidated_output_s3_uri) 
    ],
    code="consolidate_data.py", 
    job_arguments=[
        "--baseline-data-uri", baseline_data_uri,
        "--data-bucket", daily_data_bucket,
        "--actual-prefix", actual_data_prefix,
        "--output-path", "/opt/ml/processing/output",
    ]
)

# --- (MỚI) Bước 4: Huấn luyện Mô hình (Training Step) ---
# (Giữ nguyên như file build_pipeline.py trước)
model_output_s3_uri = f"s3://{default_s3_bucket}/pipeline-outputs/training-output/{model_version}"

tf_estimator = TensorFlow(
    entry_point="train_pipeline.py", 
    source_dir=".", 
    role=role,
    instance_count=1,
    instance_type=training_instance_type,
    framework_version="2.15", 
    py_version="py310",      
    hyperparameters={
        "epochs": training_epochs,
        "learning-rate": 0.001,
        "model-version": model_version
    },
    output_path=model_output_s3_uri, 
    code_location=f"s3://{default_s3_bucket}/pipeline-code/training-code/{model_version}",
    sagemaker_session=sagemaker_session,
    metric_definitions=[
       {'Name': 'validation:loss', 'Regex': 'Validation Loss \\(MSE\\) của model tốt nhất: (\\S+)'}
    ],
    environment={"SM_OUTPUT_DATA_DIR": "/opt/ml/output/data"}
)

step_train_model = TrainingStep(
    name="TrainParkingModel",
    estimator=tf_estimator,
    inputs={
        "training": TrainingInput(
            s3_data=step_consolidate_data.properties.ProcessingOutputConfig.Outputs["new_baseline_data"].S3Output.S3Uri,
            distribution="FullyReplicated",
            content_type="text/csv",
            s3_data_type="S3Prefix"
        )
    }
)

# --- Bước 2: Điều kiện (Condition Step) ---
condition_drift_detected = ConditionEquals(
    left=JsonGet(
        step_name=step_check_drift.name,
        property_file=drift_property_file, 
        json_path="drift_detected" 
    ),
    right=True 
)

step_conditional_retrain_flow = ConditionStep(
    name="DriftCheckCondition",
    conditions=[condition_drift_detected],
    if_steps=[step_consolidate_data, step_train_model], 
    else_steps=[], 
)

# --- Tạo Pipeline ---
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        baseline_data_uri,
        daily_data_bucket,
        actual_data_prefix,
        prediction_data_prefix,
        p_value_drift_threshold,
        model_mae_drift_threshold,
        model_version,
        training_epochs,
        processing_instance_type,
        training_instance_type,
    ],
    steps=[step_check_drift, step_conditional_retrain_flow],
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    print(f"Creating/Updating Pipeline: {pipeline_name}")
    pipeline.upsert(role_arn=role)
    print("Pipeline definition submitted to SageMaker.")

