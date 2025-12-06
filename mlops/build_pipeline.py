import sagemaker
import boto3
import os 
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlowProcessor 
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.tensorflow import TensorFlow
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model import Model
from sagemaker.tensorflow.model import TensorFlowModel 
from sagemaker.workflow.model_step import ModelStep 
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.model_metrics import ModelMetrics, MetricsSource 
from sagemaker.workflow.pipeline_context import PipelineSession

# --- CẤU HÌNH ĐƯỜNG DẪN TUYỆT ĐỐI ---
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "code") 
# ------------------------------------

# --- Cấu hình ---
pipeline_name = "SmartParkingMLOpsPipeline"
aws_region = boto3.Session().region_name
sagemaker_session = PipelineSession()

default_s3_bucket = "kltn-smart-parking-data" 
role = "arn:aws:iam::120569618597:role/SageMaker-SmartParking-ExecutionRole" 
model_package_group_name = "SmartParkingModelGroup"

# --- PARAMETERS ---
baseline_data_uri = ParameterString(name="BaselineDataUrl", default_value=f"s3://{default_s3_bucket}/parking_data/parking_data.csv")
daily_data_bucket = ParameterString(name="DailyDataBucket", default_value=default_s3_bucket)
actual_data_prefix = ParameterString(name="ActualDataPrefix", default_value="daily_actuals/")
prediction_data_prefix = ParameterString(name="PredictionDataPrefix", default_value="daily_predictions/")

# Tập test gốc 
test_data_uri = ParameterString(name="TestDataUrl", default_value=f"s3://{default_s3_bucket}/parking_data/parking_test.csv")

# Tham số Drift
p_value_drift_threshold = ParameterString(name="PValueDriftThreshold", default_value="0.05")
model_mae_drift_threshold = ParameterString(name="ModelMaeDriftThreshold", default_value="10.0")

# Tham số Training
model_version = ExecutionVariables.PIPELINE_EXECUTION_ID
training_epochs = ParameterInteger(name="TrainingEpochs", default_value=2) # Fine-tune ít epochs
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t3.medium")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")

# Tham số Warm Start (Model cũ để Fine-tune)
warm_start_model_uri = ParameterString(name="WarmStartModelUrl", default_value="None") 

# ==============================================================================

# --- BƯỚC 1: CHECK DRIFT ---
drift_output_s3_uri = Join(on='/', values=[f"s3://{default_s3_bucket}/pipeline-outputs/drift-check", model_version])

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", role=role, instance_type=processing_instance_type,
    instance_count=1, base_job_name="drift-check-job", sagemaker_session=sagemaker_session,
)
drift_property_file = PropertyFile(name="DriftCheckResult", output_name="drift_result", path="drift_check_result.json")

step_check_drift = ProcessingStep(
    name="CheckDataDrift",
    processor=sklearn_processor,
    code=os.path.join(BASE_DIR, "drift_detector.py"), 
    inputs=[ProcessingInput(source=baseline_data_uri, destination="/opt/ml/processing/input_baseline")],
    outputs=[ProcessingOutput(output_name="drift_result", source="/opt/ml/processing/output", destination=drift_output_s3_uri)],
    job_arguments=[
        "--baseline-data-uri", baseline_data_uri, "--data-bucket", daily_data_bucket,
        "--actual-prefix", actual_data_prefix, "--prediction-prefix", prediction_data_prefix,
        "--output-path", "/opt/ml/processing/output", "--p-value-threshold", p_value_drift_threshold,
        "--model-mae-threshold", model_mae_drift_threshold
    ],
    property_files=[drift_property_file]
)

# --- BƯỚC 2: CONSOLIDATE DATA (Tách dữ liệu Drift để Fine-tune) ---
consolidated_output_s3_uri = f"s3://{default_s3_bucket}/parking_data/" 

step_consolidate_data = ProcessingStep(
    name="ConsolidateData",
    processor=sklearn_processor, # Tái sử dụng processor
    code=os.path.join(BASE_DIR, "consolidate_data.py"), 
    inputs=[ProcessingInput(source=baseline_data_uri, destination="/opt/ml/processing/input_baseline")],
    outputs=[
        # Output 1: File tổng (cho lưu trữ)
        ProcessingOutput(output_name="new_baseline_data", source="/opt/ml/processing/output", destination=consolidated_output_s3_uri),
        # Output 2: File drift riêng biệt (cho Fine-tuning) -> QUAN TRỌNG
        ProcessingOutput(output_name="drift_data_only", source="/opt/ml/processing/drift_data") 
    ],
    job_arguments=["--baseline-data-uri", baseline_data_uri, "--data-bucket", daily_data_bucket, "--actual-prefix", actual_data_prefix, "--output-path", "/opt/ml/processing/output"]
)

# --- BƯỚC 3: UPDATE TEST DATA (Ghép dữ liệu mới vào tập Test) ---
step_update_test_data = ProcessingStep(
    name="UpdateTestData",
    processor=sklearn_processor, # Tái sử dụng processor
    code=os.path.join(BASE_DIR, "update_test_data.py"), 
    inputs=[
        # Input 1: Dữ liệu drift (từ bước Consolidate)
        ProcessingInput(
            source=step_consolidate_data.properties.ProcessingOutputConfig.Outputs["drift_data_only"].S3Output.S3Uri,
            destination="/opt/ml/processing/drift_data"
        ),
        # Input 2: Tập Test gốc
        ProcessingInput(
            source=test_data_uri,
            destination="/opt/ml/processing/original_test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="updated_test", source="/opt/ml/processing/output")
    ]
)

# --- BƯỚC 4: TRAIN MODEL (Fine-tuning trên dữ liệu Drift) ---
model_output_s3_uri = f"s3://{default_s3_bucket}/pipeline-outputs/training-output"
code_location = f"s3://{default_s3_bucket}/pipeline-code/training-code"

tf_estimator = TensorFlow(
    entry_point="train_pipeline.py", 
    source_dir=BASE_DIR, 
    role=role, instance_count=1, instance_type=training_instance_type,
    framework_version="2.14.1", py_version="py310",      
    hyperparameters={
        "epochs": training_epochs, 
        "learning-rate": 0.001, 
        "model-version": model_version,
        "warm-start-model-uri": warm_start_model_uri # Truyền model cũ vào
    },
    output_path=model_output_s3_uri, code_location=code_location, sagemaker_session=sagemaker_session,
    metric_definitions=[{'Name': 'validation:loss', 'Regex': 'Validation Loss \\(MSE\\) của model tốt nhất: (\\S+)'}],
    environment={"SM_OUTPUT_DATA_DIR": "/opt/ml/output/data"}
)

step_train_model = TrainingStep(
    name="TrainParkingModel",
    estimator=tf_estimator,
    inputs={
        "training": TrainingInput(
            s3_data=step_consolidate_data.properties.ProcessingOutputConfig.Outputs["drift_data_only"].S3Output.S3Uri, 
            distribution="FullyReplicated", content_type="text/csv", s3_data_type="S3Prefix"
        )
    }
)

# --- BƯỚC 5: EVALUATE MODEL (Đánh giá MAE) ---
# Sử dụng TensorFlowProcessor vì evaluate_model.py cần tensorflow để load model
tf_processor_eval = TensorFlowProcessor(
    framework_version="2.14.1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="evaluate-model-job",
    sagemaker_session=sagemaker_session,
    py_version="py310"
)

evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

step_evaluate_model = ProcessingStep(
    name="EvaluateModel",
    processor=tf_processor_eval, 
    code=os.path.join(BASE_DIR, "evaluate_model.py"), 
    inputs=[
        # Input 1: Model Mới (Vừa train xong)
        ProcessingInput(
            source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/new_model",
            input_name="new_model"
        ),
        # Input 2: Model Cũ (Production)
        ProcessingInput(
            source=warm_start_model_uri, 
            destination="/opt/ml/processing/old_model",
            input_name="old_model"
        ),
        # Input 3: Tập Test (Đã cập nhật)
        ProcessingInput(
            source=step_update_test_data.properties.ProcessingOutputConfig.Outputs["updated_test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
            input_name="test_data"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/output")
    ],
    property_files=[evaluation_report]
)

# --- BƯỚC 6: REGISTER MODEL (Dùng Metrics MAE từ Evaluate) ---
model = TensorFlowModel(
    model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
    role=role, framework_version="2.14.1", sagemaker_session=sagemaker_session,
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        # Lấy metrics từ file evaluation.json của bước Evaluate
        s3_uri=step_evaluate_model.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        content_type="application/json"
    )
)

step_register_model = ModelStep(
   name="RegisterModel",
   step_args=model.register(
       content_types=["application/json"], 
       response_types=["application/json"],
       inference_instances=["ml.t2.medium"], 
       model_package_group_name=model_package_group_name, 
       approval_status="PendingManualApproval", 
       model_metrics=model_metrics 
   )
)

# --- ĐIỀU KIỆN RẼ NHÁNH ---
condition_drift_detected = ConditionEquals(left=JsonGet(step_name=step_check_drift.name, property_file=drift_property_file, json_path="drift_detected"), right=True)

# Luồng đầy đủ: Check -> Consolidate -> UpdateTest -> Train -> Evaluate -> Register
step_conditional_retrain_flow = ConditionStep(
    name="DriftCheckCondition", 
    conditions=[condition_drift_detected], 
    if_steps=[step_consolidate_data, step_update_test_data, step_train_model, step_evaluate_model, step_register_model], 
    else_steps=[]
)

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        baseline_data_uri, daily_data_bucket, actual_data_prefix, prediction_data_prefix, test_data_uri,
        model_mae_drift_threshold, p_value_drift_threshold, training_epochs, 
        processing_instance_type, training_instance_type, 
        warm_start_model_uri 
    ],
    steps=[step_check_drift, step_conditional_retrain_flow],
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    print(f"Creating/Updating Pipeline: {pipeline_name} với Role: {role}")
    pipeline.upsert(role_arn=role)
    print("Pipeline definition submitted to SageMaker.")