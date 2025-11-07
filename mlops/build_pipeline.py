import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
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

# --- Cấu hình ---
pipeline_name = "SmartParkingMLOpsPipeline"
aws_region = boto3.Session().region_name
sagemaker_session = PipelineSession()

default_s3_bucket = "kltn-smart-parking-data" 
role = "arn:aws:iam::120569618597:role/SageMaker-SmartParking-ExecutionRole" 
model_package_group_name = "SmartParkingModelGroup"

# --- (Các tham số parameters giữ nguyên) ---
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
p_value_drift_threshold = ParameterString(name="PValueDriftThreshold", default_value="0.05")
model_mae_drift_threshold = ParameterString(name="ModelMaeDriftThreshold", default_value="10.0")

model_version = ExecutionVariables.PIPELINE_EXECUTION_ID

training_epochs = ParameterInteger(name="TrainingEpochs", default_value=50)
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t3.medium")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.t3.medium")

# --- (Bước 1: step_check_drift - Giữ nguyên) ---
drift_output_s3_uri = Join(
    on='/', 
    values=[
        f"s3://{default_s3_bucket}/pipeline-outputs/drift-check",
        model_version
    ]
)

sklearn_processor_drift = SKLearnProcessor(
    framework_version="1.2-1", 
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
    code="drift_detector.py", 
    
    inputs=[ProcessingInput(source=baseline_data_uri, destination="/opt/ml/processing/input_baseline")],
    outputs=[
        ProcessingOutput(output_name="drift_result", source="/opt/ml/processing/output",
                         destination=drift_output_s3_uri)
    ],
    job_arguments=[
        "--baseline-data-uri", baseline_data_uri,
        "--data-bucket", daily_data_bucket,
        "--actual-prefix", actual_data_prefix,
        "--prediction-prefix", prediction_data_prefix,
        "--output-path", "/opt/ml/processing/output",
        "--p-value-threshold", p_value_drift_threshold,
        "--model-mae-threshold", model_mae_drift_threshold
    ],
    property_files=[drift_property_file]
)

# --- (Bước 3: step_consolidate_data - Giữ nguyên) ---
consolidated_output_s3_uri = f"s3://{default_s3_bucket}/parking_data/" 
sklearn_processor_consolidate = SKLearnProcessor(
    framework_version="1.2-1", 
    role=role,
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="consolidate-data-job",
    sagemaker_session=sagemaker_session,
)
step_consolidate_data = ProcessingStep(
    name="ConsolidateData",
    processor=sklearn_processor_consolidate,
    code="consolidate_data.py", 
    
    inputs=[
         ProcessingInput(source=baseline_data_uri, destination="/opt/ml/processing/input_baseline")
    ],
    outputs=[
        ProcessingOutput(output_name="new_baseline_data", source="/opt/ml/processing/output",
                         destination=consolidated_output_s3_uri)
    ],
    job_arguments=[
        "--baseline-data-uri", baseline_data_uri,
        "--data-bucket", daily_data_bucket,
        "--actual-prefix", actual_data_prefix,
        "--output-path", "/opt/ml/processing/output",
    ]
)

# --- (Bước 4: step_train_model - Giữ nguyên) ---
model_output_s3_uri = f"s3://{default_s3_bucket}/pipeline-outputs/training-output"
code_location = f"s3://{default_s3_bucket}/pipeline-code/training-code"

tf_estimator = TensorFlow(
    entry_point="train_pipeline.py", 
    source_dir=".", 
    role=role,
    instance_count=1,
    instance_type=training_instance_type,
    framework_version="2.14.1", 
    py_version="py310",      
    hyperparameters={
        "epochs": training_epochs,
        "learning-rate": 0.001,
        "model-version": model_version 
    },
    output_path=model_output_s3_uri,
    code_location=code_location,
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

# --- (Bước 5: step_register_model) ---
model = TensorFlowModel(
    model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    framework_version="2.14.1", 
    sagemaker_session=sagemaker_session,
    entry_point="code/inference.py", 
    source_dir="."                     
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on='/',
            values=[
                model_output_s3_uri,  
                step_train_model.properties.TrainingJobName, 
                "output/data",        
                "metrics.json"        
            ]
        ),
        content_type="application/json"
    )
)

step_register_model = ModelStep(
   name="RegisterModel",
   step_args=model.register(
       content_types=["text/csv"],
       response_types=["application/json"],
       inference_instances=["ml.t2.medium"], 
       #transform_instances=["ml.t2.medium"],
       model_package_group_name=model_package_group_name, 
       
       # =========== (SỬA LỖI API TẠI ĐÂY) ===========
       approval_status="PendingManualApproval", # <== SỬA THÀNH NÀY
       # ============================================
       
       model_metrics=model_metrics 
   )
)

# --- (Bước 2: Condition - Giữ nguyên) ---
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
    if_steps=[step_consolidate_data, step_train_model, step_register_model], 
    else_steps=[], 
)

# --- Tạo Pipeline (Giữ nguyên) ---
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        baseline_data_uri,
        daily_data_bucket,
        actual_data_prefix,
        prediction_data_prefix,
        model_mae_drift_threshold,
        p_value_drift_threshold, 
        training_epochs,
        processing_instance_type,
        training_instance_type,
    ],
    steps=[step_check_drift, step_conditional_retrain_flow],
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    
    print(f"Creating/Updating Pipeline: {pipeline_name} với Role: {role}")
    pipeline.upsert(role_arn=role)
    print("Pipeline definition submitted to SageMaker.")