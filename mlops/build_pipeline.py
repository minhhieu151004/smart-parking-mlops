import sagemaker
import boto3
import os 
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tensorflow import TensorFlowProcessor, TensorFlow
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.properties import PropertyFile
from sagemaker.tensorflow.model import TensorFlowModel 
from sagemaker.workflow.model_step import ModelStep 
from sagemaker.workflow.functions import JsonGet
from sagemaker.model_metrics import ModelMetrics, MetricsSource 
from sagemaker.workflow.pipeline_context import PipelineSession

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
pipeline_name = "SmartParking-Weekly-Retrain-Pipeline"
sagemaker_session = PipelineSession()

default_s3_bucket = "kltn-smart-parking-data" 
role = "arn:aws:iam::120569618597:role/SageMaker-SmartParking-ExecutionRole" 
model_package_group_name = "SmartParkingModelGroup"

# --- PARAMETERS ---
# 1. Master Data for Retraining
input_data_uri = ParameterString(
    name="InputDataUrl", 
    default_value=f"s3://{default_s3_bucket}/parking_data/parking_data.csv"
)

# 2. Fixed Test Data 
test_data_uri = ParameterString(
    name="TestDataUrl", 
    default_value=f"s3://{default_s3_bucket}/parking_data/parking_test.csv"
)

# 3. Drift Parameters
mae_threshold = ParameterString(name="MaeThreshold", default_value="10.0") 

# 4. Training Parameters
training_epochs = ParameterInteger(name="TrainingEpochs", default_value=50)
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t3.medium")
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")

# ==============================================================================

# --- STEP 1: CHECK WEEKLY DRIFT ---
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", role=role, 
    instance_type=processing_instance_type, instance_count=1, 
    base_job_name="drift-check", sagemaker_session=sagemaker_session,
)

drift_property_file = PropertyFile(name="DriftResult", output_name="drift_output", path="drift_check_result.json")

step_check_drift = ProcessingStep(
    name="CheckWeeklyDrift",
    processor=sklearn_processor,
    code=os.path.join(BASE_DIR, "drift_detector.py"),
    outputs=[
        ProcessingOutput(output_name="drift_output", source="/opt/ml/processing/output")
    ],
    job_arguments=[
        "--bucket", default_s3_bucket,
        "--mae-threshold", mae_threshold, 
        "--actual-prefix", "daily_actuals/",
        "--prediction-prefix", "daily_predictions/"
    ],
    property_files=[drift_property_file]
)

# ==============================================================================
# RETRAIN BRANCH (Only runs if Drift Detected = True)
# ==============================================================================

# --- STEP 2: PREPROCESSING (Take last 70% data) ---
step_preprocess = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    code=os.path.join(BASE_DIR, "preprocessing.py"), 
    inputs=[
        ProcessingInput(
            source=input_data_uri, 
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
    ]
)

# --- STEP 3: TRAIN MODEL (From Scratch) ---
tf_estimator = TensorFlow(
    entry_point="train_pipeline.py", 
    source_dir=BASE_DIR, 
    role=role, 
    instance_count=1, 
    instance_type=training_instance_type,
    framework_version="2.14.1", 
    py_version="py310",      
    hyperparameters={
        "epochs": training_epochs, 
        "learning-rate": 0.001
    },
    output_path=f"s3://{default_s3_bucket}/pipeline-outputs/training-output", 
    sagemaker_session=sagemaker_session,
)

step_train = TrainingStep(
    name="TrainParkingModel",
    estimator=tf_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri, 
            content_type="application/x-npy"
        )
    }
)

# --- STEP 4: EVALUATE MODEL ---
tf_processor_eval = TensorFlowProcessor(
    framework_version="2.14.1", role=role,
    instance_type=processing_instance_type, instance_count=1,
    base_job_name="evaluate-model", sagemaker_session=sagemaker_session, py_version="py310"
)

evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

step_evaluate = ProcessingStep(
    name="EvaluateModel",
    processor=tf_processor_eval, 
    code=os.path.join(BASE_DIR, "evaluate_model.py"), 
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts, 
            destination="/opt/ml/processing/new_model",
            input_name="new_model_tar"
        ),
        ProcessingInput(
            source=test_data_uri, 
            destination="/opt/ml/processing/test",
            input_name="test_data"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/output")
    ],
    property_files=[evaluation_report]
)

# --- STEP 5: REGISTER MODEL ---
model = TensorFlowModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role, 
    framework_version="2.14.1", 
    sagemaker_session=sagemaker_session,
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri, 
        content_type="application/json"
    )
)

step_register = ModelStep(
   name="RegisterModel",
   step_args=model.register(
       content_types=["application/json"], 
       response_types=["application/json"],
       inference_instances=["ml.t2.medium", "ml.m5.large"], 
       model_package_group_name=model_package_group_name, 
       approval_status="PendingManualApproval", 
       model_metrics=model_metrics 
   )
)

# ==============================================================================
# LOGIC CONDITION
# ==============================================================================

# Condition: drift_detected == True (Output from drift_detector.py)
cond_drift = ConditionEquals(
    left=JsonGet(
        step_name=step_check_drift.name, 
        property_file=drift_property_file, 
        json_path="drift_detected"
    ),
    right=True
)

step_cond = ConditionStep(
    name="CheckDriftCondition",
    conditions=[cond_drift],
    if_steps=[step_preprocess, step_train, step_evaluate, step_register], # If Drift -> Run Retrain Flow
    else_steps=[] # If No Drift -> Stop
)

# --- CREATE PIPELINE ---
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        input_data_uri, test_data_uri, 
        training_epochs, mae_threshold, 
        processing_instance_type, training_instance_type
    ],
    steps=[step_check_drift, step_cond], 
    sagemaker_session=sagemaker_session,
)

if __name__ == "__main__":
    print(f"Creating/Updating Pipeline: {pipeline_name}")
    pipeline.upsert(role_arn=role)
    print("✅ Pipeline updated: Drift Check (7 days) -> Retrain Flow.")