from __future__ import annotations
import pendulum
import os
import json
import boto3
import logging

from airflow.models.dag import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator # THÊM MỚI
from airflow.providers.jenkins.operators.jenkins_job_trigger import JenkinsJobTriggerOperator
# --- THÊM MỚI (Mục 1) ---
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
# ----------------------

# --- Các biến cấu hình ---
# THAY ĐỔI CÁC GIÁ TRỊ NÀY CHO PHÙ HỢP
JENKINS_CONNECTION_ID = "jenkins_default"
JENKINS_RETRAIN_JOB = "parking-model-retrain"
JENKINS_RESTART_JOB = "parking-service-restart"
DRIFT_SCRIPT_PATH = "/opt/airflow/scripts/mlops/drift_detector.py" # Giữ nguyên: Đường dẫn này trên MÁY CHỦ AIRFLOW

# --- THÊM MỚI (Mục 1 & 2): Cấu hình MinIO/S3 và Môi trường ---
# Đảm bảo các biến này được set trong Airflow UI (Admin -> Variables)
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'admin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'password')
S3_BUCKET = os.getenv('S3_BUCKET', 'my-bucket')

# (Mục 1) Biến môi trường cho DockerOperator
DOCKER_ENV_VARS = {
    'MINIO_ENDPOINT': MINIO_ENDPOINT,
    'MINIO_ACCESS_KEY': MINIO_ACCESS_KEY,
    'MINIO_SECRET_KEY': MINIO_SECRET_KEY,
    'S3_BUCKET': S3_BUCKET
}
# (Mục 1) Tên Docker network mà Airflow và MinIO cùng tham gia
# (Quan trọng để container có thể kết nối tới minio:9000)
DOCKER_NETWORK = os.getenv('DOCKER_NETWORK', 'my-network') 
# ----------------------------------------------------


def decide_which_path(**kwargs):
    """Đọc kết quả từ XCom (từ DockerOperator) và quyết định luồng tiếp theo."""
    ti = kwargs['ti']
    # DockerOperator đẩy log cuối cùng vào XCom
    result = ti.xcom_pull(task_ids='check_data_drift_task', key='return_value').strip()
    logging.info(f"Drift check result from XCom: {result}")
    if result == 'trigger_retrain':
        return 'trigger_retraining_job'
    return 'no_drift_detected_task'

# --- THÊM MỚI (Mục 2): Hàm đánh giá và thúc đẩy mô hình ---
def evaluate_and_promote_model(**kwargs):
    """
    So sánh mô hình mới (từ run_id) với mô hình 'production'.
    Nếu tốt hơn -> Thúc đẩy (promote) và trigger restart.
    Nếu không -> Bỏ qua.
    """
    ti = kwargs['ti']
    # Lấy version (run_id) được truyền cho Jenkins
    run_id = ti.xcom_pull(task_ids='trigger_retraining_job', key='run_id')
    if not run_id:
        # Nếu task Jenkins fail, run_id có thể là None
        logging.error("Không thể lấy run_id từ task Jenkins. Dừng quá trình.")
        return 'model_promotion_failed' # Cần 1 task Dummy cho việc này

    new_version_id = run_id
    new_metrics_key = f'models/{new_version_id}/metrics.json'
    prod_metrics_key = 'models/production/metrics.json'

    logging.info(f"Bắt đầu đánh giá mô hình phiên bản: {new_version_id}")

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY
        )

        # 1. Lấy metrics của mô hình mới
        obj_new = s3.get_object(Bucket=S3_BUCKET, Key=new_metrics_key)
        metrics_new_data = json.loads(obj_new['Body'].read().decode('utf-8'))
        new_loss = metrics_new_data.get('val_loss') # Sử dụng val_loss từ script huấn luyện
        if new_loss is None:
            raise ValueError(f"Không tìm thấy 'val_loss' trong {new_metrics_key}")
        logging.info(f"Mô hình mới (v: {new_version_id}) có val_loss: {new_loss}")

        # 2. Lấy metrics của mô hình production (nếu có)
        current_prod_loss = float('inf') # Mặc định loss là vô cực
        try:
            obj_prod = s3.get_object(Bucket=S3_BUCKET, Key=prod_metrics_key)
            metrics_prod_data = json.loads(obj_prod['Body'].read().decode('utf-8'))
            current_prod_loss = metrics_prod_data.get('val_loss', float('inf'))
            logging.info(f"Mô hình 'production' hiện tại có val_loss: {current_prod_loss}")
        except s3.exceptions.NoSuchKey:
            logging.warning("Không tìm thấy mô hình 'production'. Bất kỳ mô hình mới nào cũng sẽ được thúc đẩy.")

        # 3. So sánh và Quyết định
        if new_loss < current_prod_loss:
            logging.warning(f"THÚC ĐẨY: Mô hình mới (loss={new_loss}) tốt hơn 'production' (loss={current_prod_loss}).")
            
            # Hàm helper để sao chép S3
            def copy_s3_object(s3_client, bucket, source_key, dest_key):
                copy_source = {'Bucket': bucket, 'Key': source_key}
                s3_client.copy_object(CopySource=copy_source, Bucket=bucket, Key=dest_key)

            # Sao chép tất cả artifacts từ 'new_version_id' sang 'production'
            artifacts = [
                'best_cnn_lstm_model.keras',
                'scaler_car_count.pkl',
                'scaler_hour.pkl',
                'metrics.json' # Cập nhật cả metrics
            ]
            for artifact in artifacts:
                source_key = f'models/{new_version_id}/{artifact}'
                dest_key = f'models/production/{artifact}'
                copy_s3_object(s3, S3_BUCKET, source_key, dest_key)
                logging.info(f"Đã sao chép {source_key} -> {dest_key}")
            
            logging.info("Thúc đẩy mô hình hoàn tất. Trigger restart service...")
            return 'trigger_service_restart_job' # Tên taskID tiếp theo
        else:
            logging.info(f"BỎ QUA: Mô hình mới (loss={new_loss}) không tốt hơn 'production' (loss={current_prod_loss}).")
            return 'model_not_promoted_task' # Tên taskID tiếp theo

    except Exception as e:
        logging.error(f"Lỗi trong quá trình đánh giá và thúc đẩy: {e}")
        return 'model_promotion_failed'
# ----------------------------------------------------

with DAG(
    dag_id='parking_drift_and_retrain_dag_v3_improved', # Đổi tên DAG
    schedule='0 1 * * *',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    doc_md="""
    ### (ĐÃ CẢI TIẾN) DAG Điều phối MLOps cho Bãi đỗ xe
    - **Mục đích:** Tự động kiểm tra data drift và quản lý vòng đời mô hình.
    - **Luồng hoạt động (Cải tiến):**
    1. Chạy `drift_detector.py` trong một **Docker container riêng biệt** (Mục 1).
    2. Nếu phát hiện drift:
        a. Kích hoạt job Jenkins `parking-model-retrain` (truyền `run_id` làm version).
        b. **(MỚI)** Chạy task `evaluate_and_promote_model` để so sánh mô hình mới với mô hình 'production' trong MinIO.
        c. **(MỚI)** Nếu mô hình mới tốt hơn:
            i. Thúc đẩy (promote) mô hình mới (copy S3: `models/<version>` -> `models/production/`).
            ii. Kích hoạt job Jenkins `parking-service-restart` để deploy (service sẽ tự động tải từ `models/production/`).
        d. **(MỚI)** Nếu mô hình mới tệ hơn: Dừng lại và không deploy.
    3. Nếu không có drift, quy trình kết thúc.
    """,
    tags=['mlops', 'parking-v3', 'improved'],
) as dag:

    start_task = BashOperator(
        task_id='start_task',
        bash_command='echo "Bắt đầu quy trình kiểm tra drift v3..."',
    )

    # --- (Mục 1) CẢI TIẾN: Thay thế BashOperator bằng DockerOperator ---
    check_data_drift_task = DockerOperator(
        task_id='check_data_drift_task',
        image='python:3.10-slim', # Sử dụng image python (nên build image riêng có sẵn thư viện)
        command="bash -c 'pip install pandas boto3 scipy && python /scripts/mlops/drift_detector.py'",
        mounts=[
            # Mount script từ máy chủ Airflow vào container
            Mount(source=DRIFT_SCRIPT_PATH, target='/scripts/mlops/drift_detector.py', type='bind', read_only=True)
        ],
        environment=DOCKER_ENV_VARS, # Cung cấp env vars cho MinIO
        network_mode=DOCKER_NETWORK, # Cung cấp network để kết nối MinIO
        auto_remove=True, # Tự động xóa container sau khi chạy
        do_xcom_push=True, # Đẩy stdout (no_drift / trigger_retrain) vào XCom
        docker_url="unix://var/run/docker.sock", # Mặc định, có thể cần thay đổi
        mount_tmp_dir=False,
    )
    # -----------------------------------------------------------------

    branching_task = BranchPythonOperator(
        task_id='branch_on_drift_result',
        python_callable=decide_which_path,
    )

    no_drift_detected_task = BashOperator(
        task_id='no_drift_detected_task',
        bash_command='echo "Không phát hiện drift. Kết thúc quy trình."',
    )

    # --- (Mục 2) CẢI TIẾN: Truyền 'run_id' làm tham số phiên bản ---
    trigger_retraining_job = JenkinsJobTriggerOperator(
        task_id='trigger_retraining_job',
        job_name=JENKINS_RETRAIN_JOB,
        jenkins_connection_id=JENKINS_CONNECTION_ID,
        wait_for_completion=True,
        # Truyền run_id của Airflow DAG cho Jenkins để làm version ID
        parameters={"MODEL_VERSION": "{{ run_id }}"},
        # Đẩy run_id ra XCom để task đánh giá có thể sử dụng
        do_xcom_push=True, 
    )
    # -------------------------------------------------------------

    # --- (Mục 2) CẢI TIẾN: Thêm bước đánh giá (Branching) ---
    evaluate_and_promote_task = BranchPythonOperator(
        task_id='evaluate_and_promote_task',
        python_callable=evaluate_and_promote_model,
    )
    # -------------------------------------------------------

    trigger_service_restart_job = JenkinsJobTriggerOperator(
        task_id='trigger_service_restart_job',
        job_name=JENKINS_RESTART_JOB,
        jenkins_connection_id=JENKINS_CONNECTION_ID,
        wait_for_completion=True,
    )

    # --- (Mục 2) CẢI TIẾN: Thêm các task Dummy cho các luồng mới ---
    model_not_promoted_task = BashOperator(
        task_id='model_not_promoted_task',
        bash_command='echo "Mô hình mới không tốt hơn, không thúc đẩy. Dừng luồng."'
    )

    model_promotion_failed = DummyOperator(
        task_id='model_promotion_failed',
    )
    # ---------------------------------------------------------

    # --- (Mục 2) CẢI TIẾN: Cập nhật luồng DAG ---
    start_task >> check_data_drift_task >> branching_task
    
    # Luồng 1: Không drift
    branching_task >> no_drift_detected_task
    
    # Luồng 2: Có drift, huấn luyện, đánh giá, và deploy (nếu tốt)
    branching_task >> trigger_retraining_job >> evaluate_and_promote_task
    
    evaluate_and_promote_task >> trigger_service_restart_job # Nếu tốt
    evaluate_and_promote_task >> model_not_promoted_task # Nếu không tốt
    evaluate_and_promote_task >> model_promotion_failed # Nếu lỗi