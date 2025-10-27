from __future__ import annotations
import pendulum
import os
import json
import boto3
import logging

from airflow.models.dag import DAG
# SỬA 1: Cập nhật các import đã cũ
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import HttpOperator
from docker.types import Mount

# --- Các biến cấu hình ---
S3_BUCKET = os.getenv('S3_BUCKET', 'kltn-smart-parking-data')

# Repo github
GITHUB_REPO = 'minhhieu151004/smart-parking-mlops'

# Đường dẫn script trên MÁY CHỦ AIRFLOW (EC2) (để mount vào Docker)
DRIFT_SCRIPT_PATH = os.getenv('AIRFLOW_SCRIPT_PATH', '/opt/airflow/scripts/mlops/drift_detector.py')
DOCKER_NETWORK = os.getenv('DOCKER_NETWORK', 'my-network')

DOCKER_ENV_VARS = {
    'S3_BUCKET': S3_BUCKET
}

# --- Hàm Helper ---

def decide_which_path(**kwargs):
    """Đọc kết quả từ XCom (từ DockerOperator) và quyết định luồng tiếp theo."""
    ti = kwargs['ti']
    result = ti.xcom_pull(task_ids='check_data_drift_task', key='return_value').strip()
    logging.info(f"Drift check result from XCom: {result}")
    
    if 'trigger_retrain' in result:
        logging.info("Phát hiện Drift. Sẽ trigger 'trigger_retraining_job'")
        return 'trigger_retraining_job'
        
    logging.info("Không phát hiện Drift. Sẽ trigger 'no_drift_detected_task'")
    return 'no_drift_detected_task'

def evaluate_and_promote_model(**kwargs):
    """
    So sánh mô hình mới (từ run_id) với mô hình 'production'.
    Hàm này chạy trên Airflow worker (EC2), nó sẽ tự động dùng IAM Role.
    """
    run_id = kwargs['run_id']
    new_version_id = run_id
    
    logging.info(f"Bắt đầu đánh giá mô hình phiên bản: {new_version_id}")

    new_metrics_key = f'models/{new_version_id}/metrics.json'
    prod_metrics_key = 'models/production/metrics.json'

    try:
        # Khởi tạo boto3 
        s3 = boto3.client('s3')

        # 1. Lấy metrics của mô hình mới
        obj_new = s3.get_object(Bucket=S3_BUCKET, Key=new_metrics_key)
        metrics_new_data = json.loads(obj_new['Body'].read().decode('utf-8'))
        new_loss = metrics_new_data.get('val_loss')
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
            logging.warning("Không tìm thấy mô hình 'production' (metrics.json). Mô hình mới sẽ được tự động thúc đẩy.")

        # 3. So sánh và Quyết định 
        if new_loss < current_prod_loss:
            logging.warning(f"THÚC ĐẨY: Mô hình mới (loss={new_loss}) tốt hơn 'production' (loss={current_prod_loss}).")
            
            def copy_s3_object(s3_client, bucket, source_key, dest_key):
                copy_source = {'Bucket': bucket, 'Key': source_key}
                s3_client.copy_object(CopySource=copy_source, Bucket=bucket, Key=dest_key)
                logging.info(f"Đã sao chép {source_key} -> {dest_key}")

            artifacts = [
                'best_cnn_lstm_model.keras',
                'scaler_car_count.pkl',
                'scaler_hour.pkl',
                'metrics.json'
            ]
            for artifact in artifacts:
                source_key = f'models/{new_version_id}/{artifact}'
                dest_key = f'models/production/{artifact}'
                copy_s3_object(s3, S3_BUCKET, source_key, dest_key)
            
            logging.info("Thúc đẩy mô hình hoàn tất. Trigger restart service...")
            return 'trigger_service_restart_job'
        else:
            logging.info(f"BỎ QUA: Mô hình mới (loss={new_loss}) không tốt hơn 'production' (loss={current_prod_loss}).")
            return 'model_not_promoted_task'

    except Exception as e:
        logging.error(f"Lỗi trong quá trình đánh giá và thúc đẩy: {e}")
        return 'model_promotion_failed'

# --- Định nghĩa DAG ---

with DAG(
    dag_id='parking_mlops_github_actions_v2', # Đổi tên DAG
    schedule='0 1 * * *', # Chạy vào 1:00 AM mỗi ngày
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    doc_md="""
    ### (EC2 + S3) DAG MLOps sử dụng GitHub Actions
    - **Mục đích:** Tự động kiểm tra data drift và quản lý vòng đời mô hình.
    - **Luồng hoạt động (Cải tiến):**
    1. Chạy `drift_detector.py` trong Docker (sử dụng IAM Role của EC2).
    2. Nếu phát hiện drift:
        a. Kích hoạt GHA `trigger-retrain` (GHA sẽ dùng Access Key để huấn luyện).
        b. Chạy `evaluate_and_promote_model` (sử dụng IAM Role) để so sánh model.
        c. Nếu mô hình mới tốt hơn:
            i. Thúc đẩy mô hình mới trên S3 (sử dụng IAM Role).
            ii. Kích hoạt GHA `trigger-restart` để deploy (restart service).
    3. Nếu không có drift, quy trình kết thúc.
    """,
    tags=['mlops', 'parking-v2', 'github-actions', 's3', 'ec2'],
) as dag:

    start_task = BashOperator(
        task_id='start_task',
        bash_command='echo "Bắt đầu quy trình kiểm tra drift (Trigger GHA)..."',
    )

    check_data_drift_task = DockerOperator(
        task_id='check_data_drift_task',
        image='python:3.10-slim',
        command="bash -c 'pip install pandas boto3 scipy && python /scripts/mlops/drift_detector.py'",
        mounts=[
            Mount(source=DRIFT_SCRIPT_PATH, target='/scripts/mlops/drift_detector.py', type='bind', read_only=True),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind')
        ],
        environment=DOCKER_ENV_VARS, 
        network_mode=DOCKER_NETWORK,
        auto_remove='success',
        do_xcom_push=True,
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        user=os.getuid() 
    )

    branching_task = BranchPythonOperator(
        task_id='branch_on_drift_result',
        python_callable=decide_which_path,
    )

    no_drift_detected_task = BashOperator(
        task_id='no_drift_detected_task',
        bash_command='echo "Không phát hiện drift. Kết thúc quy trình."',
    )

    # Task HttpOperator (trigger_retraining_job)
    trigger_retraining_job = HttpOperator(
        task_id='trigger_retraining_job',
        http_conn_id='github_api', 
        endpoint=f'/repos/{GITHUB_REPO}/dispatches',
        method='POST',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "event_type": "trigger-retrain", 
            "client_payload": {
                "model_version": "{{ run_id }}" 
            }
        }),
        log_response=True,
    )

    evaluate_and_promote_task = BranchPythonOperator(
        task_id='evaluate_and_promote_task',
        python_callable=evaluate_and_promote_model,
    )

    # Task HttpOperator (trigger_service_restart_job) 
    trigger_service_restart_job = HttpOperator(
        task_id='trigger_service_restart_job',
        http_conn_id='github_api', 
        endpoint=f'/repos/{GITHUB_REPO}/dispatches',
        method='POST',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "event_type": "trigger-restart"
        }),
        log_response=True,
    )

    model_not_promoted_task = BashOperator(
        task_id='model_not_promoted_task',
        bash_command='echo "Mô hình mới không tốt hơn, không thúc đẩy. Dừng luồng."'
    )

    model_promotion_failed = EmptyOperator(
        task_id='model_promotion_failed',
    )

    # Định nghĩa thứ tự các task 
    start_task >> check_data_drift_task >> branching_task
    
    branching_task >> no_drift_detected_task
    
    branching_task >> trigger_retraining_job >> evaluate_and_promote_task
    
    evaluate_and_promote_task >> trigger_service_restart_job 
    evaluate_and_promote_task >> model_not_promoted_task 
    evaluate_and_promote_task >> model_promotion_failed 
