from __future__ import annotations
import pendulum
import os
import json
import boto3
import logging

from airflow.models.dag import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import HttpOperator
from docker.types import Mount

# --- Các biến cấu hình ---
# Đảm bảo các biến này được set trong Airflow UI (Admin -> Variables)
# Hoặc được set làm biến môi trường cho Airflow
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'admin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'password')
S3_BUCKET = os.getenv('S3_BUCKET', 'my-bucket')

# Tên repo GitHub của bạn
GITHUB_REPO = 'minhhieu151004/smart-parking-mlops'

# Đường dẫn script trên MÁY CHỦ AIRFLOW (để mount vào Docker)
DRIFT_SCRIPT_PATH = os.getenv('AIRFLOW_SCRIPT_PATH', '/opt/airflow/scripts/mlops/drift_detector.py')
DOCKER_NETWORK = os.getenv('DOCKER_NETWORK', 'my-network')

# Biến môi trường cho DockerOperator
DOCKER_ENV_VARS = {
    'MINIO_ENDPOINT': MINIO_ENDPOINT,
    'MINIO_ACCESS_KEY': MINIO_ACCESS_KEY,
    'MINIO_SECRET_KEY': MINIO_SECRET_KEY,
    'S3_BUCKET': S3_BUCKET
}

# --- Hàm Helper ---

def decide_which_path(**kwargs):
    """Đọc kết quả từ XCom (từ DockerOperator) và quyết định luồng tiếp theo."""
    ti = kwargs['ti']
    # DockerOperator đẩy log cuối cùng vào XCom
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
    Nếu tốt hơn -> Thúc đẩy (promote) và trigger restart.
    Nếu không -> Bỏ qua.
    """
    # Lấy run_id của DAG, đây sẽ là phiên bản mô hình
    run_id = kwargs['run_id']
    new_version_id = run_id
    
    logging.info(f"Bắt đầu đánh giá mô hình phiên bản: {new_version_id}")

    new_metrics_key = f'models/{new_version_id}/metrics.json'
    prod_metrics_key = 'models/production/metrics.json'

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
            return 'trigger_service_restart_job' # Tên taskID tiếp theo
        else:
            logging.info(f"BỎ QUA: Mô hình mới (loss={new_loss}) không tốt hơn 'production' (loss={current_prod_loss}).")
            return 'model_not_promoted_task' # Tên taskID tiếp theo

    except Exception as e:
        logging.error(f"Lỗi trong quá trình đánh giá và thúc đẩy: {e}")
        # Có thể do script huấn luyện fail, không tạo ra metrics.json
        return 'model_promotion_failed'

# --- Định nghĩa DAG ---

with DAG(
    dag_id='parking_mlops_github_actions_v2', # Đổi tên DAG
    schedule='0 1 * * *', # Chạy vào 1:00 AM mỗi ngày
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    doc_md="""
    ### (ĐÃ CẢI TIẾN) DAG MLOps sử dụng GitHub Actions
    - **Mục đích:** Tự động kiểm tra data drift và quản lý vòng đời mô hình.
    - **Luồng hoạt động (Cải tiến):**
    1. Chạy `drift_detector.py` trong một **Docker container riêng biệt**.
    2. Nếu phát hiện drift:
        a. Kích hoạt (trigger) GitHub Action `mlops-jobs.yml` với event `trigger-retrain`.
        b. **(MỚI)** Chạy task `evaluate_and_promote_model` để so sánh mô hình mới với mô hình 'production' trong MinIO.
        c. **(MỚI)** Nếu mô hình mới tốt hơn:
            i. Thúc đẩy (promote) mô hình mới (copy S3: `models/<version>` -> `models/production/`).
            ii. Kích hoạt GitHub Action `mlops-jobs.yml` với event `trigger-restart` để deploy.
        d. **(MỚI)** Nếu mô hình mới tệ hơn: Dừng lại và không deploy.
    3. Nếu không có drift, quy trình kết thúc.
    """,
    tags=['mlops', 'parking-v2', 'github-actions'],
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
            Mount(source=DRIFT_SCRIPT_PATH, target='/scripts/mlops/drift_detector.py', type='bind', read_only=True)
        ],
        environment=DOCKER_ENV_VARS,
        network_mode=DOCKER_NETWORK,
        auto_remove='success',
        do_xcom_push=True,
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    branching_task = BranchPythonOperator(
        task_id='branch_on_drift_result',
        python_callable=decide_which_path,
    )

    no_drift_detected_task = BashOperator(
        task_id='no_drift_detected_task',
        bash_command='echo "Không phát hiện drift. Kết thúc quy trình."',
    )

    # THAY THẾ JENKINS BẰNG GITHUB ACTIONS
    # Task này gọi API của GitHub để trigger workflow 'mlops-jobs.yml'
    trigger_retraining_job = HttpOperator(
        task_id='trigger_retraining_job',
        http_conn_id='github_api', # Đảm bảo bạn đã tạo Connection 'github_api' trong Airflow
        endpoint=f'/repos/{GITHUB_REPO}/dispatches',
        method='POST',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "event_type": "trigger-retrain", # Phải khớp với 'type' trong mlops-jobs.yml
            "client_payload": {
                "model_version": "{{ run_id }}" # Truyền run_id làm phiên bản
            }
        }),
        log_response=True,
    )

    evaluate_and_promote_task = BranchPythonOperator(
        task_id='evaluate_and_promote_task',
        python_callable=evaluate_and_promote_model,
    )

    # THAY THẾ JENKINS BẰNG GITHUB ACTIONS
    trigger_service_restart_job = HttpOperator(
        task_id='trigger_service_restart_job',
        http_conn_id='github_api', # Sử dụng cùng connection
        endpoint=f'/repos/{GITHUB_REPO}/dispatches',
        method='POST',
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "event_type": "trigger-restart" # Khớp với 'type' trong mlops-jobs.yml
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
    
    # Luồng 1: Không drift
    branching_task >> no_drift_detected_task
    
    # Luồng 2: Có drift, huấn luyện, đánh giá, và deploy (nếu tốt)
    branching_task >> trigger_retraining_job >> evaluate_and_promote_task
    
    evaluate_and_promote_task >> trigger_service_restart_job # Nếu tốt
    evaluate_and_promote_task >> model_not_promoted_task # Nếu không tốt
    evaluate_and_promote_task >> model_promotion_failed # Nếu lỗi