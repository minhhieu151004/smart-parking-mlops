from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.jenkins.operators.jenkins_job_trigger import JenkinsJobTriggerOperator

# --- Các biến cấu hình ---
# THAY ĐỔI CÁC GIÁ TRỊ NÀY CHO PHÙ HỢP
JENKINS_CONNECTION_ID = "jenkins_default"  # ID kết nối Jenkins trong Airflow UI
JENKINS_RETRAIN_JOB = "parking-model-retrain"     # Tên job huấn luyện trên Jenkins
JENKINS_RESTART_JOB = "parking-service-restart"     # Tên job khởi động lại trên Jenkins
# Đường dẫn này giả định bạn đã mount hoặc copy script vào container của Airflow
DRIFT_SCRIPT_PATH = "/opt/airflow/scripts/mlops/drift_detector.py"

def decide_which_path(**kwargs):
    """Đọc kết quả từ XCom và quyết định luồng tiếp theo."""
    ti = kwargs['ti']
    result = ti.xcom_pull(task_ids='check_data_drift_task', key='return_value').strip()
    if result == 'trigger_retrain':
        return 'trigger_retraining_job'
    return 'no_drift_detected_task'

with DAG(
    dag_id='parking_drift_and_retrain_dag_v2',
    schedule='0 1 * * *',  # Chạy vào 1:00 AM mỗi ngày
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    doc_md="""
    ### DAG Điều phối MLOps cho Bãi đỗ xe
    - **Mục đích:** Tự động kiểm tra sự thay đổi (drift) trong phân phối dữ liệu đỗ xe hàng ngày.
    - **Luồng hoạt động:**
        1.  Chạy script `drift_detector.py`.
        2.  Nếu phát hiện drift:
            a. Kích hoạt job Jenkins `parking-model-retrain` để huấn luyện lại mô hình.
            b. Sau khi huấn luyện xong, kích hoạt job Jenkins `parking-service-restart` để áp dụng mô hình mới.
        3.  Nếu không có drift, quy trình kết thúc.
    """,
    tags=['mlops', 'parking-v2'],
) as dag:

    start_task = BashOperator(
        task_id='start_task',
        bash_command='echo "Bắt đầu quy trình kiểm tra drift..."',
    )

    # Task này sẽ chạy script kiểm tra drift.
    # Lưu ý: Môi trường chạy Airflow (worker) cần có các thư viện này.
    # Một cách tốt hơn là chạy trong một virtual environment hoặc container riêng.
    check_data_drift_task = BashOperator(
        task_id='check_data_drift_task',
        bash_command=f'pip install pandas boto3 scipy && python {DRIFT_SCRIPT_PATH}',
        do_xcom_push=True,
    )

    branching_task = BranchPythonOperator(
        task_id='branch_on_drift_result',
        python_callable=decide_which_path,
    )

    trigger_retraining_job = JenkinsJobTriggerOperator(
        task_id='trigger_retraining_job',
        job_name=JENKINS_RETRAIN_JOB,
        jenkins_connection_id=JENKINS_CONNECTION_ID,
        wait_for_completion=True, # QUAN TRỌNG: Chờ Jenkins huấn luyện xong
    )

    trigger_service_restart_job = JenkinsJobTriggerOperator(
        task_id='trigger_service_restart_job',
        job_name=JENKINS_RESTART_JOB,
        jenkins_connection_id=JENKINS_CONNECTION_ID,
        wait_for_completion=True, # Chờ Jenkins khởi động lại xong
    )

    no_drift_detected_task = BashOperator(
        task_id='no_drift_detected_task',
        bash_command='echo "Không phát hiện drift. Kết thúc quy trình."',
    )

    # Định nghĩa thứ tự các task
    start_task >> check_data_drift_task >> branching_task
    branching_task >> no_drift_detected_task
    branching_task >> trigger_retraining_job >> trigger_service_restart_job

