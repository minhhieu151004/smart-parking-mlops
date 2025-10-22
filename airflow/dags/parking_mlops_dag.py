from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.jenkins.operators.jenkins_job_trigger import JenkinsJobTriggerOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

def run_drift_check_and_decide(**kwargs):
    """
    Chạy script kiểm tra drift và trả về ID của task tiếp theo.
    Lưu ý: Bạn cần cài đặt các thư viện (pandas, boto3, scipy) vào môi trường Airflow
    hoặc sử dụng KubernetesPodOperator/DockerOperator để chạy trong môi trường riêng.
    Ở đây, chúng ta giả định chạy bằng BashOperator.
    """
    # ti = kwargs['ti']
    # result = ti.xcom_pull(task_ids='run_drift_check_script', key='return_value')
    # Giả sử dòng cuối cùng của log là kết quả
    # Logic thực tế có thể phức tạp hơn để parse log
    # Trong ví dụ này, chúng ta sẽ để BranchPythonOperator đọc trực tiếp từ file log
    # Hoặc đơn giản hơn là tạo một task riêng để đọc và push XCom
    
    # Cách đơn giản nhất: Dùng XCom
    result = kwargs['ti'].xcom_pull(task_ids='check_drift_task', key='return_value')
    if result and 'trigger_retrain' in str(result):
        return 'trigger_retraining_job'
    return 'no_drift_detected'


with DAG(
    'parking_drift_and_retrain_dag',
    default_args=default_args,
    description='Kiểm tra data drift hàng ngày và huấn luyện lại nếu cần',
    schedule_interval='@daily',  # Chạy mỗi ngày
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'parking'],
) as dag:

    start = BashOperator(
        task_id='start',
        bash_command='echo "Bắt đầu quy trình kiểm tra drift..."',
    )

    # Chạy script kiểm tra drift
    # Giả sử script `drift_detector.py` nằm trong thư mục /opt/airflow/scripts
    check_drift_task = BashOperator(
        task_id='check_drift_task',
        bash_command='python /opt/airflow/scripts/drift_detector.py',
        do_xcom_push=True, # Tự động push dòng output cuối cùng vào XCom
    )

    # Dựa vào kết quả để rẽ nhánh
    branching = BranchPythonOperator(
        task_id='branch_on_drift_result',
        python_callable=run_drift_check_and_decide,
    )

    # Kích hoạt job Jenkins để huấn luyện lại
    # Bạn cần cấu hình kết nối Jenkins trong Airflow UI
    trigger_retraining_job = JenkinsJobTriggerOperator(
        task_id='trigger_retraining_job',
        job_name='Tên Job Huấn luyện trên Jenkins', # Ví dụ: 'parking-model-training'
        jenkins_connection_id='your_jenkins_connection', # ID kết nối bạn tạo
    )

    no_drift_detected = BashOperator(
        task_id='no_drift_detected',
        bash_command='echo "Không phát hiện drift. Kết thúc quy trình."',
    )

    start >> check_drift_task >> branching >> [trigger_retraining_job, no_drift_detected]
