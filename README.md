# Dự án Bãi đỗ xe thông minh với MLOps

**Đây là dự án khóa luận tốt nghiệp xây dựng một hệ thống giám sát bãi đỗ xe thông minh, có khả năng dự đoán nhu cầu trong tương lai và được triển khai với các thực hành MLOps tiên tiến trên nền tảng AWS.**

## Kiến trúc Hệ thống

• **Data/Model Storage (Lưu trữ): Amazon S3**  
   • Lưu trữ dữ liệu đỗ xe lịch sử (`parking_data.csv`)  
   • Lưu trữ tất cả các phiên bản mô hình đã huấn luyện (theo ID phiên bản)  
   • Lưu trữ mô hình “production” (đang chạy chính thức)  
   • Lưu trữ các file metrics (`metrics.json`) và kết quả kiểm tra drift  

• **Model Service: Amazon EC2**  
   • Chạy container Docker chứa API dự đoán (Flask + Gunicorn)  
   • Sử dụng IAM Role (`ec2-s3-access-role`) để truy cập S3 và CloudWatch  

• **CI/CD - Ứng dụng (Luồng 1): GitHub Actions (`cicd.yml`)**  
   • Tự động build và push Docker image (`model-service`) lên Docker Hub khi có `git push` vào thư mục `model_service/`  
   • Tự động SSH vào EC2 và triển khai container mới nhất  

• **Workflow Orchestration - MLOps (Luồng 2):**  
   1. **Amazon EventBridge (Scheduler):** Kích hoạt quy trình MLOps hàng ngày
   2. **Amazon SageMaker Pipelines:** Điều phối chính  
      • **Drift Check Job:** Chạy script `drift_detector_sagemaker.py` để kiểm tra data drift  
      • **Condition Step:** Chỉ tiếp tục nếu phát hiện có drift  
      • **Training Job:** Chạy script `train_pipeline_sagemaker.py` để huấn luyện mô hình mới, SageMaker tự động quản lý tài nguyên (CPU/GPU)  
   3. **Amazon EventBridge (Events):** Lắng nghe sự kiện “SageMaker Pipeline Succeeded”  
   4. **AWS Lambda (`evaluate-promote-trigger`):**  
      • Được kích hoạt bởi EventBridge  
      • **Evaluate:** Tải `metrics.json` của model mới và cũ từ S3 để so sánh `val_loss`  
      • **Promote:** Nếu model mới tốt hơn, copy file `.keras` và `.pkl` vào thư mục `models/production/` trên S3  
      • **Trigger:** Gửi API call đến GitHub để kích hoạt luồng `mlops-jobs.yml`  
   5. **CI/CD - MLOps Restart (Luồng 2 tiếp): GitHub Actions (`mlops-jobs.yml`)**  
      • SSH vào EC2 và chạy lệnh `docker restart model-service`  
      • Container `model-service` khởi động lại và tự động tải mô hình mới nhất từ S3  

• **Monitoring (Giám sát): Amazon CloudWatch**  
   • Thu thập logs của `model-service` (qua `awslogs` driver của Docker)  
   • Thu thập logs của SageMaker Jobs và Lambda  
   • Thu thập custom metrics (`PredictionLatency`, `PredictedCarCount`) do `model_service.py` gửi lên  

## Cấu trúc Thư mục

• **.github/workflows/**  
   • `cicd.yml`: Luồng 1 – Tự động build và deploy `model-service`  
   • `mlops-jobs.yml`: Luồng 2 – Chứa job restart do Lambda gọi  

• **lambda/evaluate-promote-trigger/**  
   • `lambda_function.py`: Logic Evaluate, Promote, Trigger  
   • `requirements.txt`: Chứa thư viện `requests` cho Lambda  

• **mlops/**  
   • `drift_detector.py`: Script phát hiện drift cho SageMaker  
   • `train_pipeline.py`: Script huấn luyện mô hình  
   • `build_pipeline.py`: Định nghĩa và tạo SageMaker Pipeline  

• **model_service/**  
   • `model_service.py`: API (Flask) tích hợp S3 (IAM Role) và CloudWatch  
   • `requirements.txt`: Thư viện Python cần thiết  
   • `Dockerfile`: File Docker để build `model_service`  

## Các Luồng MLOps

### Luồng Cập nhật Ứng dụng (CI/CD)

• **Trigger:** `git push` vào thư mục `model_service/`  
• **Quy trình:**  
   1. GitHub Actions (`cicd.yml`) chạy  
   2. Build và push Docker image lên Docker Hub  
   3. SSH vào EC2 và chạy container mới  

### Luồng Cập nhật Mô hình (CD4ML - Serverless)

• **Trigger:** EventBridge Schedule (hàng ngày)  
• **Quy trình:**  
   1. EventBridge kích hoạt SageMaker Pipeline  
   2. Pipeline chạy Drift Check  
   3. Nếu có drift → Training Job chạy  
   4. Khi Pipeline thành công → EventBridge kích hoạt Lambda  
   5. Lambda chạy Evaluate (so sánh metrics)  
   6. Nếu model mới tốt hơn → Lambda chạy Promote (copy S3) và Trigger GitHub  
   7. GitHub Actions (`mlops-jobs.yml`) chạy → SSH vào EC2 → `docker restart model-service`  
   8. `model-service` khởi động lại và tải mô hình mới từ S3  

