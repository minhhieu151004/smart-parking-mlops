# **Dự án Bãi đỗ xe thông minh với MLOps**

**Đây là dự án khóa luận tốt nghiệp xây dựng một hệ thống giám sát bãi đỗ xe thông minh, có khả năng dự đoán nhu cầu trong tương lai và được triển khai với các thực hành MLOps tiên tiến trên nền tảng AWS.**

## **Kiến trúc Hệ thống**

* **Data/Model Storage (Lưu trữ): Amazon S3**  
   * **Lưu trữ dữ liệu đỗ xe lịch sử (parking\_data.csv)**  
   * **Lưu trữ tất cả các phiên bản mô hình đã huấn luyện (theo ID phiên bản).**
   * **Lưu trữ các mô hình "production" (đang chạy chính thức).**
   * **Lưu trữ các file metrics (metrics.json) và kết quả kiểm tra drift.**
* **Model Service: Amazon EC2** 
   * **Chạy một container Docker chứa API dự đoán (Flask + Gunicorn).**
   * **Sử dụng IAM Role (ec2-s3-access-role) để tự động truy cập S3 và CloudWatch.**
* ***CI/CD - Ứng dụng (Luồng 1): GitHub Actions (cicd.yml)**
   * **Tự động build và push Docker image (model-service) lên Docker Hub khi có git push vào thư mục model\_service/.**
   * **Tự động SSH vào EC2 và triển khai container model-service mới nhất.**
* **Workflow Orchestration - MLOps (Luồng 2):**
   1. **Amazon EventBridge (Scheduler): Kích hoạt quy trình MLOps hàng ngày (ví dụ: 1 giờ sáng)**
   2. **Amazon SageMaker Pipelines: Điều phối chính**
      * **Drift Check Job: Chạy script drift\_detector\_sagemaker.py để kiểm tra data drift.**
      * **Condition Step: Chỉ tiếp tục nếu phát hiện có drift.**
      * **Training Job: Chạy script train\_pipeline\_sagemaker.py để huấn luyện mô hình mới. SageMaker tự động quản lý tài nguyên (CPU/GPU) cho việc training.**
   3. **Amazon EventBridge (Events): Lắng nghe sự kiện "SageMaker Pipeline Succeeded" (pipeline chạy thành công).**
   4. **AWS Lambda (evaluate-promote-trigger)**
      * **Được kích hoạt bởi EventBridge.**
      * **Evaluate: Tải metrics.json của model mới (từ SageMaker) và model cũ (từ S3) để so sánh val\_loss.**
      * **Promote: Nếu model mới tốt hơn, copy các file (.keras, .pkl) vào thư mục models/production/ trên S3.**
      * **Trigger: Gửi API call đến GitHub để kích hoạt luồng mlops-jobs.yml.**
* **CI/CD - MLOps Restart (Luồng 2 tiếp): GitHub Actions (mlops-jobs.yml)**
   * **Chỉ chứa 1 job restart, được kích hoạt bởi Lambda.**
   * **SSH vào EC2 và chạy lệnh docker restart model-service.**
   * **Container model-service khởi động lại, tự động tải mô hình mới nhất từ S3.**
* **Monitoring (Giám sát): Amazon CloudWatch**
   * **Thu thập logs của model-service (thông qua awslogs driver của Docker).**
   * **Thu thập logs của SageMaker Jobs và Lambda.**
   * **Thu thập custom metrics (PredictionLatency, PredictedCarCount) do model\_service.py chủ động gửi lên.**
   
## **Cấu trúc Thư mục**

* **.github/workflows/:**
   * **cicd.yml: (Luồng 1) Tự động build/deploy model-service.**
   * **mlops-jobs.yml: (Luồng 2) Chỉ chứa job restart do Lambda gọi.**
* **lambda/evaluate-promote-trigger/:**
   * **lambda\_function.py: Code Python cho logic Evaluate, Promote, Trigger.**
   * **requirements.txt: Chứa thư viện requests cho Lambda.**
* **mlops/:**
   * **drift\_detector.py: Script phát hiện drift cho SageMaker.**
   * **train\_pipeline.py: Script huấn luyện cho SageMaker.**
   * **build\_pipeline.py: Script để định nghĩa và tạo SageMaker Pipeline.**
* **model\_service/:**
   * **model\_service.py: Code API (Flask) đã tích hợp S3 (IAM Role) và CloudWatch.**
   * **requirements.txt: Các thư viện Python.**
   * **Dockerfile: File Docker để build model\_service.**

## **Các Luồng MLOps**

1. **Luồng Cập nhật Ứng dụng (CI/CD):**
   * **Trigger: git push vào thư mục model\_service/.**
   * **Quy trình: GHA cicd.yml chạy \-\> Build & Push Image \-\> SSH vào EC2 \-\> docker run (tải image mới).**
2. **Luồng Cập nhật Mô hình (CD4ML - Serverless):**
   * **Trigger: EventBridge Schedule (hàng ngày).**
   * **Quy trình:**
      1. **EventBridge kích hoạt SageMaker Pipeline.**
      2. **Pipeline chạy Drift Check.**
      3. **(Nếu có Drift) Pipeline chạy Training Job.**
      4. **(Nếu Pipeline thành công) EventBridge (Event) kích hoạt Lambda.**
      5. **Lambda chạy Evaluate (so sánh metrics).**
      6. **(Nếu Model mới tốt hơn) Lambda chạy Promote (copy S3) và Trigger GHA.**
      7. **GHA mlops-jobs.yml chạy -> SSH vào EC2 -> docker restart model-service.**
      8. **model-service khởi động lại và tải model mới từ S3.**
