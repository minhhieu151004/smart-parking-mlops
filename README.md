# **Dự án Bãi đỗ xe thông minh với MLOps**

**Đây là dự án khóa luận tốt nghiệp xây dựng một hệ thống giám sát bãi đỗ xe thông minh, có khả năng dự đoán nhu cầu trong tương lai và được triển khai với các thực hành MLOps.**

## **Kiến trúc Hệ thống**

* **Edge Device: Raspberry Pi 5 chạy mô hình YOLO để nhận diện xe và vị trí trống.**  
* **Data/Model Storage: MinIO (hoặc AWS S3) để lưu trữ dữ liệu đỗ xe lịch sử và các phiên bản mô hình.**  
* **Model Serving: Một service Flask được đóng gói trong Docker, chịu trách nhiệm dự đoán và cung cấp metrics.**  
* **CI/CD: Jenkins tự động hóa việc build và deploy service.**  
* **Workflow Orchestration: Airflow lập lịch kiểm tra data drift và kích hoạt huấn luyện lại mô hình.**  
* **Monitoring: Prometheus & Grafana để giám sát hiệu suất hệ thống và mô hình.**

## **Cấu trúc Thư mục**

* **model\_service/: Chứa mã nguồn cho API dự đoán.**  
* **mlops/: Chứa các script phục vụ cho quy trình MLOps (huấn luyện, kiểm tra drift).**  
* **airflow/dags/: Chứa các DAG của Airflow.**  
* **Jenkinsfile: Định nghĩa pipeline CI/CD chính cho việc cập nhật ứng dụng.**

## **Các Luồng MLOps**

1. **Luồng Cập nhật Ứng dụng (CI/CD):**  
   * **Trigger: git push vào repository.**  
   * **Quy trình: Jenkins chạy Jenkinsfile \-\> Build Docker image mới \-\> Push lên Docker Hub \-\> Deploy container mới.**  
2. **Luồng Cập nhật Mô hình (CD4ML):**  
   * **Trigger: Airflow chạy hàng ngày (@daily).**  
   * **Quy trình: Airflow chạy script drift\_detector.py. Nếu phát hiện drift:**  
     1. **Airflow trigger job Jenkins parking-model-retrain.**  
     2. **Jenkins chạy train\_pipeline.py để huấn luyện lại mô hình và lưu vào MinIO.**  
     3. **Airflow trigger job Jenkins parking-service-restart để khởi động lại container, buộc nó tải mô hình mới.**