# **Dự án Bãi đỗ xe thông minh với MLOps**

**Đây là dự án xây dựng một hệ thống giám sát bãi đỗ xe thông minh phi máy chủ (Serverless), có khả năng dự đoán nhu cầu đỗ xe của người dùng trong tương lai và được tích hợp triển khai với các thành phần tự động hóa MLOps trên nền tảng AWS.**

## **Kiến trúc Hệ thống**

![](images/system_flow.png)

* **Data/Model Storage: Amazon S3**  
   * **Lưu trữ dữ liệu lịch sử đỗ xe và kết quả dự đoán từ endpoint.**   
   * **Lưu trữ các artifacts (model, metrics, drift reports) của SageMaker Pipeline.**
   * **Lưu trữ các file metrics (metrics.json) và kết quả kiểm tra drift.**

* **Model Serving: Amazon SageMaker Serverless Endpoint**
   * **Deploy model, chỉ hoạt động khi được trigger.**
   * **Auto-scalling.**

* **Workflow Orchestration (MLOps): Amazon SageMaker Pipelines**
   * **Check dift >> Condition Step >> ConsolidateData >> Train >> Register Model**
   * **CheckDataDrift: Chạy script drift_detector.py để so sánh phân phối dữ liệu mới so với dữ liệu cũ.**
   * **Condition Step: Quyết định xem có cần huấn luyện lại không (dựa trên kết quả drift).**
   * **ConsolidateData: Chạy script consolidate_data.py để gộp dữ liệu mới vào dataset chính.**
   * **TrainParkingModel: Chạy script train_pipeline.py để huấn luyện model trên tài nguyên SageMaker Training Job.**
   * **RegisterModel: Đăng ký model mới vào Model Registry với trạng thái PendingManualApproval.**

* **Automation: AWS Lambda & EventBridge**
   * **run-prediction-trigger (Lambda): Kích hoạt dự đoán khi có thay đổi trong "daily_actuals/".**
   * **evaluate-promote-trigger (Lambda): Kích hoạt đánh giá model khi có thay đổi trạng thái trong Model Registry (sau khi chạy pipeline).**

* **Monitoring: Amazon CloudWatch**
   * **Giám sát logs, gửi cảnh báo.**

## **Các Luồng Hoạt Động (Workflows)**

1. **Luồng Dự đoán Thời gian thực (Real-time Prediction).**
   * **Raspberry PI gửi kết quả nhận diện, lưu trữ vào S3 "daily_actuals/".**
   * **EventBridge kích hoạt hàm Lambda "run-prediction-trigger".**
   * **Lambda gọi SageMaker Endpoint.**
   * **Lưu kết quả dự đoán vào S3 và ứng dụng Web.**

2. **Luồng Huấn luyện & Cập nhật Tự động (Automated Retraining).**
   * **EventBridge Scheduler kích hoạt SageMaker Pipeline hằng ngày.**

3. **Workflow GitHub Action**
   * **Deploy Lambda: Cập nhật 2 Lambda functions khi có thay đổi trong thư mục "lambda/".**
   * **Build SageMaker Pipeline: Cập nhật và chạy lại pipeline mỗi khi có thay đổi trong thư mục "mlops/".**