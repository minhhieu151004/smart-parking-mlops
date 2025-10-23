FROM python:3.10-slim

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements trước để tận dụng Docker cache
COPY model_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn của service
COPY model_service/model_service.py .

# Mở port 5000
EXPOSE 5000

# Lệnh để chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "model_service:app"]
