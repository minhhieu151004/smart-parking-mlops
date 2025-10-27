FROM python:3.10-slim

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY model_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_service/model_service.py .

EXPOSE 5000

# Lệnh để chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "model_service:app"]
