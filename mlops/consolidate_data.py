import argparse
import os
import pandas as pd
import glob
import logging

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Tham số output quan trọng để gửi sang bước Train
    parser.add_argument("--output-drift-data", type=str, default="/opt/ml/processing/drift_data")
    
    # Các tham số khác 
    parser.add_argument("--baseline-data-uri", type=str, default="") 
    parser.add_argument("--data-bucket", type=str, default="")
    parser.add_argument("--actual-prefix", type=str, default="") 
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    
    args = parser.parse_args()

    logging.info("--- BẮT ĐẦU: TRÍCH XUẤT DỮ LIỆU DRIFT (FINE-TUNING) ---")
    
    # 1. Đường dẫn Input (Nơi SageMaker mount thư mục daily_actuals)
    input_daily_dir = "/opt/ml/processing/input_daily"
    
    # 2. Tìm file CSV mới nhất (theo tên file YYYY-MM-DD.csv là sort được)
    csv_files = glob.glob(os.path.join(input_daily_dir, "*.csv"))
    
    if not csv_files:
        logging.error(f"❌ Lỗi: Không tìm thấy file dữ liệu nào trong {input_daily_dir}")
        # Tạo file rỗng để Pipeline không bị crash, nhưng logic sau sẽ dừng
        df_drift = pd.DataFrame(columns=['timestamp', 'car_count'])
    else:
        # Sắp xếp để lấy file mới nhất (ngày hôm qua/hôm nay)
        latest_file = sorted(csv_files)[-1]
        logging.info(f"📅 Phát hiện file dữ liệu mới nhất: {os.path.basename(latest_file)}")
        
        # 3. Đọc dữ liệu
        df_drift = pd.read_csv(latest_file)
        
        # Xử lý format timestamp nếu cần (để khớp với train_pipeline)
        if 'timestamp' in df_drift.columns:
            df_drift['timestamp'] = pd.to_datetime(df_drift['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')

        logging.info(f"✅ Đã load {len(df_drift)} dòng dữ liệu để Fine-tune.")

    # 4. Lưu file output (train.csv)
    os.makedirs(args.output_drift_data, exist_ok=True)
    drift_output_file = os.path.join(args.output_drift_data, "train.csv")
    
    df_drift.to_csv(drift_output_file, index=False)
    logging.info(f"💾 Đã lưu file training vào: {drift_output_file}")

    logging.info("--- HOÀN TẤT ---")