import argparse
import os
import pandas as pd
import glob
import logging

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-data-dir", type=str, default="/opt/ml/processing/drift_data")
    parser.add_argument("--original-test-dir", type=str, default="/opt/ml/processing/original_test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    
    args = parser.parse_args()

    logging.info("--- BẮT ĐẦU: CẬP NHẬT TẬP TEST ---")

    # 1. Load Dữ liệu Drift (Dữ liệu ngày mới nhất)
    # Tìm file CSV trong thư mục drift_data
    drift_files = glob.glob(os.path.join(args.drift_data_dir, "*.csv"))
    if not drift_files:
        logging.warning("⚠️ Không tìm thấy dữ liệu Drift. Sẽ chỉ dùng tập test gốc.")
        df_drift = pd.DataFrame()
    else:
        logging.info(f"Đọc dữ liệu Drift từ: {drift_files[0]}")
        df_drift = pd.read_csv(drift_files[0])

    # 2. Load Tập Test Gốc
    # Tìm file CSV trong thư mục original_test
    test_files = glob.glob(os.path.join(args.original_test_dir, "*.csv"))
    if not test_files:
        logging.error("❌ Lỗi: Không tìm thấy tập Test gốc trên S3.")
        # Trường hợp này critical, nhưng ta sẽ handle mềm để ko crash pipeline nếu test
        df_original = pd.DataFrame()
    else:
        logging.info(f"Đọc tập Test gốc từ: {test_files[0]}")
        df_original = pd.read_csv(test_files[0])

    # 3. Nối dữ liệu (Concatenate)
    if not df_drift.empty and not df_original.empty:
        # Đảm bảo format timestamp thống nhất để sort
        if 'timestamp' in df_drift.columns:
            df_drift['timestamp'] = pd.to_datetime(df_drift['timestamp'], dayfirst=True, errors='coerce')
        if 'timestamp' in df_original.columns:
            df_original['timestamp'] = pd.to_datetime(df_original['timestamp'], dayfirst=True, errors='coerce')

        # Nối: Tập gốc + Dữ liệu mới
        df_updated = pd.concat([df_original, df_drift], ignore_index=True)
        
        # Loại bỏ trùng lặp (nếu có) và Sắp xếp lại
        df_updated = df_updated.drop_duplicates(subset=['timestamp'], keep='last')
        df_updated = df_updated.sort_values('timestamp')
        
        logging.info(f"✅ Đã nối thành công. Kích thước cũ: {len(df_original)}, Mới: {len(df_updated)}")
    
    elif not df_original.empty:
        logging.info("Chỉ sử dụng tập Test gốc (không có drift data).")
        df_updated = df_original
    else:
        logging.warning("Cả tập test gốc và drift data đều rỗng/thiếu!")
        df_updated = pd.DataFrame(columns=['car_count', 'timestamp'])

    # 4. Lưu kết quả
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "updated_test_set.csv")
    
    # Format lại timestamp về string chuẩn trước khi lưu
    if 'timestamp' in df_updated.columns:
        df_updated['timestamp'] = df_updated['timestamp'].dt.strftime('%d/%m/%Y %H:%M:%S')

    # Chỉ giữ các cột quan trọng
    cols_to_save = ['car_count', 'timestamp']
    # Nếu có cột 'hour', giữ lại cũng được, nhưng tối thiểu phải có 2 cột trên
    df_updated = df_updated[cols_to_save] if set(cols_to_save).issubset(df_updated.columns) else df_updated

    df_updated.to_csv(output_path, index=False)
    logging.info(f"💾 Đã lưu Tập Test Mới vào: {output_path}")

    logging.info("--- HOÀN TẤT CẬP NHẬT TẬP TEST ---")