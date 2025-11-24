import boto3
import time
from datetime import datetime
import random 

# --- CẤU HÌNH ---
REGION_NAME = 'ap-southeast-1'
TABLE_NAME = 'SmartParkingRawData'
SENSOR_ID = 'camera-01' 

# Khởi tạo kết nối DynamoDB
# Pi sẽ tự tìm credential trong ~/.aws/credentials hoặc biến môi trường
try:
    dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
    table = dynamodb.Table(TABLE_NAME)
    print("✅ Kết nối DynamoDB thành công!")
except Exception as e:
    print(f"❌ Lỗi kết nối AWS: {e}")
    exit(1)

def get_ai_car_count():
    """
    Thay hàm này bằng code AI nhận diện thật của bạn.
    Hiện tại đang trả về số ngẫu nhiên để test.
    """
    # Ví dụ: return yolo_model.detect(image)
    return random.randint(0, 20) 

def send_to_cloud(car_count):
    """
    Gửi dữ liệu lên DynamoDB
    """
    # Lấy thời gian hiện tại (ISO 8601)
    now_iso = datetime.now().isoformat()
    
    print(f"🚀 [{now_iso}] Camera: {SENSOR_ID} | Xe: {car_count} -> Đang gửi...")
    
    try:
        table.put_item(
            Item={
                'sensor_id': SENSOR_ID,  
                'timestamp': now_iso,   
                'car_count': int(car_count)
            }
        )
        print("✅ Đã gửi xong.")
    except Exception as e:
        print(f"❌ Gửi thất bại: {e}")

# --- VÒNG LẶP CHÍNH ---
if __name__ == "__main__":
    print("--- BẮT ĐẦU CHƯƠNG TRÌNH SMART PARKING ---")
    
    try:
        while True:
            # 1. Nhận diện 
            count = get_ai_car_count()
            
            # 2. Gửi lên Cloud
            send_to_cloud(count)
            
            # 3. Nghỉ 5 phút (300 giây) trước lần gửi tiếp theo
            print("💤 Chờ 5 phút...")
            time.sleep(300) 
            
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng chương trình.")