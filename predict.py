import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Thêm thư mục gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import các module cần thiết
from src.pipeline import HybridPredictionSystem, SimpleSVMTrainer
from src.config import (
    RULES_PATH, SVM_MODEL_DIR, CNN_MODEL_DIR,
    STOPWORDS_PATH, DATA_DIR, TRAIN_CLEANED_FILE, TEST_CLEANED_FILE
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

class FakeNewsPredictor:
    def __init__(self):
        """Khởi tạo bộ dự đoán tin giả."""
        logger.info("--- Đang khởi tạo bộ dự đoán ---")
        self.system = None
        self._load_system()
    
    def _load_system(self):
        """Tải hệ thống dự đoán."""
        try:
            self.system = HybridPredictionSystem(
                svm_model_dir=SVM_MODEL_DIR,
                cnn_model_dir=CNN_MODEL_DIR,
                rules_path=RULES_PATH
            )
            logger.info("✅ Đã tải hệ thống thành công")
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải hệ thống: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Dự đoán một văn bản."""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Văn bản đầu vào không hợp lệ")
            
            result = self.system.predict(text)
            return result
        except Exception as e:
            logger.error(f"❌ Lỗi khi dự đoán: {str(e)}")
            raise

def main():
    """Hàm chính để chạy hệ thống."""
    try:
        # Khởi tạo bộ dự đoán
        predictor = FakeNewsPredictor()
        
        # Ví dụ 1: Tin thật dễ
        text1 = """
        Theo thông tin từ Bộ Y tế, tính đến 18h ngày 14/6, cả nước ghi nhận thêm 1.234 ca mắc COVID-19 mới, 
        trong đó có 1.200 ca lây nhiễm trong nước. Tổng số ca mắc từ đầu dịch đến nay là 10.000.000 ca.
        """
        result1 = predictor.predict(text1)
        print("\n=== Ví dụ 1: Tin thật dễ ===")
        print(f"Kết quả: {'Tin Giả' if result1['prediction'] == 1 else 'Tin Thật'}")
        print(f"Độ tin cậy: {result1['confidence']*100:.1f}%")
        print(f"Phương pháp: {result1['method']}")
        
        # Ví dụ 2: Tin giả dễ
        text2 = """
        CẢNH BÁO: Nước uống có ga gây ung thư! Theo nghiên cứu mới nhất từ một nhóm bác sĩ không rõ danh tính, 
        uống nước có ga sẽ làm tăng 500% nguy cơ mắc ung thư. Hãy chia sẻ thông tin này để cứu người!
        """
        result2 = predictor.predict(text2)
        print("\n=== Ví dụ 2: Tin giả dễ ===")
        print(f"Kết quả: {'Tin Giả' if result2['prediction'] == 1 else 'Tin Thật'}")
        print(f"Độ tin cậy: {result2['confidence']*100:.1f}%")
        print(f"Phương pháp: {result2['method']}")
        
        # Ví dụ 3: Tin khó
        text3 = """
        Một nghiên cứu mới đây cho thấy việc sử dụng điện thoại thông minh có thể ảnh hưởng đến chất lượng giấc ngủ. 
        Các nhà nghiên cứu đã theo dõi 100 người tham gia trong 3 tháng và phát hiện ra rằng những người sử dụng 
        điện thoại trước khi ngủ có xu hướng khó ngủ hơn và chất lượng giấc ngủ kém hơn.
        """
        result3 = predictor.predict(text3)
        print("\n=== Ví dụ 3: Tin khó ===")
        print(f"Kết quả: {'Tin Giả' if result3['prediction'] == 1 else 'Tin Thật'}")
        print(f"Độ tin cậy: {result3['confidence']*100:.1f}%")
        print(f"Phương pháp: {result3['method']}")
        
    except Exception as e:
        logger.error(f"❌ Lỗi không mong muốn: {str(e)}")
        raise

if __name__ == "__main__":
    main()