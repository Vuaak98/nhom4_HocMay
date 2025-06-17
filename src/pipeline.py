import os
import sys
import logging
import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Thêm thư mục gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import (
    DATA_DIR, DATA_PROCESSED_DIR, CNN_MODEL_DIR, SVM_MODEL_DIR,
    TRAIN_CLEANED_FILE, TEST_CLEANED_FILE, RULES_PATH
)
from data_processing import DataProcessor
from train_rules import RuleBasedClassifier
from train_cnn import CNNTrainer
from train_svm import SVMTrainer

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def run_pipeline():
    """Chạy toàn bộ quy trình xử lý và huấn luyện."""
    try:
        # 1. Tiền xử lý dữ liệu
        logger.info("\n=== Bước 1: Tiền xử lý dữ liệu ===")
        processor = DataProcessor()
        processor.process_data()
        
        # 2. Phân loại bằng luật
        logger.info("\n=== Bước 2: Phân loại bằng luật ===")
        rule_classifier = RuleBasedClassifier(rules_path=RULES_PATH)
        df_train = pd.read_csv(TRAIN_CLEANED_FILE, encoding='utf-8')
        df_test = pd.read_csv(TEST_CLEANED_FILE, encoding='utf-8')
        
        # Phân loại tập train
        train_predictions, train_results = rule_classifier.classify_batch(df_train)
        logger.info("\nKết quả phân loại tập train:")
        logger.info(f"Tổng số mẫu: {len(df_train)}")
        logger.info(f"Tin Thật: {len(df_train[df_train['label'] == 0])}")
        logger.info(f"Tin Giả: {len(df_train[df_train['label'] == 1])}")
        
        # Phân loại tập test
        test_predictions, test_results = rule_classifier.classify_batch(df_test)
        logger.info("\nKết quả phân loại tập test:")
        logger.info(f"Tổng số mẫu: {len(df_test)}")
        logger.info(f"Tin Thật: {len(df_test[df_test['label'] == 0])}")
        logger.info(f"Tin Giả: {len(df_test[df_test['label'] == 1])}")
        
        # 3. Huấn luyện mô hình CNN với đặc trưng từ luật
        logger.info("\n=== Bước 3: Huấn luyện mô hình CNN với đặc trưng từ luật ===")
        cnn_trainer = CNNTrainer()
        cnn_trainer.train()
        
        # 4. Huấn luyện mô hình SVM với đặc trưng từ CNN
        logger.info("\n=== Bước 4: Huấn luyện mô hình SVM với đặc trưng từ CNN ===")
        svm_trainer = SVMTrainer()
        svm_trainer.train()
        
        # 5. Đánh giá kết hợp
        logger.info("\n=== Bước 5: Đánh giá kết hợp ===")
        
        # Đọc dữ liệu test
        df_test = pd.read_csv(TEST_CLEANED_FILE, encoding='utf-8')
        df_test = df_test[df_test['label'].isin([0, 1])].copy()
        
        # Dự đoán từ CNN
        cnn_pred = cnn_trainer.model.predict(cnn_trainer.prepare_data(df_test['post_message'].tolist()))
        cnn_pred = (cnn_pred > 0.5).astype(int)
        
        # Dự đoán từ SVM
        svm_features = svm_trainer.extract_cnn_features(df_test['post_message'].tolist())
        svm_features = svm_trainer.scaler.transform(svm_features)
        svm_pred = svm_trainer.svm.predict(svm_features)
        
        # Kết hợp dự đoán (majority voting)
        combined_pred = (cnn_pred.flatten() + svm_pred) > 1
        
        # In báo cáo
        logger.info("\nBáo cáo Phân loại (Combined Model):")
        logger.info(classification_report(
            df_test['label'].values,
            combined_pred,
            target_names=['Tin Thật (0)', 'Tin Giả (1)']
        ))
        
        logger.info("✅ Pipeline hoàn thành!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()