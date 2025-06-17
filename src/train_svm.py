import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import logging
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, SVM_MODEL_DIR,
    DATA_HARD_DIR, TRAIN_HARD_FILE, TEST_HARD_FILE
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

class SVMTrainer:
    """Lớp huấn luyện mô hình SVM."""
    
    def __init__(self):
        """Khởi tạo trainer."""
        self.svm = SVC(
            probability=True,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(SVM_MODEL_DIR, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> tuple:
        """
        Chuẩn bị đặc trưng từ DataFrame.
        
        Args:
            df: DataFrame chứa dữ liệu
            is_training: True nếu đang xử lý dữ liệu huấn luyện
            
        Returns:
            tuple: (numeric_features, text_features)
        """
        # Lấy các cột số hữu ích
        numeric_features = [
            'num_like_post',
            'num_comment_post',
            'num_share_post'
        ]
        
        # Log các đặc trưng đang sử dụng
        logger.info("\n=== Các đặc trưng đang sử dụng ===")
        logger.info("1. Đặc trưng số:")
        for feat in numeric_features:
            logger.info(f"   - {feat}")
        logger.info("2. Đặc trưng văn bản:")
        logger.info("   - post_message (TF-IDF)")
        logger.info(f"Tổng số đặc trưng số: {len(numeric_features)}")
        
        # Xử lý đặc trưng số
        X_numeric = df[numeric_features].values
        
        # Xử lý đặc trưng văn bản
        if is_training:
            X_text = self.tfidf.fit_transform(df['post_message'].fillna(''))
        else:
            X_text = self.tfidf.transform(df['post_message'].fillna(''))
        
        return X_numeric, X_text
    
    def find_best_params(self, X_train, y_train):
        """
        Tìm siêu tham số tối ưu cho SVM.
        
        Args:
            X_train: Dữ liệu huấn luyện
            y_train: Nhãn huấn luyện
        """
        # Định nghĩa grid search
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear', 'poly'],
            'degree': [2, 3, 4]  # Chỉ dùng cho kernel poly
        }
        
        # Tạo GridSearchCV
        grid_search = GridSearchCV(
            estimator=SVC(probability=True, class_weight='balanced'),
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        # Tìm tham số tối ưu
        logger.info("\n=== Tìm siêu tham số tối ưu ===")
        grid_search.fit(X_train, y_train)
        
        # Log kết quả
        logger.info("\nKết quả tìm kiếm siêu tham số:")
        logger.info(f"Tham số tốt nhất: {grid_search.best_params_}")
        logger.info(f"F1-score tốt nhất: {grid_search.best_score_:.3f}")
        
        # Cập nhật mô hình với tham số tốt nhất
        self.svm = grid_search.best_estimator_
    
    def train(self):
        """Huấn luyện mô hình SVM."""
        try:
            # 1. Đọc dữ liệu
            df_train = pd.read_csv(TRAIN_HARD_FILE, encoding='utf-8')
            df_test = pd.read_csv(TEST_HARD_FILE, encoding='utf-8')
            
            # Lọc bỏ các mẫu có nhãn -1
            df_train = df_train[df_train['label'].isin([0, 1])].copy()
            df_test = df_test[df_test['label'].isin([0, 1])].copy()
            
            logger.info("\n=== Phân phối dữ liệu ===")
            logger.info(f"Tập train: {len(df_train)} mẫu")
            logger.info(f"- Tin thật: {len(df_train[df_train['label'] == 0])}")
            logger.info(f"- Tin giả: {len(df_train[df_train['label'] == 1])}")
            
            # 2. Chuẩn bị đặc trưng
            logger.info("\n=== Chuẩn bị đặc trưng ===")
            X_train_numeric, X_train_text = self.prepare_features(df_train, is_training=True)
            X_test_numeric, X_test_text = self.prepare_features(df_test, is_training=False)
            
            # Chuẩn hóa đặc trưng số
            X_train_numeric = self.scaler.fit_transform(X_train_numeric)
            X_test_numeric = self.scaler.transform(X_test_numeric)
            
            # Kết hợp đặc trưng số và văn bản
            X_train = np.hstack([X_train_numeric, X_train_text.toarray()])
            X_test = np.hstack([X_test_numeric, X_test_text.toarray()])
            
            y_train = df_train['label'].values
            y_test = df_test['label'].values
            
            # 3. Tìm siêu tham số tối ưu
            self.find_best_params(X_train, y_train)
            
            # 4. Huấn luyện SVM với tham số tốt nhất
            logger.info("\n=== Huấn luyện SVM ===")
            self.svm.fit(X_train, y_train)
            
            # 5. Đánh giá
            y_pred = self.svm.predict(X_test)
            logger.info("\nBáo cáo Phân loại (Classification Report):")
            logger.info(classification_report(y_test, y_pred, target_names=['Tin Thật (0)', 'Tin Giả (1)']))
            
            # 6. Lưu mô hình
            svm_model_path = os.path.join(SVM_MODEL_DIR, 'svm_model.joblib')
            scaler_path = os.path.join(SVM_MODEL_DIR, 'scaler.joblib')
            tfidf_path = os.path.join(SVM_MODEL_DIR, 'tfidf.joblib')
            
            joblib.dump(self.svm, svm_model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.tfidf, tfidf_path)
            logger.info(f"✅ Đã lưu mô hình SVM tại {svm_model_path}")
            logger.info(f"✅ Đã lưu scaler tại {scaler_path}")
            logger.info(f"✅ Đã lưu TF-IDF tại {tfidf_path}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi huấn luyện mô hình SVM: {str(e)}")
            raise

def main():
    """Hàm chính để chạy huấn luyện SVM."""
    try:
        trainer = SVMTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
        raise

if __name__ == "__main__":
    main()