import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import resample
import seaborn as sns
import shutil
import logging
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR, SVM_MODEL_DIR, STOPWORDS_PATH,
    DATA_HARD_DIR, TRAIN_HARD_FILE, TEST_HARD_FILE
)
from src.data_processing import create_full_features

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

class SimpleSVMTrainer:
    """
    Lớp này xây dựng và huấn luyện một mô hình SVM đa đầu vào với tối ưu hóa siêu tham số
    và xử lý dữ liệu mất cân bằng.
    """
    def __init__(self, numerical_cols, text_col, model_dir, n_runs=5, n_bootstrap=100):
        self.numerical_cols = numerical_cols
        self.text_col = text_col
        self.model_dir = model_dir
        self.n_runs = n_runs  # Số lần chạy lặp lại
        self.n_bootstrap = n_bootstrap  # Số lần bootstrap
        
        # Xử lý thư mục model
        if os.path.exists(model_dir):
            if os.path.isfile(model_dir):
                os.remove(model_dir)
            else:
                shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

    def _select_text_column(self, x):
        return x[self.text_col]

    def _select_numeric_columns(self, x):
        return x[self.numerical_cols]

    def build_and_train(self, X_train, y_train):
        # Kiểm tra dữ liệu đầu vào
        if X_train.empty or len(X_train) == 0:
            raise ValueError("Dữ liệu huấn luyện trống!")
            
        logger.info(f"  - Kích thước dữ liệu huấn luyện: {X_train.shape}")
        logger.info(f"  - Số lượng mẫu: {len(X_train)}")
        
        # Kiểm tra văn bản
        text_col = self.text_col
        if text_col in X_train.columns:
            empty_texts = X_train[text_col].str.len() == 0
            if empty_texts.any():
                logger.warning(f"  ⚠️ Có {empty_texts.sum()} văn bản trống")
                # Sử dụng văn bản gốc nếu văn bản đã làm sạch trống
                X_train.loc[empty_texts, text_col] = X_train.loc[empty_texts, 'post_message'].str.lower()
        
        numeric_pipeline = Pipeline([
            ('selector', FunctionTransformer(self._select_numeric_columns)),
            ('scaler', StandardScaler())
        ])
        
        text_pipeline = Pipeline([
            ('selector', FunctionTransformer(self._select_text_column)),
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                max_df=0.95,
                stop_words=None
            ))
        ])
        
        feature_union = FeatureUnion([
            ('numeric_features', numeric_pipeline),
            ('text_features', text_pipeline)
        ], n_jobs=-1)

        # Tạo pipeline hoàn chỉnh với các tham số tối ưu
        self.pipeline_ = Pipeline([
            ('features', feature_union),
            ('clf', SVC(
                kernel='linear',
                C=1,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced',
                cache_size=2000
            ))
        ])
        
        logger.info("--- Bắt đầu Huấn luyện Mô hình SVM với Tham số Tối ưu ---")
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.pipeline_,
            X_train,
            y_train,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Huấn luyện trên toàn bộ dữ liệu
        self.pipeline_.fit(X_train, y_train)
        logger.info("✅ Huấn luyện hoàn tất.")

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình với nhiều lần chạy và bootstrap."""
        if self.pipeline_ is None:
            logger.error("Lỗi: Mô hình chưa được huấn luyện.")
            return
        
        logger.info("\n--- Đánh giá Mô hình SVM trên tập TEST ---")
        
        # Chạy nhiều lần
        metrics = {
            'accuracy': [],
            'precision_0': [], 'precision_1': [],
            'recall_0': [], 'recall_1': [],
            'f1_0': [], 'f1_1': [],
            'roc_auc': []
        }
        
        for i in tqdm(range(self.n_runs), desc="Chạy lặp lại"):
            # Bootstrap
            bootstrap_metrics = {
                'accuracy': [],
                'precision_0': [], 'precision_1': [],
                'recall_0': [], 'recall_1': [],
                'f1_0': [], 'f1_1': [],
                'roc_auc': []
            }
            
            for _ in range(self.n_bootstrap):
                # Tạo bootstrap sample
                indices = resample(range(len(X_test)), random_state=i)
                X_boot = X_test.iloc[indices]
                y_boot = y_test.iloc[indices]
                
                # Dự đoán
                y_pred = self.pipeline_.predict(X_boot)
                y_pred_proba = self.pipeline_.predict_proba(X_boot)[:, 1]
                
                # Tính metrics
                report = classification_report(y_boot, y_pred, output_dict=True)
                bootstrap_metrics['accuracy'].append(report['accuracy'])
                bootstrap_metrics['precision_0'].append(report['0']['precision'])
                bootstrap_metrics['precision_1'].append(report['1']['precision'])
                bootstrap_metrics['recall_0'].append(report['0']['recall'])
                bootstrap_metrics['recall_1'].append(report['1']['recall'])
                bootstrap_metrics['f1_0'].append(report['0']['f1-score'])
                bootstrap_metrics['f1_1'].append(report['1']['f1-score'])
                bootstrap_metrics['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba))
            
            # Lưu kết quả trung bình của lần chạy này
            for metric in metrics:
                metrics[metric].append(np.mean(bootstrap_metrics[metric]))
        
        # Tính kết quả cuối cùng
        final_metrics = {}
        for metric in metrics:
            mean = np.mean(metrics[metric])
            std = np.std(metrics[metric])
            final_metrics[metric] = (mean, std)
        
        # In kết quả
        logger.info("\nKết quả đánh giá (Mean ± Std):")
        logger.info(f"Accuracy: {final_metrics['accuracy'][0]:.4f} ± {final_metrics['accuracy'][1]:.4f}")
        logger.info("\nTin Thật (0):")
        logger.info(f"Precision: {final_metrics['precision_0'][0]:.4f} ± {final_metrics['precision_0'][1]:.4f}")
        logger.info(f"Recall: {final_metrics['recall_0'][0]:.4f} ± {final_metrics['recall_0'][1]:.4f}")
        logger.info(f"F1-score: {final_metrics['f1_0'][0]:.4f} ± {final_metrics['f1_0'][1]:.4f}")
        logger.info("\nTin Giả (1):")
        logger.info(f"Precision: {final_metrics['precision_1'][0]:.4f} ± {final_metrics['precision_1'][1]:.4f}")
        logger.info(f"Recall: {final_metrics['recall_1'][0]:.4f} ± {final_metrics['recall_1'][1]:.4f}")
        logger.info(f"F1-score: {final_metrics['f1_1'][0]:.4f} ± {final_metrics['f1_1'][1]:.4f}")
        logger.info(f"\nROC-AUC: {final_metrics['roc_auc'][0]:.4f} ± {final_metrics['roc_auc'][1]:.4f}")
        
        # Vẽ biểu đồ
        self._plot_evaluation_results(X_test, y_test, final_metrics)

    def _plot_evaluation_results(self, X_test, y_test, final_metrics):
        """Vẽ các biểu đồ đánh giá."""
        # Dự đoán trên toàn bộ tập test
        y_pred = self.pipeline_.predict(X_test)
        y_pred_proba = self.pipeline_.predict_proba(X_test)[:, 1]
        
        plt.figure(figsize=(20, 15))
        
        # Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # ROC Curve
        plt.subplot(2, 2, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('Đường cong ROC')
        plt.legend(loc="lower right")
        
        # Precision-Recall Curve
        plt.subplot(2, 2, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.title('Đường cong Precision-Recall')
        plt.legend(loc="lower left")
        
        # Distribution of Predictions
        plt.subplot(2, 2, 4)
        plt.hist(y_pred_proba, bins=50, alpha=0.5, label='Phân phối điểm dự đoán')
        plt.title('Phân phối điểm dự đoán')
        plt.xlabel('Điểm dự đoán')
        plt.ylabel('Số lượng')
        
        plt.suptitle('Biểu đồ Đánh giá Hiệu suất Mô hình SVM', fontsize=16)
        plt.tight_layout()
        plt.show()

    def save_model(self, filename='svm_simple_pipeline.pkl'):
        if self.pipeline_ is None:
            logger.error("Lỗi: Mô hình chưa được huấn luyện.")
            return
        model_output_path = os.path.join(self.model_dir, filename)
        try:
            with open(model_output_path, 'wb') as f:
                pickle.dump(self.pipeline_, f)
            logger.info(f"\n✅ Đã lưu pipeline SVM đơn giản vào: {model_output_path}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu file: {e}")

if __name__ == "__main__":
    logger.info("\n\n" + "="*20 + " BẮT ĐẦU THÍ NGHIỆM SVM " + "="*20)
    
    # 1. Tải và xử lý dữ liệu
    logger.info("\n1. Đang tải và xử lý dữ liệu...")
    
    df_train_raw = pd.read_csv(TRAIN_HARD_FILE, encoding='utf-8')
    df_test_raw = pd.read_csv(TEST_HARD_FILE, encoding='utf-8')
    
    logger.info(f"   - Kích thước tập train (tin khó): {df_train_raw.shape}")
    logger.info(f"   - Kích thước tập test (tin khó): {df_test_raw.shape}")
    logger.info(f"   - Phân bố nhãn trong tập train:")
    logger.info(df_train_raw['label'].value_counts(normalize=True).to_string())
    logger.info(f"   - Phân bố nhãn trong tập test:")
    logger.info(df_test_raw['label'].value_counts(normalize=True).to_string())
    
    logger.info("\n   - Đang xử lý dữ liệu train...")
    df_train_processed = create_full_features(df_train_raw, stopwords_path=STOPWORDS_PATH, df_name="Train Hard")
    logger.info("\n   - Đang xử lý dữ liệu test...")
    df_test_processed = create_full_features(df_test_raw, stopwords_path=STOPWORDS_PATH, df_name="Test Hard")

    # 2. Chuẩn bị dữ liệu cho SVM
    logger.info("\n2. Chuẩn bị dữ liệu cho SVM...")
    numerical_cols = df_train_processed.select_dtypes(include=np.number).columns.drop(['label', 'id'], errors='ignore').tolist()
    text_col = 'cleaned_message'
    
    logger.info(f"   - Số lượng đặc trưng số: {len(numerical_cols)}")
    logger.info(f"   - Các đặc trưng số: {numerical_cols}")
    
    X_train_svm = df_train_processed[numerical_cols + [text_col]]
    y_train_svm = df_train_processed['label']
    X_test_svm = df_test_processed[numerical_cols + [text_col]]
    y_test_svm = df_test_processed['label']
    
    logger.info(f"   - Kích thước tập train sau xử lý: {X_train_svm.shape}")
    logger.info(f"   - Kích thước tập test sau xử lý: {X_test_svm.shape}")
    logger.info(f"   - Phân bố nhãn trong tập train sau xử lý:")
    logger.info(pd.Series(y_train_svm).value_counts(normalize=True).to_string())
    logger.info(f"   - Phân bố nhãn trong tập test sau xử lý:")
    logger.info(pd.Series(y_test_svm).value_counts(normalize=True).to_string())

    # 3. Huấn luyện, đánh giá, lưu mô hình
    logger.info("\n3. Bắt đầu huấn luyện mô hình...")
    svm_trainer = SimpleSVMTrainer(numerical_cols=numerical_cols, text_col=text_col, model_dir=SVM_MODEL_DIR)
    svm_trainer.build_and_train(X_train_svm, y_train_svm)
    svm_trainer.evaluate(X_test_svm, y_test_svm)
    svm_trainer.save_model()