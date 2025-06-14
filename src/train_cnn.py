import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import pickle
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.config import (
    DATA_DIR, CNN_MODEL_DIR, STOPWORDS_PATH,
    DATA_HARD_DIR, TRAIN_HARD_FILE, TEST_HARD_FILE,
    CNN_CONFIG
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

class CNNTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        
        # Tạo thư mục model nếu chưa tồn tại
        os.makedirs(CNN_MODEL_DIR, exist_ok=True)
    
    def prepare_data(self, texts):
        """Chuẩn bị dữ liệu cho CNN."""
        # Chuyển đổi văn bản thành chuỗi số
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Padding chuỗi
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.config['max_len'],
            padding='post',
            truncating='post'
        )
        
        return padded_sequences
    
    def calculate_class_weights(self, y):
        """Tính toán class weights để xử lý mất cân bằng dữ liệu."""
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        # Tính toán tỷ lệ mất cân bằng
        imbalance_ratio = class_counts[0] / class_counts[1]
        
        # Áp dụng trọng số cân bằng cho cả hai lớp
        # Không tăng trọng số cho lớp thiểu số để tránh báo động giả
        class_weights = {
            0: 1.0,  # Lớp đa số (tin thật)
            1: 1.0   # Lớp thiểu số (tin giả) - giữ nguyên để tăng Precision
        }
        
        logger.info(f"Tỷ lệ mất cân bằng: {imbalance_ratio:.2f}")
        logger.info(f"Class weights: {class_weights}")
        logger.info("Sử dụng trọng số cân bằng để tối ưu Precision")
        
        return class_weights
    
    def build_model(self, vocab_size):
        """Xây dựng mô hình CNN với các kỹ thuật chống overfitting và tăng độ nhạy với tin giả."""
        model = Sequential([
            # Lớp Embedding với dropout
            Embedding(
                input_dim=vocab_size,
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_len']
            ),
            Dropout(self.config['dropout_rates']['embedding']),
            
            # Lớp Convolution với regularization
            Conv1D(
                filters=self.config['filters'],
                kernel_size=self.config['kernel_size'],
                activation='relu',
                kernel_regularizer=l2(self.config['l2_reg'])
            ),
            BatchNormalization(),
            Dropout(self.config['dropout_rates']['conv']),
            
            # Lớp Pooling
            GlobalMaxPooling1D(),
            
            # Lớp ẩn với regularization
            Dense(
                self.config['hidden_dims'],
                activation='relu',
                kernel_regularizer=l2(self.config['l2_reg'])
            ),
            BatchNormalization(),
            Dropout(self.config['dropout_rates']['dense']),
            
            # Lớp đầu ra với bias_initializer để tăng độ nhạy với tin giả
            Dense(1, activation='sigmoid', 
                  bias_initializer=tf.keras.initializers.Constant(0.1))
        ])
        
        # Biên dịch mô hình với learning rate tùy chỉnh
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Huấn luyện mô hình CNN với các kỹ thuật tăng độ chính xác."""
        logger.info("\n--- Bắt đầu Huấn luyện Mô hình CNN ---")
        
        # Khởi tạo và huấn luyện tokenizer
        logger.info("Khởi tạo và huấn luyện tokenizer...")
        self.tokenizer = Tokenizer(num_words=self.config['max_words'])
        self.tokenizer.fit_on_texts(X_train)
        logger.info(f"Kích thước từ điển: {len(self.tokenizer.word_index)}")
        
        # Tách validation set nếu chưa có
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config['validation_split'],
                random_state=42,
                stratify=y_train
            )
        
        # Chuẩn bị dữ liệu
        logger.info("Chuẩn bị dữ liệu...")
        X_train_seq = self.prepare_data(X_train)
        X_val_seq = self.prepare_data(X_val)
        
        # Tính toán class weights
        class_weights = self.calculate_class_weights(y_train)
        
        # Xây dựng mô hình
        logger.info("Xây dựng mô hình...")
        vocab_size = min(self.config['max_words'], len(self.tokenizer.word_index) + 1)
        self.model = self.build_model(vocab_size)
        
        # Callbacks với focus vào Precision
        callbacks = [
            EarlyStopping(
                monitor='val_precision',
                mode='max',
                patience=self.config['early_stopping']['patience'],
                min_delta=self.config['early_stopping']['min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(CNN_MODEL_DIR, 'best_model.h5'),
                monitor='val_precision',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_precision',
                mode='max',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Huấn luyện với class weights
        logger.info("Bắt đầu huấn luyện...")
        history = self.model.fit(
            X_train_seq,
            y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val_seq, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("✅ Huấn luyện hoàn tất.")
        return history
    
    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình trên tập test."""
        if self.model is None:
            logger.error("Lỗi: Mô hình chưa được huấn luyện.")
            return
        
        logger.info("\n--- Đánh giá Mô hình CNN trên tập TEST ---")
        
        # Chuẩn bị dữ liệu test
        X_test_seq = self.prepare_data(X_test)
        
        # Dự đoán
        y_pred_proba = self.model.predict(X_test_seq)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # In báo cáo phân loại
        logger.info("\nBáo cáo Phân loại (Classification Report):")
        logger.info(classification_report(y_test, y_pred, target_names=['Tin Thật (0)', 'Tin Giả (1)']))
        
        # Vẽ biểu đồ
        self._plot_evaluation_results(y_test, y_pred, y_pred_proba)
    
    def _plot_evaluation_results(self, y_true, y_pred, y_pred_proba):
        """Vẽ các biểu đồ đánh giá."""
        plt.figure(figsize=(20, 15))
        
        # Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # ROC Curve
        plt.subplot(2, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('Đường cong ROC')
        plt.legend(loc="lower right")
        
        # Precision-Recall Curve
        plt.subplot(2, 2, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
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
        
        plt.suptitle('Biểu đồ Đánh giá Hiệu suất Mô hình CNN', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def save_model(self):
        """Lưu mô hình và tokenizer."""
        if self.model is None:
            logger.error("Lỗi: Mô hình chưa được huấn luyện.")
            return
        
        # Lưu mô hình
        model_path = os.path.join(CNN_MODEL_DIR, 'cnn_model.h5')
        self.model.save(model_path)
        
        # Lưu tokenizer
        tokenizer_path = os.path.join(CNN_MODEL_DIR, 'tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        logger.info(f"\n✅ Đã lưu mô hình CNN vào: {model_path}")
        logger.info(f"✅ Đã lưu tokenizer vào: {tokenizer_path}")

if __name__ == "__main__":
    logger.info("\n\n" + "="*20 + " BẮT ĐẦU THÍ NGHIỆM CNN " + "="*20)
    
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

    # 2. Chuẩn bị dữ liệu cho CNN
    logger.info("\n2. Chuẩn bị dữ liệu cho CNN...")
    X_train = df_train_processed['cleaned_message']
    y_train = df_train_processed['label']
    X_test = df_test_processed['cleaned_message']
    y_test = df_test_processed['label']
    
    # 3. Huấn luyện, đánh giá, lưu mô hình
    logger.info("\n3. Bắt đầu huấn luyện mô hình...")
    cnn_trainer = CNNTrainer(CNN_CONFIG)
    history = cnn_trainer.train(X_train, y_train)
    cnn_trainer.evaluate(X_test, y_test)
    cnn_trainer.save_model()