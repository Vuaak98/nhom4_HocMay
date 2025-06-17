import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, BatchNormalization, MaxPooling1D, Concatenate, Add, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
import tensorflow as tfa
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from rule_based import RuleBasedFilterFromFile
import pickle
import joblib

# Thêm thư mục gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from config import (
    DATA_DIR, CNN_MODEL_DIR,
    DATA_HARD_DIR, TRAIN_HARD_FILE, TEST_HARD_FILE,
    CNN_CONFIG, RULES_PATH,
    CNN_MODEL_PATH, CNN_TOKENIZER_PATH,
    CNN_CONFUSION_MATRIX, CNN_ROC_CURVE, CNN_EVALUATION_PLOT
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

# Đường dẫn file
TRAIN_CLEANED_FILE = os.path.join(DATA_DIR, 'train_cleaned.csv')
TEST_CLEANED_FILE = os.path.join(DATA_DIR, 'test_cleaned.csv')
TOKENIZER_PATH = os.path.join(CNN_MODEL_DIR, 'tokenizer.pkl')
MODEL_INFO_PATH = os.path.join(CNN_MODEL_DIR, 'model_info.json')
EVALUATION_PLOT_PATH = os.path.join(CNN_MODEL_DIR, 'evaluation_plots.png')

# Cấu hình mô hình CNN
CNN_CONFIG = {
    'max_words': 15000,  # Tăng từ vựng
    'max_len': 250,      # Tăng độ dài tối đa
    'embedding_dim': 300,
    'filters': [128, 256, 512, 1024],  # Thêm một lớp CNN
    'kernel_size': [2, 3, 4, 5],       # Thêm kernel size nhỏ hơn
    'hidden_dims': 512,  # Tăng kích thước layer ẩn
    'dropout_rates': [0.4, 0.5, 0.6],  # Tăng dropout
    'l2_reg': 0.02,     # Tăng regularization
    'learning_rate': 0.0005,  # Giảm learning rate
    'batch_size': 16,   # Giảm batch size
    'epochs': 100,      # Tăng số epochs
    'validation_split': 0.2,
    'early_stopping': {
        'patience': 15,  # Tăng patience
        'min_delta': 0.0001
    },
    'focal_loss': {
        'gamma': 3.0,    # Tăng gamma để tập trung hơn vào các mẫu khó
        'alpha': 0.85    # Tăng alpha để ưu tiên lớp thiểu số (tin giả)
    }
}

class CNNTrainer:
    """Lớp huấn luyện mô hình CNN."""
    
    def __init__(self):
        """Khởi tạo trainer."""
        self.model = None
        self.tokenizer = Tokenizer(
            num_words=CNN_CONFIG['max_words'],
            oov_token='<OOV>'
        )
        self.scaler = StandardScaler()
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(CNN_MODEL_DIR, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> tuple:
        """
        Chuẩn bị đặc trưng từ DataFrame.
        
        Args:
            df: DataFrame chứa dữ liệu
            is_training: True nếu đang xử lý dữ liệu huấn luyện
            
        Returns:
            tuple: (text_features, numeric_features)
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
        logger.info("   - post_message")
        logger.info(f"Tổng số đặc trưng số: {len(numeric_features)}")
        
        # Xử lý đặc trưng văn bản
        if is_training:
            self.tokenizer.fit_on_texts(df['post_message'].fillna(''))
        
        sequences = self.tokenizer.texts_to_sequences(df['post_message'].fillna(''))
        text_features = pad_sequences(sequences, maxlen=CNN_CONFIG['max_len'])

        # Xử lý đặc trưng số
        numeric_features = df[numeric_features].values
        if is_training:
            numeric_features = self.scaler.fit_transform(numeric_features)
        else:
            numeric_features = self.scaler.transform(numeric_features)
        
        return text_features, numeric_features
    
    def build_model(self, vocab_size: int, num_numeric_features: int):
        """
        Xây dựng mô hình CNN.
        
        Args:
            vocab_size: Kích thước từ điển
            num_numeric_features: Số lượng đặc trưng số
        """
        # Input cho văn bản
        text_input = Input(shape=(CNN_CONFIG['max_len'],), name='text_input')
        embedding = Embedding(vocab_size, CNN_CONFIG['embedding_dim'])(text_input)
        
        # CNN cho văn bản
        conv1 = Conv1D(128, 5, activation='relu')(embedding)
        pool1 = MaxPooling1D(5)(conv1)
        conv2 = Conv1D(128, 5, activation='relu')(pool1)
        pool2 = MaxPooling1D(5)(conv2)
        conv3 = Conv1D(128, 5, activation='relu')(pool2)
        pool3 = MaxPooling1D(5)(conv3)
        text_features = Flatten()(pool3)
        
        # Input cho đặc trưng số
        numeric_input = Input(shape=(num_numeric_features,), name='numeric_input')
        numeric_dense = Dense(64, activation='relu')(numeric_input)
        numeric_dense = BatchNormalization()(numeric_dense)
        numeric_dense = Dropout(0.3)(numeric_dense)
        
        # Kết hợp đặc trưng
        combined = Concatenate()([text_features, numeric_dense])
        dense = Dense(128, activation='relu')(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        output = Dense(1, activation='sigmoid')(dense)
        
        # Tạo mô hình
        self.model = Model(inputs=[text_input, numeric_input], outputs=output)
        
        # Compile mô hình
        self.model.compile(
            optimizer=Adam(learning_rate=CNN_CONFIG['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # In thông tin mô hình
        self.model.summary()

    def train(self):
        """Huấn luyện mô hình CNN."""
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
            X_train_text, X_train_numeric = self.prepare_features(df_train, is_training=True)
            X_test_text, X_test_numeric = self.prepare_features(df_test, is_training=False)

            y_train = df_train['label'].values
            y_test = df_test['label'].values

            # 3. Xây dựng mô hình
            vocab_size = len(self.tokenizer.word_index) + 1
            self.build_model(vocab_size, X_train_numeric.shape[1])
            
            # 4. Callbacks
            callbacks = [
                ModelCheckpoint(
                    os.path.join(CNN_MODEL_DIR, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # 5. Huấn luyện
            logger.info("\n=== Huấn luyện CNN ===")
            history = self.model.fit(
                [X_train_text, X_train_numeric],
                y_train,
                validation_data=([X_test_text, X_test_numeric], y_test),
                epochs=CNN_CONFIG['epochs'],
                batch_size=CNN_CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # 6. Đánh giá
            logger.info("\n=== Đánh giá mô hình ===")
            y_pred = (self.model.predict([X_test_text, X_test_numeric]) > 0.5).astype(int)
            logger.info("\nBáo cáo Phân loại (Classification Report):")
            logger.info(classification_report(y_test, y_pred, target_names=['Tin Thật (0)', 'Tin Giả (1)']))
            
            # 7. Lưu mô hình và tokenizer
            self.model.save(os.path.join(CNN_MODEL_DIR, 'final_model.h5'))
            with open(os.path.join(CNN_MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
                pickle.dump(self.tokenizer, f)
            joblib.dump(self.scaler, os.path.join(CNN_MODEL_DIR, 'scaler.joblib'))
            
            logger.info(f"✅ Đã lưu mô hình CNN tại {CNN_MODEL_DIR}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi huấn luyện mô hình CNN: {str(e)}")
            raise

def main():
    """Hàm chính để chạy huấn luyện CNN."""
    try:
        trainer = CNNTrainer()
        trainer.train()
    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
        raise

if __name__ == "__main__":
    main()