import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import logging
import numpy as np
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from src.config import (
    RULES_PATH, SVM_MODEL_DIR, CNN_MODEL_DIR,
    STOPWORDS_PATH, DATA_DIR, TRAIN_CLEANED_FILE, TEST_CLEANED_FILE, DATA_RAW_DIR
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

# === Cung cấp một lớp "giả" để pickle có thể tìm thấy khi tải mô hình ===
class SimpleSVMTrainer:
    def _select_text_column(self, x):
        return x['cleaned_message']

    def _select_numeric_columns(self, x):
        numerical_cols = x.select_dtypes(include=np.number).columns.drop(['label', 'id'], errors='ignore').tolist()
        return x[numerical_cols]

# =======================================================

class RuleBasedFilterFromFile:
    def __init__(self, rules_path: str):
        """Khởi tạo bộ lọc dựa trên luật từ file JSON."""
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self._compile_keyword_patterns()
    
    def _load_rules(self) -> Dict[str, List[str]]:
        """Tải luật từ file JSON."""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            logger.info(f"✅ Đã tải {len(rules)} luật từ {self.rules_path}")
            return rules
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải luật: {str(e)}")
            return {}
    
    def _compile_keyword_patterns(self) -> None:
        """Biên dịch các mẫu từ khóa thành regex."""
        self.patterns = {}
        for category, keywords in self.rules.items():
            if isinstance(keywords, list):
                pattern = '|'.join(map(re.escape, keywords))
                self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def _is_journalistic_pattern(self, text: str) -> bool:
        """
        Kiểm tra mẫu báo chí (Final). Yêu cầu nguồn uy tín + hành động chính thức,
        và không có bất kỳ dấu hiệu rủi ro cao nào (thuyết âm mưu/giả khoa học).
        """
        text_lower = text.lower()
        
        # Điều kiện cần: Phải có cả nguồn và hành động chính thức
        has_authoritative_source = self.patterns.get('authoritative_entities', re.compile('')).search(text_lower)
        has_official_action = self.patterns.get('official_actions', re.compile('')).search(text_lower)
        if not (has_authoritative_source and has_official_action):
            return False

        # Điều kiện loại trừ: Tin tức chính thống không chứa thuyết âm mưu/giả khoa học.
        if self.patterns.get('risk_conspiracy_pseudoscience', re.compile('')).search(text_lower):
            return False
            
        return True

    def _is_baseless_rumor_pattern(self, text: str) -> bool:
        """
        Kiểm tra tin đồn vô căn cứ bằng hệ thống điểm rủi ro đa chiều.
        Phiên bản thắt chặt hơn để tăng độ chính xác cho tin giả.
        """
        text_lower = text.lower()
        
        # --- Tính điểm cho từng chiều rủi ro ---
        
        # CHIỀU 1: RỦI RO VỀ NỘI DUNG (Content Risk) - Tăng trọng số
        # Thuyết âm mưu và giả khoa học là dấu hiệu mạnh nhất.
        score_content = 5.0 if self.patterns.get('risk_conspiracy_pseudoscience', re.compile('')).search(text_lower) else 0

        # CHIỀU 2: RỦI RO VỀ CẢM XÚC (Emotional & Hyperbole Risk) - Tăng trọng số
        # Lạm dụng từ ngữ cảm xúc và phóng đại.
        emotional_matches = len(self.patterns.get('risk_emotional_hyperbole', re.compile('')).findall(text_lower))
        score_emotion = min(emotional_matches * 2.5, 5.0)  # Tăng trọng số và giới hạn

        # CHIỀU 3: RỦI RO VỀ NGUỒN TIN (Source Risk) - Tăng trọng số
        # Nguồn tin mơ hồ, không xác thực.
        source_matches = len(self.patterns.get('risk_unverified_sources', re.compile('')).findall(text_lower))
        score_source = min(source_matches * 2.0, 4.0)  # Tăng trọng số và giới hạn

        # CHIỀU 4: RỦI RO VỀ MỤC ĐÍCH (Intent Risk) - Tăng trọng số
        # Có ý đồ tạo sự cấp bách hoặc kêu gọi chia sẻ.
        urgency_matches = len(self.patterns.get('risk_urgency', re.compile('')).findall(text_lower))
        cta_matches = len(self.patterns.get('risk_call_to_action', re.compile('')).findall(text_lower))
        score_intent = min((urgency_matches * 1.0 + cta_matches * 1.0), 4.0)  # Tăng trọng số và giới hạn

        # CHIỀU 5: RỦI RO VỀ MẠNG XÃ HỘI (Social Media Risk) - Tăng trọng số
        # Các dấu hiệu lan truyền trên mạng xã hội
        social_matches = len(self.patterns.get('risk_social_media_hype', re.compile('')).findall(text_lower))
        score_social = min(social_matches * 1.5, 3.0)  # Tăng trọng số và giới hạn

        # CHIỀU 6: RỦI RO VỀ TÍNH ĐỘC QUYỀN (Exclusivity Risk) - Tăng trọng số
        # Các tuyên bố về tính độc quyền của thông tin
        exclusivity_matches = len(self.patterns.get('risk_exclusivity_claims', re.compile('')).findall(text_lower))
        score_exclusivity = min(exclusivity_matches * 1.5, 3.0)  # Tăng trọng số và giới hạn

        # CHIỀU 7: RỦI RO VỀ ĐỊNH DẠNG VĂN BẢN (Text Formatting Risk)
        # Kiểm tra việc lạm dụng chữ in hoa và ký tự đặc biệt
        score_formatting = 0
        
        # 1. Kiểm tra tỷ lệ chữ in hoa
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if uppercase_ratio > 0.4:  # Tăng ngưỡng lên 40%
            score_formatting += 1.5  # Tăng điểm cho tỷ lệ cao
            
        # 2. Kiểm tra các từ khóa cảm xúc mạnh được viết hoa
        uppercase_words = set(word for word in text.split() if word.isupper() and len(word) > 1)
        formatting_keywords = set(word.lower() for word in self.rules.get('risk_text_formatting', []))
        uppercase_matches = sum(1 for word in uppercase_words if word.lower() in formatting_keywords)
        score_formatting += min(uppercase_matches * 0.75, 2.0)  # Tăng điểm cho từ khóa viết hoa
        
        # 3. Kiểm tra ký tự đặc biệt liên tiếp
        special_chars = len(re.findall(r'[!?]{2,}|[!?]{1,}[!?]{1,}|[!?]{1,}\s*[!?]{1,}', text))
        if special_chars >= 3:  # Nếu có 3 hoặc nhiều hơn các ký tự đặc biệt liên tiếp
            score_formatting += 1.5  # Tăng điểm cho ký tự đặc biệt
            
        # 4. Kiểm tra emoji và biểu tượng cảm xúc
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', text))
        if emoji_count >= 3:  # Nếu có 3 hoặc nhiều emoji
            score_formatting += 1.0

        # --- Tổng hợp điểm và quyết định ---
        total_risk_score = (
            score_content + 
            score_emotion + 
            score_source + 
            score_intent + 
            score_social + 
            score_exclusivity +
            score_formatting
        )

        # ĐIỀU KIỆN 1: Ngưỡng điểm tổng cao hơn (5.0 điểm)
        if total_risk_score < 5.0:
            return False

        # ĐIỀU KIỆN 2: Yêu cầu ít nhất 2 chiều rủi ro có điểm > 0
        risk_scores = [
            score_content, score_emotion, score_source,
            score_intent, score_social, score_exclusivity,
            score_formatting
        ]
        if sum(1 for score in risk_scores if score > 0) < 2:
            return False

        # ĐIỀU KIỆN 3: Không có nguồn uy tín
        if self.patterns.get('authoritative_entities', re.compile('')).search(text_lower):
            return False

        # ĐIỀU KIỆN 4: Không có hành động chính thức
        if self.patterns.get('official_actions', re.compile('')).search(text_lower):
            return False

        return True
    
    def calculate_scores(self, text: str) -> Tuple[float, float]:
        """Tính toán điểm tin cậy và điểm giả mạo."""
        # Kiểm tra tin thật dễ
        if self._is_journalistic_pattern(text):
            return -1.0, 0.0  # Tin thật dễ
        
        # Kiểm tra tin giả dễ với logic mới
        if self._is_baseless_rumor_pattern(text):
            return 0.0, 1.0  # Tin giả dễ
        
        return 0.0, 0.0  # Tin khó

class HybridPredictionSystem:
    def __init__(self, svm_model_dir: str, cnn_model_dir: str, rules_path: str):
        """Khởi tạo hệ thống dự đoán kết hợp."""
        self.rule_filter = None
        self.svm_model = None
        self.cnn_model = None
        self.cnn_tokenizer = None
        
        # Tải các mô hình
        self._load_models(svm_model_dir, cnn_model_dir, rules_path)
    
    def _load_models(self, svm_model_dir: str, cnn_model_dir: str, rules_path: str) -> None:
        """Tải tất cả các mô hình cần thiết."""
        try:
            # Tải bộ lọc luật
            self.rule_filter = RuleBasedFilterFromFile(rules_path)
            logger.info("✅ Đã tải bộ lọc luật")
            
            # Tải mô hình SVM (đối tượng Pipeline)
            svm_path = os.path.join(svm_model_dir, 'svm_simple_pipeline.pkl')
            with open(svm_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            logger.info("✅ Đã tải mô hình SVM (Pipeline)")
            
            # Tải mô hình CNN
            cnn_path = os.path.join(cnn_model_dir, 'best_model.h5')
            self.cnn_model = load_model(cnn_path)
            logger.info("✅ Đã tải mô hình CNN")
            
            # Tải tokenizer cho CNN
            tokenizer_path = os.path.join(cnn_model_dir, 'tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                self.cnn_tokenizer = pickle.load(f)
            logger.info("✅ Đã tải tokenizer")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
            raise
    
    def _predict_svm(self, text: str) -> Tuple[int, np.ndarray]:
        """Dự đoán bằng mô hình SVM."""
        # Tạo DataFrame với các cột cần thiết
        features = pd.DataFrame({
            'post_message': [text],
            'timestamp_post': [pd.Timestamp.now()],
            'num_like_post': [0],
            'num_comment_post': [0],
            'num_share_post': [0]
        })
        
        # Xử lý dữ liệu
        features = create_full_features(features, STOPWORDS_PATH, "Prediction")
        
        # Dự đoán
        prediction = self.svm_model.predict(features)[0]
        probability = self.svm_model.predict_proba(features)[0]
        
        return prediction, probability
    
    def _predict_cnn(self, text: str) -> Tuple[int, float]:
        """Dự đoán bằng mô hình CNN."""
        sequence = self.cnn_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=300, padding='post')
        
        prediction = self.cnn_model.predict(padded)[0][0]
        return int(prediction > 0.5), prediction
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Dự đoán kết quả cuối cùng cho một văn bản."""
        logger.info("\n--- Bắt đầu Dự đoán ---")
        
        # Tầng 1: Kiểm tra bằng bộ lọc luật
        trust_score, fake_score = self.rule_filter.calculate_scores(text)
        
        if trust_score == -1.0:  # Tin thật dễ
            logger.info("Kết quả: Tin Thật (Dễ)")
            return {
                'prediction': 0,
                'confidence': 1.0,
                'method': 'rule_based',
                'details': {
                    'trust_score': float(trust_score),
                    'fake_score': float(fake_score)
                }
            }
        
        if fake_score == 1.0:  # Tin giả dễ
            logger.info("Kết quả: Tin Giả (Dễ)")
            return {
                'prediction': 1,
                'confidence': 1.0,
                'method': 'rule_based',
                'details': {
                    'trust_score': float(trust_score),
                    'fake_score': float(fake_score)
                }
            }
        
        # Tầng 2: Xử lý tin khó bằng ML
        logger.info("Đây là tin khó, sử dụng mô hình ML...")
        
        # Dự đoán bằng SVM
        svm_pred, svm_prob = self._predict_svm(text)
        
        # Dự đoán bằng CNN
        cnn_pred, cnn_prob = self._predict_cnn(text)
        
        # Tầng 3: Kết hợp kết quả với chiến lược "An toàn là trên hết"
        if svm_pred == cnn_pred:  # Cả hai mô hình đồng ý
            final_prediction = svm_pred
            # Tính độ tin cậy dựa trên nhãn dự đoán
            if final_prediction == 1:  # Tin giả
                confidence = (svm_prob[1] + cnn_prob) / 2
            else:  # Tin thật
                confidence = (svm_prob[0] + (1 - cnn_prob)) / 2
            method = 'ensemble_agreement'
        else:  # Mô hình không đồng ý
            # Chiến lược "An toàn là trên hết": ưu tiên cảnh báo tin giả
            if svm_pred == 1 or cnn_pred == 1:  # Nếu có ít nhất một mô hình dự đoán là tin giả
                final_prediction = 1
                # Lấy xác suất cao nhất của tin giả
                confidence = max(svm_prob[1], cnn_prob)
                method = 'safety_first'
            else:  # Cả hai đều dự đoán là tin thật (trường hợp này không nên xảy ra)
                final_prediction = 0
                confidence = min(svm_prob[0], 1 - cnn_prob)
                method = 'ensemble_agreement'
        
        result = {
            'prediction': int(final_prediction),
            'confidence': float(confidence),
            'method': method,
            'details': {
                'svm_prediction': int(svm_pred),
                'svm_probability': float(svm_prob[1]),
                'cnn_prediction': int(cnn_pred),
                'cnn_probability': float(cnn_prob)
            }
        }
        
        logger.info(f"Kết quả: {'Tin Giả' if final_prediction == 1 else 'Tin Thật'} (Khó)")
        logger.info(f"Độ tin cậy: {confidence:.2f}")
        logger.info(f"Phương pháp: {method}")
        
        return result

def main():
    """Hàm chính để chạy hệ thống."""
    try:
        # Khởi tạo hệ thống
        detector = HybridPredictionSystem(SVM_MODEL_DIR, CNN_MODEL_DIR, RULES_PATH)
        
        # Tải dữ liệu test gốc
        logger.info("\n=== Đang tải dữ liệu test ===")
        test_raw_path = os.path.join(DATA_RAW_DIR, 'test.csv')
        df_test = pd.read_csv(test_raw_path, encoding='utf-8')
        logger.info(f"Đã tải {len(df_test)} mẫu từ {test_raw_path}")
        
        # Dự đoán cho từng mẫu
        predictions = []
        confidences = []
        methods = []
        details = []
        
        logger.info("\n=== Bắt đầu dự đoán trên tập test ===")
        for idx, row in df_test.iterrows():
            if idx % 100 == 0:
                logger.info(f"Đang xử lý mẫu {idx}/{len(df_test)}")
            
            result = detector.predict(row['post_message'])
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
            methods.append(result['method'])
            details.append(result['details'])
        
        # Tính toán các chỉ số đánh giá
        y_true = df_test['label'].values
        y_pred = np.array(predictions)
        
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame({
            'True Label': y_true,
            'Predicted': y_pred,
            'Confidence': confidences,
            'Method': methods
        })
        
        # Tính toán các chỉ số
        accuracy = (y_true == y_pred).mean()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Tạo bảng kết quả
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [accuracy, precision, recall, f1]
        })
        
        # Hiển thị bảng kết quả
        print("\n=== Kết quả đánh giá trên tập test ===")
        print("\nBảng chỉ số đánh giá:")
        print(metrics_df.to_string(index=False))
        
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_true, y_pred, target_names=['Tin Thật (0)', 'Tin Giả (1)']))
        
        # Vẽ biểu đồ đánh giá
        plt.figure(figsize=(20, 15))
        
        # Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Phân phối độ tin cậy
        plt.subplot(2, 2, 2)
        sns.histplot(data=results_df, x='Confidence', bins=50)
        plt.title('Phân phối độ tin cậy')
        plt.xlabel('Độ tin cậy')
        plt.ylabel('Số lượng')
        
        # Phân phối phương pháp
        plt.subplot(2, 2, 3)
        method_counts = results_df['Method'].value_counts()
        sns.barplot(x=method_counts.index, y=method_counts.values)
        plt.title('Phân phối phương pháp dự đoán')
        plt.xlabel('Phương pháp')
        plt.ylabel('Số lượng')
        plt.xticks(rotation=45)
        
        # Độ tin cậy theo phương pháp
        plt.subplot(2, 2, 4)
        sns.boxplot(data=results_df, x='Method', y='Confidence')
        plt.title('Độ tin cậy theo phương pháp')
        plt.xlabel('Phương pháp')
        plt.ylabel('Độ tin cậy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Thống kê theo phương pháp
        method_stats = results_df.groupby('Method').agg({
            'True Label': 'count',
            'Predicted': lambda x: (x == results_df.loc[x.index, 'True Label']).mean(),
            'Confidence': 'mean'
        }).round(4)
        
        method_stats.columns = ['Số lượng mẫu', 'Độ chính xác', 'Độ tin cậy trung bình']
        print("\nThống kê theo phương pháp:")
        print(method_stats)
        
    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
        raise

if __name__ == "__main__":
    main()