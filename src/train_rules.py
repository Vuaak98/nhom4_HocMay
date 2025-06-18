import os
import json
import re
import logging
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import từ các module khác trong src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import các biến đường dẫn
from config import (
    DATA_DIR, RULES_PATH, STOPWORDS_PATH,
    DATA_PROCESSED_DIR, TRAIN_CLEANED_FILE, TEST_CLEANED_FILE,
    RULE_SYSTEM_PATH, TRAIN_CONFUSION_MATRIX_PATH, TEST_CONFUSION_MATRIX_PATH,
    TRAIN_FEATURES_FILE, TEST_FEATURES_FILE,
    VISUALIZATION_DIR
)

# Đường dẫn cho file JSON kết quả
TRAIN_RESULTS_JSON_PATH = os.path.join(VISUALIZATION_DIR, 'train_results.json')
TEST_RESULTS_JSON_PATH = os.path.join(VISUALIZATION_DIR, 'test_results.json')

# Đường dẫn mới cho file log mistake
MISSED_FAKES_PATH = os.path.join(VISUALIZATION_DIR, 'missed_fakes.json')
MISSED_FAKES_TRAIN_PATH = os.path.join(VISUALIZATION_DIR, 'missed_fakes_train.json')

# === Cấu hình logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

class RuleSystem:
    """Lớp hệ thống luật kép: lọc tin và trích xuất đặc trưng."""
    
    def __init__(self, rules_path: str):
        """
        Khởi tạo hệ thống luật.
        
        Args:
            rules_path: Đường dẫn đến file JSON chứa các luật
        """
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self.components = self.rules.get('pattern_components', {})
        self.percentiles = {}
        self.is_fitted = False
        logger.info(f"✅ Đã tải {len(self.components)} thành phần luật từ {rules_path}")
        logger.info("✅ Đã khởi tạo RuleSystem")
    
    def _load_rules(self) -> Dict:
        """Tải các luật từ file JSON."""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            return rules
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải luật từ {self.rules_path}: {str(e)}")
            raise
    
    def _is_match(self, text: str, component_name: str) -> bool:
        """Kiểm tra văn bản có khớp với một component không."""
        keywords = self.components.get(component_name, {}).get('keywords', [])
        return any(kw.lower() in text.lower() for kw in keywords)
    
    def _is_reliable_real_pattern(self, text: str) -> bool:
        """
        Kiểm tra xem văn bản có khớp với mẫu tin tức báo chí không.
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            bool: True nếu khớp với mẫu tin tức báo chí
        """
        real_pattern = self.rules['patterns_for_filtering']['reliable_real_pattern']
        
        # Kiểm tra các điều kiện bắt buộc
        must_have = real_pattern['must_have']
        must_not_have = real_pattern['must_not_have']
        
        # Kiểm tra từng điều kiện bắt buộc
        has_authoritative = self._is_match(text, 'authoritative_source')
        has_informative = self._is_match(text, 'informative_tone')
        
        # Kiểm tra các điều kiện cấm
        has_pseudoscience = self._is_match(text, 'pseudoscience_hoax')
        has_scam = self._is_match(text, 'scam_call_to_action')
        has_spam = self._is_match(text, 'advertisement_spam')
        has_critical = self._is_match(text, 'critical_tone')
        
        # Log chi tiết cho debug
        logger.debug(f"\nKiểm tra mẫu tin thật:")
        logger.debug(f"  - Có nguồn tin đáng tin cậy: {has_authoritative}")
        logger.debug(f"  - Có văn phong báo chí: {has_informative}")
        logger.debug(f"  - Có dấu hiệu giả khoa học: {has_pseudoscience}")
        logger.debug(f"  - Có dấu hiệu lừa đảo: {has_scam}")
        logger.debug(f"  - Có dấu hiệu spam: {has_spam}")
        logger.debug(f"  - Có dấu hiệu chỉ trích: {has_critical}")
        
        # Nếu có bất kỳ dấu hiệu xấu nào, loại bỏ ngay
        if any([has_pseudoscience, has_scam, has_spam, has_critical]):
            return False
        
        # Phải có CẢ nguồn tin đáng tin cậy VÀ văn phong báo chí
        if not (has_authoritative and has_informative):
            return False
        
        return True
    
    def fit(self, df: pd.DataFrame) -> 'RuleSystem':
        """
        'Huấn luyện' bộ luật bằng cách tính toán các giá trị thống kê cần thiết từ dữ liệu train.
        
        Args:
            df: DataFrame chứa dữ liệu train
            
        Returns:
            RuleSystem: Đối tượng RuleSystem đã được fit
        """
        logger.info("🔄 Đang tính toán các giá trị thống kê từ dữ liệu train...")
        
        # Tính toán các percentiles cho các cột tương tác
        interaction_cols = ['num_like_post', 'num_comment_post', 'num_share_post']
        for col in interaction_cols:
            if col in df.columns:
                self.percentiles[col] = {
                    'p25': df[col].quantile(0.25),
                    'p50': df[col].quantile(0.50),
                    'p75': df[col].quantile(0.75),
                    'p90': df[col].quantile(0.90),
                    'p95': df[col].quantile(0.95),
                    'p99': df[col].quantile(0.99)
                }
        
        # Tính toán các ngưỡng cho các đặc trưng khác
        if 'cleaned_message' in df.columns:
            # Tỷ lệ chữ hoa
            df['uppercase_ratio'] = df['cleaned_message'].str.findall(r'[A-Z]').str.len() / (df['cleaned_message'].str.len() + 1e-6)
            self.percentiles['uppercase_ratio'] = {
                'p75': df['uppercase_ratio'].quantile(0.75),
                'p90': df['uppercase_ratio'].quantile(0.90),
                'p95': df['uppercase_ratio'].quantile(0.95)
            }
            
            # Số lượng hashtag
            df['hashtag_count'] = df['cleaned_message'].str.count('#')
            self.percentiles['hashtag_count'] = {
                'p75': df['hashtag_count'].quantile(0.75),
                'p90': df['hashtag_count'].quantile(0.90),
                'p95': df['hashtag_count'].quantile(0.95)
            }
        
        self.is_fitted = True
        logger.info("✅ Đã hoàn thành việc tính toán các giá trị thống kê")
        return self
    
    def save(self, filepath: str):
        """
        Lưu toàn bộ đối tượng RuleSystem đã được fit.
        
        Args:
            filepath: Đường dẫn để lưu file
        """
        if not self.is_fitted:
            logger.warning("⚠️ RuleSystem chưa được fit, các giá trị thống kê có thể không chính xác")
        
        try:
            joblib.dump(self, filepath)
            logger.info(f"✅ Đã lưu đối tượng RuleSystem vào {filepath}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu RuleSystem: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'RuleSystem':
        """
        Tải một đối tượng RuleSystem đã được lưu.
        
        Args:
            filepath: Đường dẫn đến file đã lưu
            
        Returns:
            RuleSystem: Đối tượng RuleSystem đã được tải
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"✅ Đã tải đối tượng RuleSystem từ {filepath}")
            return model
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải RuleSystem: {str(e)}")
            raise
    
    def classify_difficulty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phân loại độ khó của tin tức.
        
        Args:
            df: DataFrame chứa dữ liệu
            
        Returns:
            pd.DataFrame: DataFrame đã được phân loại
        """
        def get_verdict(row):
            # Sử dụng post_message gốc, không dùng cleaned_message
            text = str(row['post_message']).lower()
            
            # Kiểm tra các điều kiện cho tin thật dễ
            is_reliable = self._is_reliable_real_pattern(text)
            
            # Kiểm tra các điều kiện cho tin khó
            has_pseudoscience = self._is_match(text, 'pseudoscience_hoax')
            has_scam = self._is_match(text, 'scam_call_to_action')
            has_spam = self._is_match(text, 'advertisement_spam')
            has_critical = self._is_match(text, 'critical_tone')
            
            # Phân loại
            if is_reliable and not (has_pseudoscience or has_scam or has_spam or has_critical):
                return 'Tin Thật Dễ'
            else:
                return 'Tin Khó'
        
        df_copy = df.copy()
        df_copy['case_difficulty'] = df_copy.apply(get_verdict, axis=1)
        
        # Log thống kê phân loại
        total = len(df_copy)
        easy_count = (df_copy['case_difficulty'] == 'Tin Thật Dễ').sum()
        hard_count = (df_copy['case_difficulty'] == 'Tin Khó').sum()
        
        logger.info(f"\nThống kê phân loại:")
        logger.info(f"  - Tổng số mẫu: {total}")
        logger.info(f"  - Tin Thật Dễ: {easy_count} ({easy_count/total*100:.1f}%)")
        logger.info(f"  - Tin Khó: {hard_count} ({hard_count/total*100:.1f}%)")
        
        return df_copy

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các cột đặc trưng số từ các luật cho mô hình ML.
        
        Args:
            df: DataFrame chứa dữ liệu
            
        Returns:
            pd.DataFrame: DataFrame với các đặc trưng mới
        """
        if not self.is_fitted:
            logger.warning("⚠️ RuleSystem chưa được fit, các đặc trưng có thể không chính xác")
        
        df_featured = df.copy()
        # Sử dụng cleaned_message nếu có, ngược lại dùng post_message
        df_featured['text_for_analysis'] = df_featured.get('cleaned_message', df_featured['post_message'])
        # Chuyển đổi tất cả giá trị sang string và chuyển về chữ thường
        df_featured['text_for_analysis'] = df_featured['text_for_analysis'].astype(str)
        df_featured['text_for_analysis_lower'] = df_featured['text_for_analysis'].str.lower()
        
        # 1. Trích xuất điểm từ các component nội dung
        for comp_name, details in self.components.items():
            feature_name = f'rule_score_{comp_name}'
            keywords = details.get('keywords', [])
            weight = details.get('weight', 0)
            
            df_featured[feature_name] = df_featured['text_for_analysis_lower'].apply(
                lambda text: sum(1 for kw in keywords if kw.lower() in str(text).lower()) * weight
            )
        
        # 2. Xử lý các đặc trưng chất lượng văn bản
        df_featured['uppercase_ratio'] = df_featured['text_for_analysis'].str.findall(r'[A-Z]').str.len() / (df_featured['text_for_analysis'].str.len() + 1e-6)
        df_featured['feat_hashtag_count'] = df_featured['text_for_analysis'].str.count('#')
        df_featured['feat_url_count'] = df_featured['text_for_analysis'].str.count('http|www|<URL>')

        # 3. Trích xuất đặc trưng metadata (tương tác)
        meta_rules = self.rules.get('metadata_rules', {})
        
        # Tỷ lệ share/like
        share_rule = meta_rules.get('high_share_ratio', {})
        ratio_threshold = share_rule.get('ratio_threshold', 2.0)
        min_likes = share_rule.get('min_likes', 50)
        weight = share_rule.get('weight', 2.0)
        
        df_featured['rule_score_high_share_ratio'] = (
            ((df_featured['num_share_post'] / (df_featured['num_like_post'] + 1)) > ratio_threshold) &
            (df_featured['num_like_post'] > min_likes)
        ).astype(int) * weight

        # Số lượng hashtag
        hashtag_rule = meta_rules.get('many_hashtags', {})
        hashtag_threshold = hashtag_rule.get('hashtag_threshold', 5)
        weight = hashtag_rule.get('weight', 1.0)
        df_featured['rule_score_many_hashtags'] = (df_featured['feat_hashtag_count'] > hashtag_threshold).astype(int) * weight

        # 4. Tính toán tổng điểm fake và real
        fake_scores = [col for col in df_featured.columns if col.startswith('rule_score_') and col != 'rule_score_authoritative_source']
        real_scores = ['rule_score_authoritative_source']
        
        df_featured['feat_total_fake_score'] = df_featured[fake_scores].sum(axis=1)
        df_featured['feat_total_real_score'] = df_featured[real_scores].sum(axis=1)
        
        # 5. Tính toán tỷ lệ tin thật
        df_featured['feat_truth_ratio'] = df_featured['feat_total_real_score'] / (df_featured['feat_total_fake_score'] + df_featured['feat_total_real_score'] + 1e-6)
        
        # 6. Kiểm tra xung đột
        df_featured['feat_has_conflict'] = (
            (df_featured['feat_total_fake_score'] > 0) & 
            (df_featured['feat_total_real_score'] > 0)
        ).astype(int)
        
        # Xóa các cột tạm thời
        df_featured = df_featured.drop(['text_for_analysis', 'text_for_analysis_lower'], axis=1)
        
        return df_featured

def analyze_and_save_results(df_classified: pd.DataFrame, dataset_name: str, output_path: str, cm_path: str):
    """
    Phân tích kết quả của BỘ LỌC trên các tin được phân loại là DỄ.
    """
    logger.info(f"\n=== Phân tích độ chính xác trên tập {dataset_name} (chỉ xét các tin DỄ) ===")

    # BƯỚC QUAN TRỌNG: Lọc ra các tin đã được bộ lọc xử lý
    df_easy = df_classified[df_classified['case_difficulty'] != 'Tin Khó'].copy()
    
    if df_easy.empty:
        logger.warning(f"Không có 'Tin Dễ' nào được tìm thấy trong tập {dataset_name}. Bỏ qua phân tích.")
        return

    # Tạo nhãn dự đoán từ kết quả của bộ lọc
    # 0 là Thật, 1 là Giả
    df_easy['predicted_label'] = df_easy['case_difficulty'].apply(lambda x: 0 if x == 'Tin Thật Dễ' else 1)

    y_true = df_easy['label']
    y_pred = df_easy['predicted_label']

    # --- Báo cáo ra Console ---
    report_dict = classification_report(y_true, y_pred, target_names=['Tin Thật (0)', 'Tin Giả (1)'], output_dict=True, zero_division=0)
    logger.info(f"\nBáo cáo phân loại trên các tin DỄ của tập {dataset_name}:\n" +
                classification_report(y_true, y_pred, target_names=['Tin Thật (0)', 'Tin Giả (1)'], zero_division=0))

    # --- Vẽ và lưu Ma trận nhầm lẫn ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # Chỉ định labels để đảm bảo thứ tự đúng
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dự đoán Thật', 'Dự đoán Giả'], 
                yticklabels=['Thực tế Thật', 'Thực tế Giả'])
    plt.title(f'Ma trận nhầm lẫn bộ lọc trên tập {dataset_name} (Tin Dễ)')
    plt.ylabel('Nhãn Thực tế')
    plt.xlabel('Nhãn Dự đoán')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Đã lưu ma trận nhầm lẫn vào: {cm_path}")

    # --- Lưu kết quả chi tiết ra file JSON ---
    results = {
        'dataset_name': dataset_name,
        'total_samples_in_set': len(df_classified),
        'easy_samples_count': len(df_easy),
        'hard_samples_count': len(df_classified) - len(df_easy),
        'easy_samples_ratio': f"{len(df_easy) / len(df_classified) * 100:.2f}%" if len(df_classified) > 0 else "0.00%",
        'classification_report_on_easy_cases': report_dict,
        'confusion_matrix_on_easy_cases': cm.tolist()
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"✅ Đã lưu báo cáo chi tiết vào: {output_path}")

def run_rules_workflow(train_cleaned_df: pd.DataFrame, test_cleaned_df: pd.DataFrame) -> tuple:
    """
    Hàm workflow chính: nhận DataFrame đã làm sạch, trả về các DataFrame đặc trưng và rule_system đã fit.
    """
    rule_system = RuleSystem(RULES_PATH)
    rule_system.fit(train_cleaned_df)
    train_classified = rule_system.classify_difficulty(train_cleaned_df)
    test_classified = rule_system.classify_difficulty(test_cleaned_df)
    train_features = rule_system.extract_features(train_classified)
    test_features = rule_system.extract_features(test_classified)
    return train_features, test_features, rule_system, train_classified, test_classified

def main():
    # Đường dẫn mặc định
    train_cleaned_path = os.path.join('data', 'processed', 'train_cleaned.csv')
    test_cleaned_path = os.path.join('data', 'processed', 'test_cleaned.csv')
    train_cleaned_df = pd.read_csv(train_cleaned_path, encoding='utf-8')
    test_cleaned_df = pd.read_csv(test_cleaned_path, encoding='utf-8')
    train_features, test_features, rule_system, train_classified, test_classified = run_rules_workflow(train_cleaned_df, test_cleaned_df)
    # Ghi file nếu chạy từ CLI
    train_features.to_csv('data/features/train_features.csv', index=False, encoding='utf-8')
    test_features.to_csv('data/features/test_features.csv', index=False, encoding='utf-8')
    print("✅ Đã lưu đặc trưng rule.")

if __name__ == '__main__':
    main() 