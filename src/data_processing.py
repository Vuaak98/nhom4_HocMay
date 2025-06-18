import os
import pandas as pd
import numpy as np
import re
import logging
import joblib
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Set
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
import unicodedata
from pyvi import ViTokenizer

# Thêm thư mục gốc vào PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import từ config.py
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, STOPWORDS_PATH,
    TRAIN_CLEANED_FILE, TEST_CLEANED_FILE, RULES_PATH
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """Lớp cơ sở cho các bộ xử lý dữ liệu."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseProcessor':
        """Học các thông số từ dữ liệu train."""
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng các phép biến đổi đã học lên dữ liệu."""
        pass

class InteractionProcessor(BaseProcessor):
    """Bộ xử lý cho các cột tương tác."""
    
    def __init__(self, outlier_quantile: float = 0.99):
        self.interaction_columns = ['num_like_post', 'num_comment_post', 'num_share_post']
        self.outlier_quantile = outlier_quantile
        self.capping_values = {}
        self.pattern = re.compile(r'[^0-9.kmb]')
    
    def fit(self, df: pd.DataFrame) -> 'InteractionProcessor':
        """Học các giá trị chặn trên từ dữ liệu train."""
        logger.info("Fitting InteractionProcessor...")
        for col in self.interaction_columns:
            if col in df.columns:
                # Chuyển đổi dữ liệu sang dạng số trước khi tính quantile
                numeric_data = df[col].apply(self._clean_interaction)
                # Chỉ tính quantile trên các giá trị hợp lệ (không phải 0)
                valid_data = numeric_data[numeric_data > 0]
                if not valid_data.empty:
                    self.capping_values[col] = valid_data.quantile(self.outlier_quantile)
                    logger.info(f"  - Learned capping value for {col}: {self.capping_values[col]:.2f}")
                else:
                    logger.warning(f"  - Không có giá trị hợp lệ cho {col}")
        return self
    
    @lru_cache(maxsize=1000)
    def _clean_interaction(self, value: str) -> float:
        """Làm sạch giá trị tương tác với cache."""
        try:
            if pd.isna(value) or str(value).lower() in ['unknown', '']:
                return 0
                
            value = str(value).lower().strip()
            
            # Loại bỏ các ký tự không mong muốn
            value = self.pattern.sub('', value)
            
            # Xử lý các trường hợp đặc biệt
            if not value or value == '.':
                return 0
                
            # Xử lý các hậu tố k, m, b
            multiplier = 1
            if 'k' in value:
                multiplier = 1000
                value = value.replace('k', '')
            elif 'm' in value:
                multiplier = 1000000
                value = value.replace('m', '')
            elif 'b' in value:
                multiplier = 1000000000
                value = value.replace('b', '')
            
            # Loại bỏ các dấu chấm thừa
            value = value.strip('.')
            
            # Chuyển đổi sang số
            result = float(value) * multiplier
            
            # Kiểm tra tính hợp lệ của kết quả
            if result < 0 or result > 1e12:  # Giới hạn hợp lý cho số tương tác
                logger.warning(f"Giá trị tương tác nằm ngoài phạm vi hợp lệ: {result}")
                return 0
                
            return result
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Không thể chuyển đổi giá trị tương tác: {value}, lỗi: {str(e)}")
            return 0
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng các phép biến đổi đã học lên dữ liệu."""
        logger.info("Transforming interaction data...")
        df_transformed = df.copy()
        
        # Làm sạch các giá trị tương tác
        for col in self.interaction_columns:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].apply(self._clean_interaction)
                # Áp dụng giá trị chặn trên đã học
                if col in self.capping_values:
                    df_transformed.loc[df_transformed[col] > self.capping_values[col], col] = self.capping_values[col]
        
        return df_transformed

class TimestampProcessor(BaseProcessor):
    """Bộ xử lý cho cột timestamp."""
    
    def __init__(self):
        self.time_columns = ['hours', 'days', 'weeks', 'months']
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}/\d{2}/\d{2}',  # DD/MM/YY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{2}-\d{2}-\d{2}',  # DD-MM-YY
        ]
    
    def fit(self, df: pd.DataFrame) -> 'TimestampProcessor':
        """Không cần học thông số từ dữ liệu train."""
        return self
    
    def _is_valid_timestamp(self, value: str) -> bool:
        """Kiểm tra tính hợp lệ của timestamp."""
        if pd.isna(value):
            return False
            
        value = str(value).strip()
        
        # Kiểm tra độ dài tối thiểu
        if len(value) < 8:
            return False
            
        # Kiểm tra các từ khóa thông thường
        common_words = ['tin', 'báo', 'thông', 'tin tức', 'cập nhật']
        if any(word in value.lower() for word in common_words):
            return False
            
        # Kiểm tra các định dạng timestamp phổ biến
        patterns = [
            # Định dạng ISO
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?',
            r'\d{4}-\d{2}-\d{2}',
            
            # Định dạng Việt Nam
            r'\d{2}/\d{2}/\d{4}[ ]\d{2}:\d{2}:\d{2}(\.\d+)?',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}/\d{2}/\d{2}',
            
            # Định dạng khác
            r'\d{2}-\d{2}-\d{4}[ ]\d{2}:\d{2}:\d{2}(\.\d+)?',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}-\d{2}-\d{2}',
            
            # Định dạng tiếng Anh
            r'[A-Za-z]+ \d{1,2}(st|nd|rd|th)? \d{4},? \d{2}:\d{2}:\d{2}(\.\d+)?',
            r'[A-Za-z]+ \d{1,2}(st|nd|rd|th)? \d{4}',
            
            # Định dạng Unix timestamp
            r'^\d{10}$',
            r'^\d{13}$'
        ]
        
        return any(re.search(pattern, value) for pattern in patterns)
    
    def _clean_timestamp(self, value: str, default_ts: pd.Timestamp) -> Tuple[pd.Timestamp, bool]:
        """
        Làm sạch timestamp.
        
        Returns:
            Tuple[pd.Timestamp, bool]: 
                - Timestamp đã làm sạch
                - Flag cho biết timestamp có bị thiếu không
        """
        try:
            # Chuyển về chữ thường và loại bỏ khoảng trắng
            value = str(value).lower().strip()
            
            # Nếu là NaN hoặc rỗng, trả về timestamp mặc định và đánh dấu là thiếu
            if pd.isna(value) or value == '' or value == 'nan':
                return default_ts, True
            
            # Thử parse Unix timestamp
            try:
                ts = pd.to_numeric(value)
                if 946684800 <= ts <= 1893456000:  # 2000-2030
                    dt = pd.to_datetime(ts, unit='s')
                    return dt.replace(year=2020), False
            except (ValueError, TypeError):
                pass
            
            # Thử parse các định dạng phổ biến
            try:
                # Định dạng: "june 11th 2020, 19:03:45.000"
                if ',' in value:
                    dt = pd.to_datetime(value)
                    return dt.replace(year=2020), False
                    
                # Định dạng: "2020-06-11 19:03:45"
                if '-' in value:
                    dt = pd.to_datetime(value)
                    return dt.replace(year=2020), False
                    
                # Định dạng: "11/06/2020 19:03:45"
                if '/' in value:
                    dt = pd.to_datetime(value)
                    return dt.replace(year=2020), False
                    
                # Định dạng: "11.06.2020 19:03:45"
                if '.' in value:
                    dt = pd.to_datetime(value)
                    return dt.replace(year=2020), False
                    
            except Exception:
                pass
            
            # Nếu không parse được, trả về timestamp mặc định và đánh dấu là thiếu
            logger.warning(f"Không thể parse timestamp: {value}")
            return default_ts, True
            
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý timestamp: {value}, lỗi: {str(e)}")
            return default_ts, True
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng các phép biến đổi lên dữ liệu."""
        logger.info("Transforming timestamp data...")
        df_transformed = df.copy()
        
        if 'timestamp_post' in df_transformed.columns:
            # Xử lý timestamp và lấy flag thiếu
            timestamp_results = df_transformed['timestamp_post'].astype(str).apply(
                lambda x: self._clean_timestamp(x, pd.to_datetime(f'2020-01-01 00:00:00'))
            )
            
            # Tách timestamp và flag
            df_transformed['timestamp_post'] = timestamp_results.apply(lambda x: x[0])
            df_transformed['is_timestamp_missing'] = timestamp_results.apply(lambda x: x[1])
            
            # Thống kê số lượng timestamp thiếu
            missing_count = df_transformed['is_timestamp_missing'].sum()
            logger.info(f"\nThống kê timestamp:")
            logger.info(f"  - Tổng số dòng: {len(df_transformed)}")
            logger.info(f"  - Số timestamp thiếu: {missing_count} ({missing_count/len(df_transformed)*100:.1f}%)")
            
            # Nếu có timestamp thiếu, điền bằng giá trị trung vị
            if missing_count > 0:
                # Tính trung vị của các timestamp hợp lệ
                valid_timestamps = df_transformed[~df_transformed['is_timestamp_missing']]['timestamp_post']
                if not valid_timestamps.empty:
                    median_timestamp = valid_timestamps.median()
                    logger.info(f"  - Điền {missing_count} timestamp thiếu bằng giá trị trung vị: {median_timestamp}")
                    
                    # Điền giá trị trung vị cho các timestamp thiếu
                    df_transformed.loc[df_transformed['is_timestamp_missing'], 'timestamp_post'] = median_timestamp
            
            # Trích xuất các trường thời gian
            df_transformed['hours'] = df_transformed['timestamp_post'].dt.hour.astype('int32')
            df_transformed['days'] = df_transformed['timestamp_post'].dt.day.astype('int32')
            df_transformed['weeks'] = df_transformed['timestamp_post'].dt.isocalendar().week.astype('int32')
            df_transformed['months'] = df_transformed['timestamp_post'].dt.month.astype('int32')
            
            # Thống kê phân bố thời gian
            logger.info("\nPhân bố thời gian:")
            logger.info(f"  - Giờ: {df_transformed['hours'].value_counts().sort_index().to_dict()}")
            logger.info(f"  - Ngày: {df_transformed['days'].value_counts().sort_index().to_dict()}")
            logger.info(f"  - Tuần: {df_transformed['weeks'].value_counts().sort_index().to_dict()}")
            logger.info(f"  - Tháng: {df_transformed['months'].value_counts().sort_index().to_dict()}")
            
            # Thống kê phân bố thời gian theo flag thiếu
            logger.info("\nPhân bố thời gian theo flag thiếu:")
            for flag in [True, False]:
                flag_desc = "Timestamp thiếu" if flag else "Timestamp hợp lệ"
                subset = df_transformed[df_transformed['is_timestamp_missing'] == flag]
                logger.info(f"\n{flag_desc}:")
                logger.info(f"  - Giờ: {subset['hours'].value_counts().sort_index().to_dict()}")
                logger.info(f"  - Ngày: {subset['days'].value_counts().sort_index().to_dict()}")
                logger.info(f"  - Tuần: {subset['weeks'].value_counts().sort_index().to_dict()}")
                logger.info(f"  - Tháng: {subset['months'].value_counts().sort_index().to_dict()}")
        
        return df_transformed

class TextProcessor(BaseProcessor):
    """Bộ xử lý cho cột văn bản."""
    
    def __init__(self, stopwords_set: Optional[Set[str]] = None):
        self.stopwords = stopwords_set or set()
        # Cập nhật pattern để giữ lại dấu gạch dưới trong hashtag và từ ghép
        self.pattern = re.compile(r'[^\w\s_]')
        # Pattern để tách hashtag
        self.hashtag_pattern = re.compile(r'#\w+')
        # Pattern để nhận diện URL
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        # Pattern để nhận diện số điện thoại
        self.phone_pattern = re.compile(r'\b\d{10,11}\b')
        # Pattern để nhận diện email
        self.email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    
    def fit(self, df: pd.DataFrame) -> 'TextProcessor':
        """Không cần học thông số từ dữ liệu train."""
        return self
    
    @lru_cache(maxsize=1000)
    def _clean_text(self, text: str) -> str:
        """Làm sạch văn bản với cache."""
        try:
            if pd.isna(text):
                return ""
                
            # Chuyển về chữ thường
            text = str(text).lower()
            
            # Chuẩn hóa Unicode về dạng NFC
            text = unicodedata.normalize('NFC', text)
            
            # Loại bỏ URL
            text = self.url_pattern.sub('url', text)
            
            # Loại bỏ email
            text = self.email_pattern.sub('email', text)
            
            # Loại bỏ số điện thoại
            text = self.phone_pattern.sub('phone', text)
            
            # Xử lý hashtag
            hashtags = self.hashtag_pattern.findall(text)
            for hashtag in hashtags:
                # Loại bỏ dấu # và thay thế dấu gạch dưới bằng khoảng trắng
                clean_hashtag = hashtag[1:].replace('_', ' ')
                text = text.replace(hashtag, clean_hashtag)
            
            # Tách từ tiếng Việt
            text = ViTokenizer.tokenize(text)
            
            # Loại bỏ các ký tự đặc biệt, giữ lại dấu gạch dưới
            text = self.pattern.sub(' ', text)
            
            # Loại bỏ khoảng trắng thừa
            text = ' '.join(text.split())
            
            # Loại bỏ stopwords
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            
            return ' '.join(words)
            
        except Exception as e:
            logger.warning(f"Lỗi khi làm sạch văn bản: {str(e)}")
            return ""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng các phép biến đổi lên dữ liệu."""
        logger.info("Transforming text data...")
        df_transformed = df.copy()
        
        if 'post_message' in df_transformed.columns:
            # Giữ nguyên văn bản gốc
            df_transformed['post_message'] = df_transformed['post_message'].fillna('')
            
            # Thêm cột cleaned_message
            df_transformed['cleaned_message'] = df_transformed['post_message'].apply(self._clean_text)
            
            # Thống kê độ dài văn bản
            df_transformed['text_length'] = df_transformed['cleaned_message'].str.len()
            
            # Thống kê số từ
            df_transformed['word_count'] = df_transformed['cleaned_message'].str.split().str.len()
            
            # Thống kê số câu (dựa trên dấu chấm)
            df_transformed['sentence_count'] = df_transformed['post_message'].str.count(r'[.!?]+')
            
            # Thống kê số hashtag
            df_transformed['hashtag_count'] = df_transformed['post_message'].str.count(r'#\w+')
            
            # Thống kê số URL
            df_transformed['url_count'] = df_transformed['post_message'].str.count(r'https?://\S+|www\.\S+')
            
            # Tính tỷ lệ stopwords
            df_transformed['stopwords_ratio'] = df_transformed['post_message'].str.lower().apply(
                lambda text: sum(1 for word in text.split() if word in self.stopwords) / (len(text.split()) + 1e-6)
            )
            
            # Tính tỷ lệ từ ghép (có dấu gạch dưới)
            df_transformed['compound_word_ratio'] = df_transformed['cleaned_message'].apply(
                lambda text: sum(1 for word in text.split() if '_' in word) / (len(text.split()) + 1e-6)
            )
            
            # Log thống kê
            logger.info(f"\nThống kê văn bản:")
            logger.info(f"  - Độ dài trung bình: {df_transformed['text_length'].mean():.1f}")
            logger.info(f"  - Số từ trung bình: {df_transformed['word_count'].mean():.1f}")
            logger.info(f"  - Số câu trung bình: {df_transformed['sentence_count'].mean():.1f}")
            logger.info(f"  - Số hashtag trung bình: {df_transformed['hashtag_count'].mean():.1f}")
            logger.info(f"  - Số URL trung bình: {df_transformed['url_count'].mean():.1f}")
            logger.info(f"  - Tỷ lệ stopwords trung bình: {df_transformed['stopwords_ratio'].mean():.2f}")
            logger.info(f"  - Tỷ lệ từ ghép trung bình: {df_transformed['compound_word_ratio'].mean():.2f}")
        
        return df_transformed

def get_file_hash(filepath: str) -> str:
    """Tính hash MD5 của file."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

class DataProcessor:
    """Bộ xử lý dữ liệu chính."""
    
    def __init__(self, outlier_quantile: float = 0.99, default_year: int = 2020):
        self.outlier_quantile = outlier_quantile
        self.default_year = default_year
        self.default_timestamp = pd.Timestamp(f"{default_year}-01-01")
        self.is_fitted = False
        self.capping_values_ = {}
        self.stopwords_hash = None
        self.rules_hash = None
        
        # Khởi tạo các bộ xử lý
        self.interaction_processor = InteractionProcessor(outlier_quantile)
        self.timestamp_processor = TimestampProcessor()
        self.text_processor = TextProcessor()
        
        # Các cột cần xử lý
        self.interaction_columns = ['num_like_post', 'num_comment_post', 'num_share_post']
    
    def fit(self, df: pd.DataFrame, stopwords_path: str = None, rules_path: str = None) -> 'DataProcessor':
        """Học các thông số từ dữ liệu train."""
        logger.info("Fitting DataProcessor...")
        
        # Tải từ dừng nếu có
        if stopwords_path:
            stopwords = load_stopwords(stopwords_path)
            self.text_processor = TextProcessor(stopwords)
            self.stopwords_hash = get_file_hash(stopwords_path)
        
        # Tải luật nếu có
        if rules_path:
            self.rules_hash = get_file_hash(rules_path)
        
        # Fit các bộ xử lý
        self.interaction_processor.fit(df)
        self.timestamp_processor.fit(df)
        self.text_processor.fit(df)
        
        # Lưu các giá trị chặn trên
        self.capping_values_ = self.interaction_processor.capping_values
        
        self.is_fitted = True
        logger.info("✅ DataProcessor đã được fit thành công")
        return self
    
    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Áp dụng các phép biến đổi lên dữ liệu."""
        logger.info("Transforming data...")
        df_transformed = df.copy()
        
        # Bước 1: Xử lý văn bản
        df_transformed = self.text_processor.transform(df_transformed)
        
        # Bước 2: Xử lý các cột tương tác
        df_transformed = self.interaction_processor.transform(df_transformed)
        
        # Bước 3: Xử lý timestamp
        df_transformed = self.timestamp_processor.transform(df_transformed)
        
        # Bước 4: Xử lý ngoại lai
        if self.is_fitted:
            for col, cap_value in self.capping_values_.items():
                if col in df_transformed.columns:
                    df_transformed.loc[df_transformed[col] > cap_value, col] = cap_value
        else:
            logger.warning("DataProcessor chưa được fit. Bỏ qua xử lý ngoại lai.")
        
        logger.info(f"✅ Transformation complete. Final data has {len(df_transformed)} rows")
        return df_transformed
    
    def save(self, filepath: str):
        """Lưu đối tượng DataProcessor."""
        if not self.is_fitted:
            logger.warning("DataProcessor chưa được fit. Các giá trị thống kê có thể không chính xác")
        
        try:
            joblib.dump(self, filepath)
            logger.info(f"✅ Đã lưu DataProcessor vào {filepath}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu DataProcessor: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str, stopwords_path: str = None, rules_path: str = None) -> 'DataProcessor':
        """Tải đối tượng DataProcessor đã lưu."""
        try:
            processor = joblib.load(filepath)
            
            # Kiểm tra hash của các file cấu hình
            if stopwords_path and processor.stopwords_hash:
                current_hash = get_file_hash(stopwords_path)
                if current_hash != processor.stopwords_hash:
                    logger.warning("File stopwords đã thay đổi kể từ khi pipeline được huấn luyện")
                    
            if rules_path and processor.rules_hash:
                current_hash = get_file_hash(rules_path)
                if current_hash != processor.rules_hash:
                    logger.warning("File rules đã thay đổi kể từ khi pipeline được huấn luyện")
            
            logger.info(f"✅ Đã tải DataProcessor từ {filepath}")
            return processor
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải DataProcessor: {str(e)}")
            raise

def load_stopwords(filepath: str) -> Set[str]:
    """Tải danh sách từ dừng từ file."""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ Không tìm thấy file từ dừng tại '{filepath}'.")
            return set()
            
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
        logger.info(f"✅ Tải thành công {len(stopwords)} từ dừng.")
        return stopwords
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải từ dừng: {str(e)}")
        return set()

def run_data_processing_workflow(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, 'DataProcessor']:
    """
    Hàm workflow chính: nhận DataFrame, trả về DataFrame đã làm sạch và processor đã fit.
    """
    stopwords = load_stopwords(STOPWORDS_PATH)
    processor = DataProcessor()
    processor.text_processor.stopwords = stopwords
    processor.fit(df_train, stopwords_path=STOPWORDS_PATH, rules_path=RULES_PATH)
    train_cleaned = processor.transform(df_train, is_training=True)
    test_cleaned = processor.transform(df_test, is_training=False)
    return train_cleaned, test_cleaned, processor

def main():
    # Đường dẫn mặc định
    train_raw_path = os.path.join('data', 'raw', 'train.csv')
    test_raw_path = os.path.join('data', 'raw', 'test.csv')
    stopwords_path = os.path.join('config', 'resources', 'vietnamese-stopwords.txt')
    rules_path = os.path.join('config', 'rules.json')
    df_train = pd.read_csv(train_raw_path, encoding='utf-8')
    df_test = pd.read_csv(test_raw_path, encoding='utf-8')
    train_cleaned, test_cleaned, processor = run_data_processing_workflow(df_train, df_test)
    # Ghi file nếu chạy từ CLI
    train_cleaned.to_csv('data/processed/train_cleaned.csv', index=False, encoding='utf-8')
    test_cleaned.to_csv('data/processed/test_cleaned.csv', index=False, encoding='utf-8')
    print("✅ Đã lưu dữ liệu đã làm sạch.")

if __name__ == '__main__':
    main()