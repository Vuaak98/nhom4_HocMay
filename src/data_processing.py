import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, Dict, List, Union, Tuple

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('data_processing')

class DataProcessor:
    def __init__(self, outlier_quantile=0.99):
        """Khởi tạo bộ xử lý dữ liệu."""
        self.interaction_columns = ['num_like_post', 'num_comment_post', 'num_share_post']
        self.required_columns_base = ['id', 'post_message']
        self.outlier_quantile = outlier_quantile
        logger.info(f"✅ Đã khởi tạo DataProcessor với outlier_quantile={outlier_quantile}")

    def _clean_interaction(self, value) -> float:
        """Làm sạch giá trị tương tác."""
        try:
            if pd.isna(value): return np.nan
            value = str(value).lower().strip()
            value = re.sub(r'[^0-9.km]', '', value)
            multiplier = 1
            if 'k' in value:
                multiplier = 1000
                value = value.replace('k', '')
            elif 'm' in value:
                multiplier = 1000000
                value = value.replace('m', '')
            return float(value) * multiplier
        except (ValueError, TypeError) as e:
            logger.warning(f"Không thể chuyển đổi giá trị tương tác: {value}, lỗi: {str(e)}")
            return np.nan

    def _clean_timestamp(self, value) -> float:
        """Làm sạch giá trị thời gian."""
        try:
            if pd.isna(value): return np.nan
            try:
                return pd.to_numeric(value)
            except (ValueError, TypeError):
                return pd.to_datetime(value).timestamp()
        except Exception as e:
            logger.warning(f"Không thể chuyển đổi timestamp: {value}, lỗi: {str(e)}")
            return np.nan

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch DataFrame."""
        try:
            logger.info("  - Bắt đầu quy trình làm sạch...")
            
            # Kiểm tra dữ liệu đầu vào
            if df.empty:
                logger.error("❌ DataFrame trống!")
                return df
            
            # Thay thế các giá trị null
            df.replace(["unknown", "", "null", "NULL", "None", "NONE"], np.nan, inplace=True)
            
            # Làm sạch các cột tương tác
            for col in self.interaction_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._clean_interaction).fillna(0)
                    logger.info(f"  - Đã làm sạch cột {col}")
            
            # Làm sạch timestamp
            if 'timestamp_post' in df.columns:
                df['timestamp_post'] = df['timestamp_post'].apply(self._clean_timestamp).fillna(0)
                df['timestamp_post'] = pd.to_datetime(df['timestamp_post'], unit='s', errors='coerce')
                logger.info("  - Đã làm sạch cột timestamp_post")
            
            # Xử lý giá trị ngoại lai
            logger.info("  - Đang xử lý các giá trị ngoại lai...")
            for col in self.interaction_columns:
                if col in df.columns:
                    cap_value = df[col].quantile(self.outlier_quantile)
                    df.loc[df[col] > cap_value, col] = cap_value
                    logger.info(f"  - Đã xử lý ngoại lai cho cột {col}")
            
            # Kiểm tra và loại bỏ các mẫu thiếu dữ liệu quan trọng
            required_cols = self.required_columns_base.copy()
            if 'label' in df.columns:
                required_cols.append('label')
            df.dropna(subset=required_cols, inplace=True)
            
            logger.info("  - Quy trình làm sạch hoàn tất.")
            return df
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi làm sạch DataFrame: {str(e)}")
            raise

    def process_data(self, input_path: str, stopwords_path: str) -> Dict[str, pd.DataFrame]:
        """Xử lý dữ liệu một lần duy nhất và trả về tất cả các dạng cần thiết."""
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(input_path):
                logger.error(f"❌ Không tìm thấy file: {input_path}")
                return {}
                
            file_name = os.path.basename(input_path)
            logger.info(f"\n================ BẮT ĐẦU XỬ LÝ FILE: {file_name} ================")
            
            # Đọc file
            df = pd.read_csv(input_path, encoding='utf-8')
            logger.info(f"✅ Đọc thành công {len(df)} dòng.")
            
            # Làm sạch dữ liệu cơ bản
            df_cleaned = self.clean_dataframe(df)
            
            # Tạo đặc trưng cho ML
            df_ml = create_full_features(df_cleaned, stopwords_path, "ML Features")
            
            return {
                'cleaned': df_cleaned,
                'ml': df_ml
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi xử lý file '{input_path}': {str(e)}")
            return {}

def load_stopwords(filepath: str) -> set:
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

def text_preprocessor(text: str, stopwords_set: Optional[set] = None) -> str:
    """Làm sạch văn bản."""
    try:
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        
        # Giữ lại các ký tự đặc biệt quan trọng
        text = re.sub(r'[^a-z0-9àáạảãăằắặẳẵâầấậẩẫèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if stopwords_set:
            words = text.split()
            # Chỉ loại bỏ stopwords nếu còn ít nhất 3 từ
            if len(words) > 3:
                text = ' '.join([word for word in words if word not in stopwords_set])
                
        return text
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi xử lý văn bản: {str(e)}")
        return ""

def create_full_features(df: pd.DataFrame, stopwords_path: str, df_name: str = "Data") -> pd.DataFrame:
    """Trích xuất tất cả các đặc trưng cần thiết."""
    try:
        logger.info(f"  -> Bắt đầu trích xuất đặc trưng cho {df_name}...")
        
        # Kiểm tra dữ liệu đầu vào
        if df.empty:
            logger.error(f"❌ Dữ liệu {df_name} trống!")
            return df
        
        logger.info(f"  - Số lượng mẫu ban đầu: {len(df)}")
        
        df_featured = df.copy()

        # Đảm bảo các cột cần thiết tồn tại
        required_columns = {
            'timestamp_post': pd.Timestamp.now(),
            'num_like_post': 0,
            'num_comment_post': 0,
            'num_share_post': 0
        }
        
        for col, default_value in required_columns.items():
            if col not in df_featured.columns:
                df_featured[col] = default_value
                logger.info(f"  - Đã thêm cột {col} với giá trị mặc định")

        # Trích xuất các đặc trưng mới từ văn bản
        df_featured['uppercase_ratio'] = df_featured['post_message'].str.findall(
            r'[A-ZÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]'
        ).str.len() / (df_featured['post_message'].str.len() + 1e-6)
        df_featured['num_special_chars'] = df_featured['post_message'].str.count(r'[!?,.(){}\[\]]')
        df_featured['num_words'] = df_featured['post_message'].str.split().str.len()
        logger.info("  - Đã trích xuất đặc trưng văn bản")

        # Làm sạch văn bản chính
        stopwords = load_stopwords(stopwords_path)
        df_featured['cleaned_message'] = df_featured['post_message'].apply(lambda x: text_preprocessor(x, stopwords))
        
        # Kiểm tra văn bản sau khi làm sạch
        empty_texts = df_featured['cleaned_message'].str.len() == 0
        if empty_texts.any():
            logger.warning(f"  ⚠️ Có {empty_texts.sum()} văn bản trống sau khi làm sạch")
            # Sử dụng văn bản gốc nếu văn bản đã làm sạch trống
            df_featured.loc[empty_texts, 'cleaned_message'] = df_featured.loc[empty_texts, 'post_message'].str.lower()

        # Trích xuất Đặc trưng từ Thời gian
        df_featured['timestamp_post'] = pd.to_datetime(df_featured['timestamp_post'], errors='coerce')
        df_featured['hour'] = df_featured['timestamp_post'].dt.hour
        df_featured['weekday'] = df_featured['timestamp_post'].dt.dayofweek
        df_featured['day'] = df_featured['timestamp_post'].dt.day
        df_featured['month'] = df_featured['timestamp_post'].dt.month
        logger.info("  - Đã trích xuất đặc trưng thời gian")

        # Điền giá trị rỗng
        numeric_cols_to_fill = df_featured.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols_to_fill:
            df_featured[col] = df_featured[col].fillna(0)
        
        # Loại bỏ các mẫu không hợp lệ
        df_featured = df_featured.dropna(subset=['post_message', 'cleaned_message'])
        logger.info(f"  - Số lượng mẫu sau khi xử lý: {len(df_featured)}")
        
        return df_featured
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi trích xuất đặc trưng: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        from src.config import (
            DATA_RAW_DIR, DATA_PROCESSED_DIR, STOPWORDS_PATH,
            TRAIN_CLEANED_FILE, TEST_CLEANED_FILE
        )
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        
        # Xử lý dữ liệu train
        train_raw = os.path.join(DATA_RAW_DIR, 'train.csv')
        processor = DataProcessor()
        train_data = processor.process_data(train_raw, STOPWORDS_PATH)
        
        if train_data:
            # Lưu dữ liệu đã làm sạch
            train_data['cleaned'].to_csv(TRAIN_CLEANED_FILE, index=False, encoding='utf-8')
            logger.info(f"✅ Đã lưu dữ liệu train đã làm sạch vào: {TRAIN_CLEANED_FILE}")
        
        # Xử lý dữ liệu test
        test_raw = os.path.join(DATA_RAW_DIR, 'test.csv')
        test_data = processor.process_data(test_raw, STOPWORDS_PATH)
        
        if test_data:
            # Lưu dữ liệu đã làm sạch
            test_data['cleaned'].to_csv(TEST_CLEANED_FILE, index=False, encoding='utf-8')
            logger.info(f"✅ Đã lưu dữ liệu test đã làm sạch vào: {TEST_CLEANED_FILE}")
                
    except Exception as e:
        logger.error(f"❌ Lỗi không mong muốn: {str(e)}")
        raise