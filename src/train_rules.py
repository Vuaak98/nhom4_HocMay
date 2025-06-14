import os
import pandas as pd
import logging
import sys
from typing import List, Dict, Tuple
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path để import từ các module khác trong src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import các biến đường dẫn và lớp bộ lọc
from src.config import (
    DATA_DIR, RULES_PATH, STOPWORDS_PATH,
    DATA_HARD_DIR, TRAIN_HARD_FILE, TEST_HARD_FILE,
    DATA_PROCESSED_DIR
)
from src.pipeline import RuleBasedFilterFromFile
from src.data_processing import DataProcessor

# === Cấu hình logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('main')

# === Cấu hình đường dẫn ===
TRAIN_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'train_cleaned.csv')
TEST_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'test_cleaned.csv')

class RuleBasedClassifier:
    def __init__(self, rules_path: str):
        """Khởi tạo bộ phân loại dựa trên luật."""
        self.rule_filter = RuleBasedFilterFromFile(rules_path=rules_path)
        logger.info("✅ Đã khởi tạo RuleBasedClassifier")
    
    def classify_batch(self, texts: List[str]) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """
        Phân loại một batch văn bản.
        
        Args:
            texts: Danh sách văn bản cần phân loại
            
        Returns:
            Tuple[List[str], Dict[str, pd.DataFrame]]: 
                - Danh sách nhãn dự đoán
                - Dictionary chứa các DataFrame đã phân loại
        """
        # Dự đoán nhãn
        predictions = []
        for text in tqdm(texts, desc="Phân loại văn bản"):
            trust_score, fake_score = self.rule_filter.calculate_scores(text)
            if trust_score == -1.0:
                predictions.append('Tin Thật Dễ')
            elif fake_score == 1.0:
                predictions.append('Tin Giả Dễ')
            else:
                predictions.append('Tin Khó')
        
        # Tạo DataFrame với kết quả
        df_results = pd.DataFrame({
            'text': texts,
            'prediction': predictions
        })
        
        # Phân loại thành các nhóm
        easy_real = df_results[df_results['prediction'] == 'Tin Thật Dễ']
        easy_fake = df_results[df_results['prediction'] == 'Tin Giả Dễ']
        hard = df_results[df_results['prediction'] == 'Tin Khó']
        
        return predictions, {
            'easy_real': easy_real,
            'easy_fake': easy_fake,
            'hard': hard
        }

def _calculate_metrics(df: pd.DataFrame, predictions: List[str]) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá.
    
    Args:
        df: DataFrame chứa dữ liệu
        predictions: Danh sách dự đoán
        
    Returns:
        Dict[str, float]: Các chỉ số đánh giá
    """
    total_samples = len(df)
    easy_real_df = df.loc[pd.Series(predictions) == 'Tin Thật Dễ']
    easy_fake_df = df.loc[pd.Series(predictions) == 'Tin Giả Dễ']
    hard_df = df.loc[pd.Series(predictions) == 'Tin Khó']

    correct_real = (easy_real_df['label'] == 0).sum()
    correct_fake = (easy_fake_df['label'] == 1).sum()
    total_correct_easy = correct_real + correct_fake

    # Thêm thông tin về các trường hợp phân loại sai
    misclassified_real = easy_real_df[easy_real_df['label'] == 1]['post_message'].tolist()
    misclassified_fake = easy_fake_df[easy_fake_df['label'] == 0]['post_message'].tolist()

    return {
        'total_samples': total_samples,
        'easy_real': len(easy_real_df),
        'easy_fake': len(easy_fake_df),
        'hard': len(hard_df),
        'correct_real': correct_real,
        'correct_fake': correct_fake,
        'total_correct_easy': total_correct_easy,
        'total_easy': len(easy_real_df) + len(easy_fake_df),
        'misclassified_real': misclassified_real,
        'misclassified_fake': misclassified_fake
    }

def analyze_and_report(metrics: Dict[str, float], dataset_name: str):
    """In báo cáo phân tích kết quả."""
    logger.info(f"\n=== Kết quả phân loại trên tập {dataset_name} ===")
    
    # Tính toán số lượng tin dễ và tin khó
    total_samples = metrics['total_samples']
    easy_real = metrics['easy_real']
    easy_fake = metrics['easy_fake']
    hard = metrics['hard']
    total_easy = easy_real + easy_fake
    
    # Tính độ chính xác cho tin dễ
    correct_real = metrics['correct_real']
    correct_fake = metrics['correct_fake']
    total_correct_easy = correct_real + correct_fake
    
    accuracy_real = correct_real / easy_real * 100 if easy_real > 0 else 0
    accuracy_fake = correct_fake / easy_fake * 100 if easy_fake > 0 else 0
    accuracy_total = total_correct_easy / total_easy * 100 if total_easy > 0 else 0
    
    # In kết quả tổng quan
    logger.info(f"\nTổng số mẫu: {total_samples}")
    logger.info(f"\nTin dễ phân biệt: {total_easy} ({total_easy/total_samples*100:.1f}%)")
    logger.info(f"- Tin Thật Dễ: {easy_real} ({easy_real/total_samples*100:.1f}%)")
    logger.info(f"- Tin Giả Dễ: {easy_fake} ({easy_fake/total_samples*100:.1f}%)")
    logger.info(f"\nTin khó: {hard} ({hard/total_samples*100:.1f}%)")
    
    if total_easy > 0:
        logger.info(f"\nĐộ chính xác trên tin dễ: {accuracy_total:.1f}%")
        logger.info(f"- Tin Thật Dễ: {accuracy_real:.1f}% ({correct_real}/{easy_real})")
        logger.info(f"- Tin Giả Dễ: {accuracy_fake:.1f}% ({correct_fake}/{easy_fake})")
        
        # In chi tiết các trường hợp phân loại sai
        if dataset_name == "train":
            logger.info("\n=== Chi tiết các trường hợp phân loại sai ===")
            
            # Lấy các trường hợp phân loại sai
            misclassified_real = metrics.get('misclassified_real', [])
            misclassified_fake = metrics.get('misclassified_fake', [])
            
            if misclassified_real:
                logger.info("\nTin Thật bị phân loại sai:")
                for idx, text in enumerate(misclassified_real[:10], 1):  # In 10 mẫu đầu tiên
                    logger.info(f"\n{idx}. {text[:200]}...")
            
            if misclassified_fake:
                logger.info("\nTin Giả bị phân loại sai:")
                for idx, text in enumerate(misclassified_fake[:10], 1):  # In 10 mẫu đầu tiên
                    logger.info(f"\n{idx}. {text[:200]}...")

def process_dataset(df: pd.DataFrame, classifier: RuleBasedClassifier, output_file: str) -> Tuple[List[str], Dict[str, float]]:
    """
    Xử lý một tập dữ liệu.
    
    Args:
        df: DataFrame chứa dữ liệu
        classifier: Bộ phân loại
        output_file: Đường dẫn file output
        
    Returns:
        Tuple[List[str], Dict[str, float]]: 
            - Danh sách dự đoán
            - Các chỉ số đánh giá
    """
    # Phân loại
    predictions, classified_data = classifier.classify_batch(df['post_message'].tolist())
    
    # Lưu tin khó
    if 'hard' in classified_data and not classified_data['hard'].empty:
        hard_data = df.loc[classified_data['hard'].index].copy()
        hard_data.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✅ Đã lưu {len(hard_data)} tin khó vào {output_file}")
    
    # Tính toán metrics
    metrics = _calculate_metrics(df, predictions)
    
    return predictions, metrics

def main():
    """Hàm chính để chạy bộ lọc, phân tích và lưu kết quả."""
    try:
        # **BƯỚC 1: TẠO THƯ MỤC LƯU TRỮ (NẾU CHƯA CÓ)**
        os.makedirs(DATA_HARD_DIR, exist_ok=True)
        
        # **BƯỚC 2: KHỞI TẠO BỘ PHÂN LOẠI**
        classifier = RuleBasedClassifier(rules_path=RULES_PATH)
        
        # **BƯỚC 3: XỬ LÝ TẬP TRAIN**
        logger.info("\n=== Xử lý tập train ===")
        logger.info(f"Đang nạp dữ liệu từ: {TRAIN_CLEANED_FILE}")
        df_train = pd.read_csv(TRAIN_CLEANED_FILE, encoding='utf-8')
        logger.info(f"Đã nạp {len(df_train)} mẫu từ {TRAIN_CLEANED_FILE}")
        
        train_predictions, train_metrics = process_dataset(
            df_train, 
            classifier,
            TRAIN_HARD_FILE
        )
        analyze_and_report(train_metrics, "train")

        # **BƯỚC 4: XỬ LÝ TẬP TEST**
        logger.info("\n=== Xử lý và đánh giá tập test ===")
        logger.info(f"Đang nạp dữ liệu từ: {TEST_CLEANED_FILE}")
        df_test = pd.read_csv(TEST_CLEANED_FILE, encoding='utf-8')
        logger.info(f"Đã nạp {len(df_test)} mẫu từ {TEST_CLEANED_FILE}")
        
        test_predictions, test_metrics = process_dataset(
            df_test,
            classifier,
            TEST_HARD_FILE
        )
        analyze_and_report(test_metrics, "test")

        # **BƯỚC 5: TỔNG KẾT CUỐI CÙNG**
        logger.info("\n=== Tổng kết kết quả ===")
        logger.info("\nTập train:")
        logger.info(f"Tổng số mẫu: {train_metrics['total_samples']}")
        logger.info(f"Tin Thật Dễ: {train_metrics['easy_real']} ({train_metrics['easy_real']/train_metrics['total_samples']*100:.1f}%)")
        logger.info(f"Tin Giả Dễ: {train_metrics['easy_fake']} ({train_metrics['easy_fake']/train_metrics['total_samples']*100:.1f}%)")
        logger.info(f"Tin Khó: {train_metrics['hard']} ({train_metrics['hard']/train_metrics['total_samples']*100:.1f}%)")
        
        logger.info("\nTập test:")
        logger.info(f"Tổng số mẫu: {test_metrics['total_samples']}")
        logger.info(f"Tin Thật Dễ: {test_metrics['easy_real']} ({test_metrics['easy_real']/test_metrics['total_samples']*100:.1f}%)")
        logger.info(f"Tin Giả Dễ: {test_metrics['easy_fake']} ({test_metrics['easy_fake']/test_metrics['total_samples']*100:.1f}%)")
        logger.info(f"Tin Khó: {test_metrics['hard']} ({test_metrics['hard']/test_metrics['total_samples']*100:.1f}%)")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Lỗi không tìm thấy file: {e}. Hãy đảm bảo các file train_cleaned.csv và test_cleaned.csv tồn tại trong {DATA_PROCESSED_DIR}.")
    except Exception as e:
        logger.error(f"❌ Lỗi không mong muốn xảy ra trong quá trình xử lý: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 