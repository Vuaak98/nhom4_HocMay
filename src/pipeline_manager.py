import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import os
import logging
from typing import Dict
from src.data_processing import run_data_processing_workflow
from src.train_rules import run_rules_workflow
from src.train_evaluate import main_workflow as run_training_workflow, setup_paths
from src.utils.validation import validate_dataframe
from src.utils.hashing import get_file_hash
from config import FEATURES_DIR, TRAIN_FEATURES_FILE, TEST_FEATURES_FILE
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _hash_path(path):
    return path + '.hash'

def _check_and_save_hash(input_path, output_path):
    """Kiểm tra hash file input và lưu hash vào file .hash bên cạnh output."""
    input_hash = get_file_hash(input_path)
    hash_file = _hash_path(output_path)
    if os.path.exists(output_path) and os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == input_hash:
            logging.info(f"✅ Checkpoint: {output_path} đã hợp lệ, bỏ qua bước này.")
            return True
    # Nếu chưa có hoặc hash khác, lưu lại hash mới
    with open(hash_file, 'w') as f:
        f.write(input_hash)
    return False

def run_full_pipeline(train_raw_path: str, test_raw_path: str, progress_callback=None) -> Dict:
    paths = setup_paths()
    """
    Pipeline có checkpointing: chỉ chạy lại bước nếu dữ liệu đầu vào thay đổi.
    """
    # --- Giai đoạn 1: Xử lý dữ liệu ---
    logging.info("===== GIAI ĐOẠN 1: XỬ LÝ DỮ LIỆU =====")
    # Checkpoint cho train_features.csv
    skip_data_processing = False
    if os.path.exists(TRAIN_FEATURES_FILE) and os.path.exists(TEST_FEATURES_FILE):
        # Nếu hash của train/test raw không đổi, bỏ qua bước xử lý dữ liệu và luật
        train_ok = _check_and_save_hash(train_raw_path, TRAIN_FEATURES_FILE)
        test_ok = _check_and_save_hash(test_raw_path, TEST_FEATURES_FILE)
        if train_ok and test_ok:
            skip_data_processing = True
    if not skip_data_processing:
        df_train = pd.read_csv(train_raw_path, encoding='utf-8')
        df_test = pd.read_csv(test_raw_path, encoding='utf-8')
        validate_dataframe(df_train, ['post_message', 'label'], 'train_df')
        validate_dataframe(df_test, ['post_message', 'label'], 'test_df')
        if progress_callback: progress_callback(0.1, "Bắt đầu xử lý dữ liệu...")
        train_cleaned_df, test_cleaned_df, data_processor = run_data_processing_workflow(df_train, df_test)
        # Lưu file cleaned vào processed
        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
        train_cleaned_df.to_csv(os.path.join('data', 'processed', 'train_cleaned.csv'), index=False, encoding='utf-8')
        test_cleaned_df.to_csv(os.path.join('data', 'processed', 'test_cleaned.csv'), index=False, encoding='utf-8')
        # Lưu data_processor vào processors
        joblib.dump(data_processor, os.path.join(paths['processors'], 'data_processor.pkl'))
        if progress_callback: progress_callback(0.4, "Đã xử lý xong. Bắt đầu áp dụng luật...")
        logging.info("\n===== GIAI ĐOẠN 2: ÁP DỤNG LUẬT & TRÍCH XUẤT ĐẶC TRƯNG =====")
        train_features_df, test_features_df, rule_system, train_classified, test_classified = run_rules_workflow(
            train_cleaned_df, test_cleaned_df
        )
        os.makedirs(FEATURES_DIR, exist_ok=True)
        train_features_df.to_csv(TRAIN_FEATURES_FILE, index=False)
        test_features_df.to_csv(TEST_FEATURES_FILE, index=False)
        # Lưu rule_system vào processors
        joblib.dump(rule_system, os.path.join(paths['processors'], 'rule_system.pkl'))
        # Lưu tokenizer nếu có (nếu đã fit ở bước này)
        # (tokenizer sẽ được lưu lại ở bước train_evaluate, nhưng nếu có ở đây thì lưu luôn)
        # Lưu hash cho checkpoint
        _check_and_save_hash(train_raw_path, TRAIN_FEATURES_FILE)
        _check_and_save_hash(test_raw_path, TEST_FEATURES_FILE)
        logging.info(f"Đã lưu các tệp đặc trưng tạm thời vào '{FEATURES_DIR}'")
    else:
        logging.info("✅ Đã có đặc trưng, bỏ qua bước xử lý dữ liệu và luật.")
    # --- Giai đoạn 3: Huấn luyện và Đánh giá ---
    if progress_callback: progress_callback(0.7, "Đã có đặc trưng. Bắt đầu huấn luyện...")
    logging.info("\n===== GIAI ĐOẠN 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ =====")
    final_report_df = run_training_workflow()
    if progress_callback: progress_callback(1.0, "Hoàn tất!")
    logging.info("\n✅✅✅ PIPELINE HOÀN TẤT ✅✅✅")
    return {
        "final_report": final_report_df,
    }

if __name__ == '__main__':
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    results = run_full_pipeline(train_path, test_path)
    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(results['final_report'].to_string(index=False)) 