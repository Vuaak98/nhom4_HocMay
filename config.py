import os

# Đường dẫn gốc của dự án
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến các thư mục dữ liệu
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DATA_HARD_DIR = os.path.join(DATA_DIR, 'hard')

# Đường dẫn đến các file dữ liệu
TRAIN_FILE = os.path.join(DATA_RAW_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_RAW_DIR, 'test.csv')
TRAIN_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'train_cleaned.csv')
TEST_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'test_cleaned.csv')
TRAIN_HARD_FILE = os.path.join(DATA_HARD_DIR, 'train_hard.csv')
TEST_HARD_FILE = os.path.join(DATA_HARD_DIR, 'test_hard.csv')

# Đường dẫn đến file từ dừng và luật
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
STOPWORDS_PATH = os.path.join(CONFIG_DIR, 'vietnamese-stopwords.txt')
RULES_PATH = os.path.join(CONFIG_DIR, 'rules.json')

# Đường dẫn đến các thư mục mô hình
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SVM_MODEL_DIR = os.path.join(MODELS_DIR, 'svm')
CNN_MODEL_DIR = os.path.join(MODELS_DIR, 'cnn')

# Cấu hình cho CNN
CNN_CONFIG = {
    'max_words': 10000,
    'max_len': 300,
    'embedding_dim': 100,
    'filters': 128,
    'kernel_size': 5,
    'hidden_dims': 64,
    'dropout_rates': {
        'embedding': 0.2,
        'conv': 0.3,
        'dense': 0.4
    },
    'l2_reg': 0.01,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'early_stopping': {
        'patience': 5,
        'min_delta': 0.001
    }
}

# Tạo các thư mục nếu chưa tồn tại
for dir_path in [
    DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_HARD_DIR,
    MODELS_DIR, SVM_MODEL_DIR, CNN_MODEL_DIR, CONFIG_DIR
]:
    os.makedirs(dir_path, exist_ok=True) 