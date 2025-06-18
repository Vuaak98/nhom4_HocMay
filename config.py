import os

# --- PATHS ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
BASE_MODELS_DIR = os.path.join(SAVED_MODELS_DIR, 'base_models')
ENSEMBLES_DIR = os.path.join(SAVED_MODELS_DIR, 'ensembles')
PROCESSORS_DIR = os.path.join(SAVED_MODELS_DIR, 'processors')
RULES_PATH = os.path.join(ROOT_DIR, 'config', 'rules.json')
STOPWORDS_PATH = os.path.join(ROOT_DIR, 'config', 'resources', 'vietnamese-stopwords.txt')

# --- DATA PROCESSING PARAMS ---
OUTLIER_QUANTILE = 0.99
DEFAULT_YEAR = 2020

# --- MODEL TRAINING PARAMS ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
VOCAB_SIZE = 5000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
CNN_EPOCHS = 3
CNN_BATCH_SIZE = 32

# --- LABEL MAPPING ---
LABEL_MAP = {'REAL': 0, 'FAKE': 1}

# Đường dẫn gốc của project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dữ liệu
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# File dữ liệu đã xử lý
TRAIN_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'train_cleaned.csv')
TEST_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'test_cleaned.csv')

# Đường dẫn model cơ sở
CNN_MODEL_PATH = os.path.join(BASE_MODELS_DIR, 'cnn_base.h5')
SVM_MODEL_PATH = os.path.join(BASE_MODELS_DIR, 'svm_base.pkl')

# Đường dẫn ensemble
SVM_ENSEMBLE_PATH = os.path.join(ENSEMBLES_DIR, 'svm_for_feature_ensemble.pkl')
STACKING_META_PATH = os.path.join(ENSEMBLES_DIR, 'stacking_meta_classifier.pkl')

# Đường dẫn processor/tokenizer/rule
DATA_PROCESSOR_PATH = os.path.join(PROCESSORS_DIR, 'data_processor.pkl')
TOKENIZER_PATH = os.path.join(PROCESSORS_DIR, 'tokenizer.pkl')
RULE_SYSTEM_PATH = os.path.join(PROCESSORS_DIR, 'rule_system.pkl')

# Đường dẫn đến các file dữ liệu thô
TRAIN_FILE = os.path.join(DATA_RAW_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_RAW_DIR, 'test.csv')

# Đường dẫn đến các file đặc trưng
TRAIN_FEATURES_FILE = os.path.join(FEATURES_DIR, 'train_features.csv')
TEST_FEATURES_FILE = os.path.join(FEATURES_DIR, 'test_features.csv')

# Đường dẫn đến các file hình ảnh
TRAIN_CONFUSION_MATRIX_PATH = os.path.join(VISUALIZATION_DIR, 'train_confusion_matrix.png')
TEST_CONFUSION_MATRIX_PATH = os.path.join(VISUALIZATION_DIR, 'test_confusion_matrix.png')

# Đường dẫn đến file từ dừng và luật
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

# Tạo các thư mục nếu chưa tồn tại
for dir_path in [
    DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, FEATURES_DIR,
    VISUALIZATION_DIR, BASE_MODELS_DIR, CONFIG_DIR,
    SAVED_MODELS_DIR, BASE_MODELS_DIR, ENSEMBLES_DIR, PROCESSORS_DIR
]:
    os.makedirs(dir_path, exist_ok=True) 