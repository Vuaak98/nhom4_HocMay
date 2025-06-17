import os

# Đường dẫn gốc của dự án
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến các thư mục dữ liệu
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')  # Dữ liệu thô
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')  # Dữ liệu đã xử lý
DATA_FEATURES_DIR = os.path.join(DATA_DIR, 'features')  # Đặc trưng
DATA_VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization')  # Hình ảnh

# Đường dẫn đến các file dữ liệu thô
TRAIN_FILE = os.path.join(DATA_RAW_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_RAW_DIR, 'test.csv')

# Đường dẫn đến các file dữ liệu đã xử lý
TRAIN_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'train_cleaned.csv')
TEST_CLEANED_FILE = os.path.join(DATA_PROCESSED_DIR, 'test_cleaned.csv')
DATA_PROCESSOR_FILE = os.path.join(DATA_PROCESSED_DIR, 'data_processor.pkl')

# Đường dẫn đến các file đặc trưng
TRAIN_FEATURES_FILE = os.path.join(DATA_FEATURES_DIR, 'train_features.csv')
TEST_FEATURES_FILE = os.path.join(DATA_FEATURES_DIR, 'test_features.csv')

# Đường dẫn đến các file hình ảnh
TRAIN_CONFUSION_MATRIX_PATH = os.path.join(DATA_VISUALIZATION_DIR, 'train_confusion_matrix.png')
TEST_CONFUSION_MATRIX_PATH = os.path.join(DATA_VISUALIZATION_DIR, 'test_confusion_matrix.png')

# Đường dẫn đến file từ dừng và luật
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
STOPWORDS_PATH = os.path.join(CONFIG_DIR, 'resources', 'vietnamese-stopwords.txt')
RULES_PATH = os.path.join(CONFIG_DIR, 'rules.json')

# Đường dẫn đến các thư mục mô hình
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RULE_SYSTEM_MODEL_PATH = os.path.join(MODELS_DIR, 'rule_system.pkl')

# Thư mục lưu mô hình huấn luyện và ensemble
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
BASE_MODELS_DIR = os.path.join(SAVED_MODELS_DIR, 'base_models')
ENSEMBLES_DIR = os.path.join(SAVED_MODELS_DIR, 'ensembles')
PROCESSORS_DIR = os.path.join(SAVED_MODELS_DIR, 'processors')

# Tạo các thư mục nếu chưa tồn tại
for dir_path in [
    DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_FEATURES_DIR,
    DATA_VISUALIZATION_DIR, MODELS_DIR, CONFIG_DIR,
    SAVED_MODELS_DIR, BASE_MODELS_DIR, ENSEMBLES_DIR, PROCESSORS_DIR
]:
    os.makedirs(dir_path, exist_ok=True) 