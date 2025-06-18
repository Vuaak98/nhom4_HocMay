import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Any, Tuple, List
from sklearn.utils import resample

from src.models.cnn_model import train_cnn, extract_cnn_features
from src.models.svm_model import train_svm
from src.ensembles.voting_ensemble import VotingEnsemble
from src.ensembles.feature_ensemble import FeatureEnsemble
from src.ensembles.stacking_ensemble import StackingEnsemble

# --- 1. Quản lý cấu hình tập trung ---
CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "vocab_size": 5000,
    "max_length": 100,
    "embedding_dim": 128,
    "cnn_epochs": 3,
    "cnn_batch_size": 32,
    "label_map": {'REAL': 0, 'FAKE': 1}
}

def setup_paths() -> Dict[str, str]:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = {
        'root': root_dir,
        'base_models': os.path.join(root_dir, 'saved_models', 'base_models'),
        'ensembles': os.path.join(root_dir, 'saved_models', 'ensembles'),
        'processors': os.path.join(root_dir, 'saved_models', 'processors'),
        'visualization': os.path.join(root_dir, 'data', 'visualization'),
        'features': os.path.join(root_dir, 'data', 'features')
    }
    for key, path in paths.items():
        if key != 'root':
            os.makedirs(path, exist_ok=True)
    return paths

def load_and_prepare_data(feature_path: str, config: Dict) -> Tuple:
    train_df = pd.read_csv(os.path.join(feature_path, 'train_features.csv'))
    test_df = pd.read_csv(os.path.join(feature_path, 'test_features.csv'))
    all_texts = pd.concat([
        train_df['cleaned_message'].fillna('').astype(str),
        test_df['cleaned_message'].fillna('').astype(str)
    ])
    tokenizer = Tokenizer(num_words=config['vocab_size'], oov_token='<OOV>')
    tokenizer.fit_on_texts(all_texts)
    X_text_train_full = pad_sequences(
        tokenizer.texts_to_sequences(train_df['cleaned_message'].fillna('').astype(str)),
        maxlen=config['max_length'], padding='post'
    )
    X_text_test = pad_sequences(
        tokenizer.texts_to_sequences(test_df['cleaned_message'].fillna('').astype(str)),
        maxlen=config['max_length'], padding='post'
    )
    non_feature_cols = ['id', 'user_name', 'post_message', 'timestamp_post', 'label', 'cleaned_message']
    X_features_full = train_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    print("CÁC CỘT KHI TRAIN SVM:", X_features_full.columns.tolist())
    X_features_test = test_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    X_features_test = X_features_test.reindex(columns=X_features_full.columns, fill_value=0)
    y_full = train_df['label']
    y_test = test_df['label'].values.ravel()
    X_text_train, X_text_val, X_features_train, X_features_val, y_train, y_val = train_test_split(
        X_text_train_full, X_features_full, y_full, 
        test_size=config['test_size'], random_state=config['random_state'], stratify=y_full
    )
    return (X_text_train, X_text_val, X_text_test, 
            X_features_train, X_features_val, X_features_test, 
            y_train, y_val, y_test, tokenizer)

def balance_data(X_text, X_features, y):
    import pandas as pd
    from sklearn.utils import resample
    # Chuyển về DataFrame để dễ xử lý
    X_features_df = pd.DataFrame(X_features)
    X_text_df = pd.DataFrame(X_text)
    y_series = pd.Series(y).reset_index(drop=True)
    X_features_df = X_features_df.reset_index(drop=True)
    X_text_df = X_text_df.reset_index(drop=True)
    df = pd.concat([X_text_df, X_features_df, y_series.rename('label')], axis=1)
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]
    if len(df_minority) == 0:
        return X_text, X_features, y  # Không có mẫu FAKE để upsample
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    # Tách lại
    X_text_bal = df_balanced.iloc[:, :X_text.shape[1]].values
    X_features_bal = df_balanced.iloc[:, X_text.shape[1]:-1].values
    y_bal = df_balanced['label'].values
    return X_text_bal, X_features_bal, y_bal

def train_base_models(X_text_train, X_features_train, y_train, tokenizer, paths, config):
    vocab_size = min(config['vocab_size'], len(tokenizer.word_index) + 1)
    cnn_base_model = train_cnn(X_text_train, y_train, vocab_size, config['embedding_dim'], config['max_length'], 
                               epochs=config['cnn_epochs'], batch_size=config['cnn_batch_size'])
    cnn_model_path = os.path.join(paths['base_models'], 'cnn_base.h5')
    cnn_base_model.save(cnn_model_path)
    svm_base_model = train_svm(X_features_train, y_train)
    svm_model_path = os.path.join(paths['base_models'], 'svm_base.pkl')
    joblib.dump(svm_base_model, svm_model_path)
    tokenizer_path = os.path.join(paths['processors'], 'tokenizer.pkl')
    joblib.dump(tokenizer, tokenizer_path)
    return cnn_base_model, svm_base_model

def evaluate_ensemble(model, model_name: str, X_text_test, X_features_test, y_test, save_path: str) -> Dict:
    y_pred_proba = model.predict_proba(X_text_test, X_features_test)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    report = classification_report(y_test, y_pred, output_dict=True)
    safe_model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name.replace(' ', '_').lower())
    filepath = os.path.join(save_path, f'confusion_matrix_{safe_model_name}.png')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dự đoán Thật (0)', 'Dự đoán Giả (1)'],
                yticklabels=['Thực tế Thật (0)', 'Thực tế Giả (1)'])
    plt.title(f'Ma trận Nhầm lẫn - {model_name}', fontsize=16)
    plt.ylabel('Nhãn Thực tế', fontsize=12)
    plt.xlabel('Nhãn Dự đoán', fontsize=12)
    plt.savefig(filepath)
    plt.close()
    return report

def main_workflow():
    paths = setup_paths()
    (X_text_train, X_text_val, X_text_test, 
     X_features_train, X_features_val, X_features_test, 
     y_train, y_val, y_test, tokenizer) = load_and_prepare_data(paths['features'], CONFIG)

    # Lưu danh sách tên cột feature để dùng cho inference
    if hasattr(X_features_train, 'columns'):
        feature_columns = list(X_features_train.columns)
        joblib.dump(feature_columns, os.path.join(paths['processors'], 'feature_columns.pkl'))

    # Thêm bước cân bằng dữ liệu train
    X_text_train, X_features_train, y_train = balance_data(X_text_train, X_features_train, y_train)
    # Ép kiểu nhãn về int để tránh lỗi KeyError
    y_train = np.array(y_train).astype(int)
    y_val = np.array(y_val).astype(int)
    y_test = np.array(y_test).astype(int)
    cnn_base_model, svm_base_model = train_base_models(
        X_text_train, X_features_train, y_train, tokenizer, paths, CONFIG
    )
    all_results = []
    voting_model = VotingEnsemble(cnn_base_model, svm_base_model)
    voting_report = evaluate_ensemble(voting_model, 'Voting', X_text_test, X_features_test, y_test, paths['visualization'])
    all_results.append({'Ensemble': 'Voting', **voting_report})
    cnn_features_train = extract_cnn_features(cnn_base_model, X_text_train)
    combined_features_train = np.concatenate([cnn_features_train, X_features_train], axis=1)
    svm_for_feature_ens = train_svm(combined_features_train, y_train)
    feature_model = FeatureEnsemble(cnn_base_model, svm_for_feature_ens)
    svm_feature_ens_path = os.path.join(paths['ensembles'], 'svm_for_feature_ensemble.pkl')
    joblib.dump(svm_for_feature_ens, svm_feature_ens_path)
    feature_report = evaluate_ensemble(feature_model, 'CNN -> SVM', X_text_test, X_features_test, y_test, paths['visualization'])
    all_results.append({'Ensemble': 'CNN -> SVM', **feature_report})
    stacking_model = StackingEnsemble(cnn_base_model, svm_base_model)
    stacking_model.fit(X_text_val, X_features_val, y_val)
    stacking_meta_path = os.path.join(paths['ensembles'], 'stacking_meta_classifier.pkl')
    joblib.dump(stacking_model.meta_classifier, stacking_meta_path)
    stacking_report = evaluate_ensemble(stacking_model, 'Stacking', X_text_test, X_features_test, y_test, paths['visualization'])
    all_results.append({'Ensemble': 'Stacking', **stacking_report})
    report_data = []
    REAL_LBL = CONFIG['label_map']['REAL']
    FAKE_LBL = CONFIG['label_map']['FAKE']
    for res in all_results:
        print('Keys in classification_report:', res.keys())
        report_data.append({
            'Ensemble': res['Ensemble'],
            'Accuracy': res['accuracy'],
            'Precision (Fake)': res[str(FAKE_LBL)]['precision'],
            'Recall (Fake)': res[str(FAKE_LBL)]['recall'],
            'F1-Score (Fake)': res[str(FAKE_LBL)]['f1-score'],
            'Support (Fake)': res[str(FAKE_LBL)]['support'],
            'Precision (Real)': res[str(REAL_LBL)]['precision'],
            'Recall (Real)': res[str(REAL_LBL)]['recall'],
            'F1-Score (Real)': res[str(REAL_LBL)]['f1-score'],
            'Support (Real)': res[str(REAL_LBL)]['support'],
        })
    comparison_df = pd.DataFrame(report_data)
    print("\n\n" + "="*20 + " FINAL COMPARISON REPORT " + "="*20)
    print(comparison_df.to_string(index=False))
    return comparison_df

if __name__ == '__main__':
    main_workflow() 