import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re

from models.cnn_model import train_cnn, extract_cnn_features
from models.svm_model import train_svm
from ensembles.voting_ensemble import VotingEnsemble
from ensembles.feature_ensemble import FeatureEnsemble
from ensembles.stacking_ensemble import StackingEnsemble

def main():
    # --- SETUP: TẠO CÁC THƯ MỤC LƯU TRỮ ---
    print("--- SETUP: Ensuring save directories exist ---")
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # src/
    ROOT_DIR = os.path.dirname(PROJECT_ROOT)  # lên 1 cấp: FakeNewsDetector/FakeNewsDetector
    BASE_MODELS_PATH = os.path.join(ROOT_DIR, 'saved_models', 'base_models')
    ENSEMBLE_MODELS_PATH = os.path.join(ROOT_DIR, 'saved_models', 'ensembles')
    PROCESSORS_PATH = os.path.join(ROOT_DIR, 'saved_models', 'processors')
    VISUALIZATION_PATH = os.path.join(ROOT_DIR, 'data', 'visualization')
    os.makedirs(BASE_MODELS_PATH, exist_ok=True)
    os.makedirs(ENSEMBLE_MODELS_PATH, exist_ok=True)
    os.makedirs(PROCESSORS_PATH, exist_ok=True)
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    print(f"✅ Model save directories checked/created.")
    print(f"✅ Visualization save directory checked/created at '{VISUALIZATION_PATH}'")
    # ----------------------------------------------

    # 1. Load data
    train_df = pd.read_csv('data/features/train_features.csv')
    test_df = pd.read_csv('data/features/test_features.csv')

    # 2. Tokenize + padding cho cleaned_message (xử lý NaN và ép kiểu str)
    all_texts = pd.concat([
        train_df['cleaned_message'].fillna('').astype(str),
        test_df['cleaned_message'].fillna('').astype(str)
    ])
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(all_texts)
    max_length = 100  # hoặc chọn giá trị phù hợp với dữ liệu của bạn

    X_text_train_full = pad_sequences(
        tokenizer.texts_to_sequences(train_df['cleaned_message'].fillna('').astype(str)),
        maxlen=max_length, padding='post')
    X_text_test = pad_sequences(
        tokenizer.texts_to_sequences(test_df['cleaned_message'].fillna('').astype(str)),
        maxlen=max_length, padding='post')

    # Đặc trưng số cho SVM: chỉ giữ lại các cột số và thay NaN bằng 0
    non_feature_cols = ['id', 'user_name', 'post_message', 'timestamp_post', 'label', 'cleaned_message']
    X_features_full = train_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y_full = train_df['label']

    # Chia train/val từ train_df
    X_text_train, X_text_val, X_features_train, X_features_val, y_train, y_val = train_test_split(
        X_text_train_full, X_features_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # Dữ liệu test
    X_features_test = test_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    X_features_test = X_features_test.reindex(columns=X_features_full.columns, fill_value=0)
    y_test = test_df['label']
    y_test = np.array(y_test).ravel()

    # 2. Train base models
    vocab_size = min(5000, len(tokenizer.word_index) + 1)
    embedding_dim = 128
    cnn_base_model = train_cnn(X_text_train, y_train, vocab_size, embedding_dim, max_length, epochs=3, batch_size=32)
    svm_base_model = train_svm(X_features_train, y_train)

    print("\n--- 2. Training Base Models ---")
    print("...Training CNN...")
    print("...Saving CNN base model...")
    cnn_model_path = os.path.join(BASE_MODELS_PATH, 'cnn_base.h5')
    cnn_base_model.save(cnn_model_path)
    print(f"✅ CNN model saved to {cnn_model_path}")

    print("...Training SVM...")
    print("...Saving SVM base model...")
    svm_model_path = os.path.join(BASE_MODELS_PATH, 'svm_base.pkl')
    joblib.dump(svm_base_model, svm_model_path)
    print(f"✅ SVM model saved to {svm_model_path}")

    print("...Saving Tokenizer...")
    tokenizer_path = os.path.join(PROCESSORS_PATH, 'tokenizer.pkl')
    joblib.dump(tokenizer, tokenizer_path)
    print(f"✅ Tokenizer saved to {tokenizer_path}")

    # 3. Build and evaluate ensembles
    results = {}

    # Voting
    voting_model = VotingEnsemble(cnn_base_model, svm_base_model)
    y_pred_proba_voting = voting_model.predict_proba(X_text_test, X_features_test)
    y_pred_voting = (y_pred_proba_voting > 0.5).astype(int).ravel()
    results['Voting'] = classification_report(y_test, y_pred_voting, output_dict=True)
    plot_and_save_confusion_matrix(y_test, y_pred_voting, 'Voting Ensemble', VISUALIZATION_PATH)

    # CNN -> SVM
    print("\n--- Evaluating Ensemble B: CNN -> SVM ---")
    cnn_features_train = extract_cnn_features(cnn_base_model, X_text_train)
    combined_features_train = np.concatenate([cnn_features_train, X_features_train], axis=1)
    svm_for_feature_ens = train_svm(combined_features_train, y_train)
    feature_model = FeatureEnsemble(cnn_base_model, svm_for_feature_ens)
    print("...Saving SVM model for Feature Ensemble...")
    svm_feature_ens_path = os.path.join(ENSEMBLE_MODELS_PATH, 'svm_for_feature_ensemble.pkl')
    joblib.dump(svm_for_feature_ens, svm_feature_ens_path)
    print(f"✅ SVM for Feature Ensemble saved to {svm_feature_ens_path}")
    y_pred_proba_feature = feature_model.predict_proba(X_text_test, X_features_test)
    y_pred_feature = (y_pred_proba_feature > 0.5).astype(int).ravel()
    results['CNN -> SVM'] = classification_report(y_test, y_pred_feature, output_dict=True)
    plot_and_save_confusion_matrix(y_test, y_pred_feature, 'CNN -> SVM Ensemble', VISUALIZATION_PATH)

    # Stacking
    stacking_model = StackingEnsemble(cnn_base_model, svm_base_model)
    stacking_model.fit(X_text_val, X_features_val, y_val)
    print("...Saving Stacking meta-classifier...")
    stacking_meta_path = os.path.join(ENSEMBLE_MODELS_PATH, 'stacking_meta_classifier.pkl')
    joblib.dump(stacking_model.meta_classifier, stacking_meta_path)
    print(f"✅ Stacking meta-classifier saved to {stacking_meta_path}")
    y_pred_proba_stacking = stacking_model.predict_proba(X_text_test, X_features_test)
    y_pred_stacking = (y_pred_proba_stacking > 0.5).astype(int).ravel()
    results['Stacking'] = classification_report(y_test, y_pred_stacking, output_dict=True)
    plot_and_save_confusion_matrix(y_test, y_pred_stacking, 'Stacking Ensemble', VISUALIZATION_PATH)

    # 4. Print comparison
    print("\n\n" + "="*20 + " FINAL COMPARISON REPORT " + "="*20)
    comparison_df = pd.DataFrame({
        'Ensemble': [],
        'Precision (Fake)': [],
        'Recall (Fake)': [],
        'F1-Score (Fake)': [],
        'Accuracy': []
    })
    for name, report in results.items():
        comparison_df.loc[len(comparison_df)] = [
            name,
            report['0']['precision'],
            report['0']['recall'],
            report['0']['f1-score'],
            report['accuracy']
        ]
    print(comparison_df.to_string(index=False))

def plot_and_save_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dự đoán Giả (0)', 'Dự đoán Thật (1)'],
                yticklabels=['Thực tế Giả (0)', 'Thực tế Thật (1)'])
    plt.title(f'Ma trận Nhầm lẫn - {model_name}', fontsize=16)
    plt.ylabel('Nhãn Thực tế', fontsize=12)
    plt.xlabel('Nhãn Dự đoán', fontsize=12)
    # Tạo tên file an toàn
    safe_model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name.replace(' ', '_').lower())
    filepath = os.path.join(save_path, f'confusion_matrix_{safe_model_name}.png')
    plt.savefig(filepath)
    print(f"✅ Confusion matrix for {model_name} saved to {filepath}")
    plt.close()

if __name__ == '__main__':
    main() 