import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime, date
import sys
import os

# Đảm bảo import đầy đủ các class custom trước khi load pickle
from src.data_processing import DataProcessor, InteractionProcessor, TimestampProcessor, TextProcessor
from src.train_rules import RuleSystem
from src.ensembles.voting_ensemble import VotingEnsemble
from src.ensembles.feature_ensemble import FeatureEnsemble
from src.ensembles.stacking_ensemble import StackingEnsemble
from src.train_evaluate import setup_paths

st.title("Giai đoạn 4: Dự đoán Trực tiếp")

paths = setup_paths()
model_files = [
    os.path.join(paths['base_models'], 'cnn_base.h5'),
    os.path.join(paths['base_models'], 'svm_base.pkl'),
    os.path.join(paths['ensembles'], 'svm_for_feature_ensemble.pkl'),
    os.path.join(paths['ensembles'], 'stacking_meta_classifier.pkl'),
    os.path.join(paths['processors'], 'data_processor.pkl'),
    os.path.join(paths['processors'], 'rule_system.pkl'),
    os.path.join(paths['processors'], 'tokenizer.pkl'),
]

has_all_models = all(os.path.exists(f) for f in model_files)

if has_all_models:
    st.success("Tất cả mô hình và processor đã được tải thành công! Bạn có thể dự đoán ngay.")
    @st.cache_resource
    def load_all_models_and_processors():
        data_processor = joblib.load(os.path.join(paths['processors'], 'data_processor.pkl'))
        rule_system = joblib.load(os.path.join(paths['processors'], 'rule_system.pkl'))
        tokenizer = joblib.load(os.path.join(paths['processors'], 'tokenizer.pkl'))
        cnn_model = tf.keras.models.load_model(os.path.join(paths['base_models'], 'cnn_base.h5'))
        svm_model = joblib.load(os.path.join(paths['base_models'], 'svm_base.pkl'))
        svm_for_feature_ens = joblib.load(os.path.join(paths['ensembles'], 'svm_for_feature_ensemble.pkl'))
        stacking_meta_classifier = joblib.load(os.path.join(paths['ensembles'], 'stacking_meta_classifier.pkl'))
        voting_model = VotingEnsemble(cnn_model, svm_model)
        feature_model = FeatureEnsemble(cnn_model, svm_for_feature_ens)
        stacking_model = StackingEnsemble(cnn_model, svm_model)
        stacking_model.meta_classifier = stacking_meta_classifier
        return {
            "data_processor": data_processor,
            "rule_system": rule_system,
            "tokenizer": tokenizer,
            "voting_model": voting_model,
            "feature_model": feature_model,
            "stacking_model": stacking_model
        }

    def predict_single(text, likes, comments, shares, timestamp_post, components):
        input_data = pd.DataFrame([{
            'post_message': text,
            'num_like_post': likes,
            'num_comment_post': comments,
            'num_share_post': shares,
            'timestamp_post': timestamp_post
        }])
        cleaned_df = components['data_processor'].transform(input_data)
        classified_df = components['rule_system'].classify_difficulty(cleaned_df)
        if classified_df['case_difficulty'].iloc[0] == 'Tin Thật Dễ':
            return {
                'Voting': {'label': 'Real', 'confidence': 1.0, 'source': 'Rule Filter'},
                'CNN -> SVM': {'label': 'Real', 'confidence': 1.0, 'source': 'Rule Filter'},
                'Stacking': {'label': 'Real', 'confidence': 1.0, 'source': 'Rule Filter'}
            }
        features_df = components['rule_system'].extract_features(classified_df)
        X_text_seq = components['tokenizer'].texts_to_sequences(features_df['cleaned_message'])
        X_text_padded = tf.keras.preprocessing.sequence.pad_sequences(X_text_seq, maxlen=100)
        non_feature_cols = ['id', 'user_name', 'post_message', 'timestamp_post', 'label', 'cleaned_message', 'case_difficulty']
        X_features = features_df.drop(columns=non_feature_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        try:
            feature_columns = joblib.load(os.path.join(paths['processors'], 'feature_columns.pkl'))
            if feature_columns is not None:
                X_features = X_features.reindex(columns=feature_columns, fill_value=0)
                st.write("CÁC CỘT MODEL EXPECT:", feature_columns)
                st.write("CÁC CỘT KHI PREDICT (SAU REINDEX):", X_features.columns.tolist())
        except Exception as e:
            # Không cần warning vì feature columns là optional
            pass
        p_voting = components['voting_model'].predict_proba(X_text_padded, X_features)[0]
        p_feature = components['feature_model'].predict_proba(X_text_padded, X_features)[0]
        p_stacking = components['stacking_model'].predict_proba(X_text_padded, X_features)[0]
        return {
            'Voting': {
                'label': 'Fake' if p_voting > 0.5 else 'Real',
                'proba_fake': float(p_voting),
                'proba_real': float(1 - p_voting),
                'confidence': float(p_voting) if p_voting > 0.5 else float(1 - p_voting)
            },
            'CNN -> SVM': {
                'label': 'Fake' if p_feature > 0.5 else 'Real',
                'proba_fake': float(p_feature),
                'proba_real': float(1 - p_feature),
                'confidence': float(p_feature) if p_feature > 0.5 else float(1 - p_feature)
            },
            'Stacking': {
                'label': 'Fake' if p_stacking > 0.5 else 'Real',
                'proba_fake': float(p_stacking),
                'proba_real': float(1 - p_stacking),
                'confidence': float(p_stacking) if p_stacking > 0.5 else float(1 - p_stacking)
            }
        }

    components = load_all_models_and_processors()
    with st.form("inference_form"):
        text_input = st.text_area("Nội dung bài đăng:", height=150, placeholder="Dán nội dung tin tức vào đây...")
        col1, col2, col3 = st.columns(3)
        with col1:
            likes = st.number_input("Số lượt thích", min_value=0, value=100)
        with col2:
            comments = st.number_input("Số bình luận", min_value=0, value=10)
        with col3:
            shares = st.number_input("Số lượt chia sẻ", min_value=0, value=5)
        date_col, time_col = st.columns(2)
        with date_col:
            post_date = st.date_input("Ngày đăng", value=date.today())
        with time_col:
            post_time = st.time_input("Giờ đăng", value=datetime.now().time())
        submitted = st.form_submit_button("🔍 Phân loại")
    if submitted:
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung bài đăng.")
        else:
            timestamp_post = datetime.combine(post_date, post_time).strftime('%Y-%m-%d %H:%M:%S')
            with st.spinner("Đang phân tích..."):
                results = predict_single(text_input, likes, comments, shares, timestamp_post, components)
            st.subheader("Kết quả Phân loại")
            col_a, col_b, col_c = st.columns(3)
            def display_result(column, model_name, result):
                proba_fake = result['proba_fake']
                proba_real = result['proba_real']
                # Độ tin cậy luôn là xác suất Real (class 0)
                confidence = proba_real
                label = 'Real' if proba_real > 0.5 else 'Fake'
                color = "green" if label == "Real" else "red"
                column.markdown(f"**{model_name}**")
                column.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)
                column.write(f"Xác suất là tin giả (Fake): {proba_fake:.2%}")
                column.write(f"Xác suất là tin thật (Real): {proba_real:.2%}")
                if 'source' in result:
                    column.info(f"Nguồn: {result['source']}")
                column.progress(confidence, text=f"Độ tin cậy (tin thật): {confidence:.2%}")
            display_result(col_a, "Voting Ensemble", results['Voting'])
            display_result(col_b, "CNN -> SVM", results['CNN -> SVM'])
            display_result(col_c, "Stacking Ensemble", results['Stacking'])
else:
    st.warning("Chưa có đủ mô hình và processor. Vui lòng hoàn thành các bước trước để sử dụng dự đoán trực tiếp.")
