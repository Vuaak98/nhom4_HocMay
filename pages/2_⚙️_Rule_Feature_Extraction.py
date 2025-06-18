import streamlit as st
import sys
import os
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_rules import run_rules_workflow

st.title("Giai đoạn 2: Áp dụng Luật và Trích xuất Đặc trưng")

train_features_path = 'data/features/train_features.csv'
test_features_path = 'data/features/test_features.csv'

has_features = os.path.exists(train_features_path) and os.path.exists(test_features_path)

if has_features:
    st.success("Đã có file đặc trưng. Bạn có thể xem trước hoặc tiếp tục.")
    train_features = pd.read_csv(train_features_path)
    st.write("Xem trước dữ liệu đặc trưng của tập train:")
    st.dataframe(train_features.head())
    st.download_button(
        label="Tải xuống train_features.csv",
        data=train_features.to_csv(index=False).encode('utf-8'),
        file_name='train_features.csv',
        mime='text/csv',
    )
    if st.button("Trích xuất lại đặc trưng từ đầu"):
        has_features = False
        st.session_state.clear()
        st.experimental_rerun()

if not has_features:
    if not (os.path.exists('data/processed/train_cleaned.csv') and os.path.exists('data/processed/test_cleaned.csv')):
        st.error("Chưa có dữ liệu đã làm sạch. Vui lòng hoàn thành Giai đoạn 1 trước.")
    else:
        train_cleaned = pd.read_csv('data/processed/train_cleaned.csv')
        test_cleaned = pd.read_csv('data/processed/test_cleaned.csv')
        if st.button("🚀 Bắt đầu Trích xuất Đặc trưng"):
            with st.spinner("Đang áp dụng luật và tạo các đặc trưng cho mô hình ML..."):
                (train_features, test_features, rule_system, _, _) = run_rules_workflow(
                    train_cleaned,
                    test_cleaned
                )
                os.makedirs('data/features', exist_ok=True)
                train_features.to_csv(train_features_path, index=False, encoding='utf-8')
                test_features.to_csv(test_features_path, index=False, encoding='utf-8')
                joblib.dump(rule_system, 'saved_models/processors/rule_system.pkl')
            st.success("Hoàn tất trích xuất đặc trưng!")
            st.write("Xem trước dữ liệu đặc trưng của tập train:")
            st.dataframe(train_features.head())
            st.download_button(
                label="Tải xuống train_features.csv",
                data=train_features.to_csv(index=False).encode('utf-8'),
                file_name='train_features.csv',
                mime='text/csv',
            )
else:
    st.info("Vui lòng hoàn thành các bước trước để tiếp tục.")
