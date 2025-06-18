import streamlit as st
import pandas as pd
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import run_data_processing_workflow

st.title("Giai đoạn 1: Chuẩn bị Dữ liệu")

train_cleaned_path = 'data/processed/train_cleaned.csv'
test_cleaned_path = 'data/processed/test_cleaned.csv'

# Kiểm tra file cleaned đã tồn tại chưa
has_cleaned = os.path.exists(train_cleaned_path) and os.path.exists(test_cleaned_path)

if has_cleaned:
    st.success("Đã có dữ liệu đã làm sạch. Bạn có thể xem trước hoặc tải lại nếu muốn.")
    train_cleaned = pd.read_csv(train_cleaned_path)
    test_cleaned = pd.read_csv(test_cleaned_path)
    st.write("Xem trước dữ liệu train đã làm sạch:")
    st.dataframe(train_cleaned.head())
    st.download_button(
        label="Tải xuống train_cleaned.csv",
        data=train_cleaned.to_csv(index=False).encode('utf-8'),
        file_name='train_cleaned.csv',
        mime='text/csv',
    )
    if st.button("Làm lại tiền xử lý từ đầu"):
        has_cleaned = False
        st.session_state.clear()
        st.experimental_rerun()

if not has_cleaned:
    with st.expander("Bước 1: Tải lên tệp train.csv và test.csv", expanded=True):
        uploaded_train_file = st.file_uploader("Chọn tệp train.csv", type=['csv'])
        uploaded_test_file = st.file_uploader("Chọn tệp test.csv", type=['csv'])
        if uploaded_train_file and uploaded_test_file:
            st.success("Đã tải lên thành công 2 tệp!")
            st.session_state['raw_train_df'] = pd.read_csv(uploaded_train_file)
            st.session_state['raw_test_df'] = pd.read_csv(uploaded_test_file)
            st.write("Xem trước dữ liệu train thô:")
            st.dataframe(st.session_state['raw_train_df'].head())
    st.subheader("Bước 2: Chạy Tiền xử lý và Làm sạch Dữ liệu")
    if 'raw_train_df' in st.session_state:
        if st.button("🚀 Bắt đầu Tiền xử lý"):
            with st.spinner("Đang làm sạch và xử lý dữ liệu... Quá trình này có thể mất vài phút."):
                train_cleaned, test_cleaned, processor = run_data_processing_workflow(
                    st.session_state['raw_train_df'], 
                    st.session_state['raw_test_df']
                )
                st.session_state['cleaned_train_df'] = train_cleaned
                st.session_state['cleaned_test_df'] = test_cleaned
                st.session_state['data_processor'] = processor
                os.makedirs('data/processed', exist_ok=True)
                train_cleaned.to_csv(train_cleaned_path, index=False, encoding='utf-8')
                test_cleaned.to_csv(test_cleaned_path, index=False, encoding='utf-8')
                joblib.dump(processor, 'saved_models/processors/data_processor.pkl')
            st.success("Hoàn tất tiền xử lý!")
            st.write("Xem trước dữ liệu train đã làm sạch:")
            st.dataframe(st.session_state['cleaned_train_df'].head())
            st.download_button(
                label="Tải xuống train_cleaned.csv",
                data=train_cleaned.to_csv(index=False).encode('utf-8'),
                file_name='train_cleaned.csv',
                mime='text/csv',
            )
else:
    st.info("Vui lòng tải lên dữ liệu và chạy tiền xử lý để tiếp tục.")
