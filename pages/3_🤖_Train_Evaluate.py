import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_evaluate import main_workflow as run_training_workflow
from src.train_evaluate import setup_paths

st.title("Giai đoạn 3: Huấn luyện và Đánh giá Mô hình")

model_path = "saved_models/base_models/svm_base.pkl"
has_model = os.path.exists(model_path)

if has_model:
    st.success("Đã có mô hình huấn luyện. Bạn có thể xem kết quả hoặc huấn luyện lại nếu muốn.")
    paths = setup_paths()
    vis_path = paths['visualization']
    comparison_df = None
    if os.path.exists('data/features/train_features.csv') and os.path.exists('data/features/test_features.csv'):
        comparison_df = pd.DataFrame()
        if st.button("Xem lại kết quả lần trước"):
            # Có thể load lại kết quả nếu đã lưu, hoặc chạy lại đánh giá nhanh
            comparison_df = run_training_workflow()
        if st.button("Huấn luyện lại"):
            comparison_df = run_training_workflow()
        if comparison_df is not None and not comparison_df.empty:
            st.subheader("Bảng so sánh hiệu suất các mô hình")
            st.dataframe(comparison_df)
            st.subheader("Ma trận nhầm lẫn chi tiết")
            col1, col2, col3 = st.columns(3)
            with col1:
                voting_img = os.path.join(vis_path, 'confusion_matrix_voting.png')
                if os.path.exists(voting_img):
                    st.image(voting_img, caption='Voting Ensemble')
                else:
                    st.info("Chưa có ảnh ma trận nhầm lẫn Voting. Hãy bấm 'Huấn luyện lại' để tạo.")
            with col2:
                cnnsvm_img = os.path.join(vis_path, 'confusion_matrix_cnn____svm.png')
                if os.path.exists(cnnsvm_img):
                    st.image(cnnsvm_img, caption='CNN -> SVM Ensemble')
                else:
                    st.info("Chưa có ảnh ma trận nhầm lẫn CNN->SVM. Hãy bấm 'Huấn luyện lại' để tạo.")
            with col3:
                stacking_img = os.path.join(vis_path, 'confusion_matrix_stacking.png')
                if os.path.exists(stacking_img):
                    st.image(stacking_img, caption='Stacking Ensemble')
                else:
                    st.info("Chưa có ảnh ma trận nhầm lẫn Stacking. Hãy bấm 'Huấn luyện lại' để tạo.")
else:
    st.warning("Chưa có mô hình. Vui lòng hoàn thành các bước trước và bấm 'Huấn luyện & Đánh giá'.")
    if st.button("Huấn luyện & Đánh giá"):
        comparison_df = run_training_workflow()
        if comparison_df is not None and not comparison_df.empty:
            st.subheader("Bảng so sánh hiệu suất các mô hình")
            st.dataframe(comparison_df)
            st.subheader("Ma trận nhầm lẫn chi tiết")
            paths = setup_paths()
            vis_path = paths['visualization']
            col1, col2, col3 = st.columns(3)
            with col1:
                voting_img = os.path.join(vis_path, 'confusion_matrix_voting.png')
                if os.path.exists(voting_img):
                    st.image(voting_img, caption='Voting Ensemble')
                else:
                    st.info("Chưa có ảnh ma trận nhầm lẫn Voting. Hãy bấm 'Huấn luyện & Đánh giá' để tạo.")
            with col2:
                cnnsvm_img = os.path.join(vis_path, 'confusion_matrix_cnn____svm.png')
                if os.path.exists(cnnsvm_img):
                    st.image(cnnsvm_img, caption='CNN -> SVM Ensemble')
                else:
                    st.info("Chưa có ảnh ma trận nhầm lẫn CNN->SVM. Hãy bấm 'Huấn luyện & Đánh giá' để tạo.")
            with col3:
                stacking_img = os.path.join(vis_path, 'confusion_matrix_stacking.png')
                if os.path.exists(stacking_img):
                    st.image(stacking_img, caption='Stacking Ensemble')
                else:
                    st.info("Chưa có ảnh ma trận nhầm lẫn Stacking. Hãy bấm 'Huấn luyện & Đánh giá' để tạo.") 