import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang chủ - Fake News Detector",
    page_icon="🏠",
    layout="wide"
)

st.title("Chào mừng đến với Hệ thống Phát hiện Tin giả 📰")

st.markdown("""
Đây là một công cụ toàn diện để chạy và tương tác với pipeline phát hiện tin giả. 
Hệ thống này cho phép bạn thực hiện toàn bộ quy trình từ xử lý dữ liệu thô,
huấn luyện mô hình, đến dự đoán trên các tin tức mới.

**Vui lòng sử dụng thanh điều hướng bên trái để truy cập các chức năng:**
- **🗂️ Data Preparation:** Tải lên và xử lý dữ liệu thô.
- **⚙️ Rule & Feature Extraction:** Áp dụng luật và trích xuất đặc trưng.
- **🤖 Train & Evaluate:** Huấn luyện và đánh giá các mô hình học máy.
- **📰 Live Inference:** Dự đoán tin thật/giả cho một bài đăng mới.
""")

try:
    image = Image.open('assets/pipeline_diagram.png')
    st.image(image, caption='Sơ đồ tổng quan của Pipeline')
except Exception:
    st.warning("Không tìm thấy sơ đồ pipeline. Hãy tạo một file tại 'assets/pipeline_diagram.png'")

st.info("Để bắt đầu, hãy chuẩn bị các tệp `train.csv` và `test.csv` và đi đến trang **Data Preparation**.")
