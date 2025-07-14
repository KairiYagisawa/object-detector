import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 🌈 ページの基本設定
st.set_page_config(
    page_title="📸 物体検出アプリ",
    page_icon="🐶",
    layout="centered",
)

# 💅 カスタムCSS（背景色、フォント、タイトル色）
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Zen Maru Gothic', sans-serif;
        background-color: #fff7f0;
        color: #3e3e3e;
    }

    h1 {
        text-align: center;
        color: #a55eea;
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background-color: #ffadad;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #ff7575;
    }

    .stFileUploader > div > div {
        background-color: #fff0f5;
        border-radius: 12px;
        padding: 12px;
        border: 2px dashed #ffa5c9;
    }

    .stFileUploader label {
        font-weight: bold;
        color: #d63384;
    }
    </style>
""", unsafe_allow_html=True)

# 🐾 タイトル表示
st.title("物体検出")
st.caption("画像をアップロードすると物体を検出するよ")

# 🔍 モデルの読み込み
model = YOLO("yolov8n.pt")

# 📁 アップロードUI
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

# 🎯 推論処理
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    results = model(img)
    result_img = results[0].plot()

    # 📷 結果画像の表示
    st.image(result_img, caption="🔍 検出結果", channels="BGR")

    # 📋 ラベル一覧の出力
    st.subheader("📌 検出されたオブジェクト")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]
        st.write(f"- {label}（確度: {conf:.2f}）")
