import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# モデルの読み込み
model = YOLO("yolov8n.pt")

# ページの基本設定
st.set_page_config(
    page_title="📸 物体検出アプリ",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="auto"
)

# CSSスタイル調整
st.markdown("""
    <style>
    .main {
        background-color: #f9f4ef;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #6c5ce7;
        font-size: 36px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 物体検出アプリ")

# ファイルアップローダー
uploaded_file = st.file_uploader("📷 画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # YOLO推論
    results = model(img)
    boxes = results[0].boxes
    names = results[0].names

    # 手動で枠とラベル（％）を描画
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names[cls_id]

        # 表示用のラベル（信頼度をパーセントで）
        text = f"{label} {conf * 100:.1f}%"

        # 四角枠とテキストを描画
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # 結果表示
    st.image(img, caption="🔍 検出結果（信頼度：パーセント表示）", channels="BGR")

    # オブジェクト一覧
    st.subheader("📋 検出されたオブジェクト")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        st.write(f"- {label}（信頼度: {conf * 100:.1f}％）")
