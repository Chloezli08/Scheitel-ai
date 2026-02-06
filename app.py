import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Scheitelgemach", layout="wide")
st.title("🧢 Scheitelgemach Zürich")
st.subheader("Virtuelle Perückenanprobe mit KI")

col1, col2 = st.columns(2)
with col1:
    customer_file = st.file_uploader("📸 Ihr Foto", type=['jpg', 'jpeg', 'png'])
with col2:
    wig_file = st.file_uploader("🎀 Perückenbild", type=['jpg', 'jpeg', 'png'])

if customer_file and wig_file:
    if st.button("🚀 ANPROBE STARTEN", use_container_width=True, type="primary"):
        with st.spinner("🤖 Verarbeite..."):
            customer_img = Image.open(customer_file).convert('RGB')
            wig_img = Image.open(wig_file).convert('RGB')
            customer_cv = cv2.cvtColor(np.array(customer_img), cv2.COLOR_RGB2BGR)
            wig_cv = cv2.cvtColor(np.array(wig_img), cv2.COLOR_RGB2BGR)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(customer_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                new_w = int(w * 1.4)
                new_h = int(h * 1.3)
                wig_resized = cv2.resize(wig_cv, (new_w, new_h))
                wig_x = max(0, min(x - int((new_w - w) / 2), customer_cv.shape[1] - new_w))
                wig_y = max(0, min(y - int(new_h * 0.3), customer_cv.shape[0] - new_h))
                result = customer_cv.copy()
                roi = result[wig_y:wig_y+new_h, wig_x:wig_x+new_w]
                blended = cv2.addWeighted(wig_resized, 0.8, roi, 0.2, 0)
                result[wig_y:wig_y+new_h, wig_x:wig_x+new_w] = blended
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                result_img = Image.fromarray(result_rgb)
                st.success("✅ Fertig!")
                st.image(result_img)
                buf = BytesIO()
                result_img.save(buf, format='JPEG', quality=95)
                buf.seek(0)
                st.download_button("📥 Download", buf.getvalue(), "scheitelgemach.jpg", "image/jpeg", use_container_width=True)
            else:
                st.error("❌ Gesicht nicht erkannt!")
