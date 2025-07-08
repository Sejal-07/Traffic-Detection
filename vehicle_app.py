import streamlit as st
import cv2
import numpy as np
from detector import VehicleDetector
from utils import display_image
import tempfile
import os


st.title("🚗 Traffic Detection Web App")
st.markdown("Upload an image to detect and count vehicles using YOLOv8.")


detector = VehicleDetector()


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name

    
    annotated_img, counts = detector.detect_vehicles(img_path)

   
    st.subheader("🔍 Annotated Image")
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detection Output", use_column_width=True)

    
    st.subheader("📊 Vehicle Counts")
    st.write(counts)

    
    is_success, buffer = cv2.imencode(".jpg", annotated_img)
    st.download_button(label="📥 Download Annotated Image", data=buffer.tobytes(), file_name="annotated_output.jpg", mime="image/jpeg")
