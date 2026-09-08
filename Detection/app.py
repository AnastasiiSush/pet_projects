import streamlit as st
import cv2
import numpy as np
from PIL import Image
from road_detector import load_segformer_model, processor_segmentation
from live_detector import load_yolo_model, process_detection

st.set_page_config(page_title="Computer Vision Hub", layout="wide")

@st.cache_resource
def get_yolo():
    return load_yolo_model()
@st.cache_resource
def get_segformer():
    return load_segformer_model()

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Choose a window :",
    [
        "1. FAQ",
        "2. Live detection",
        "3. Road segmentation"
    ]
)

def handle_image_inputs(key_prefix):
    source = st.radio("Source:" ["Upload photo", "Camera"], horizontal=True, key=f"{key_prefix}_src")
    if source == "Upload photo":
        file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"], key=f"{key_prefix}_file")
        if file:
            return Image.open(file).convert("RGB")
        else:
            cam = st.camera_input("Take a picture", key=f"{key_prefix}_cam")
            if cam:
                return Image.open(cam).convert("RGB")
        return None

if page == "1. FAQ":
    st.title("Live detection")
elif page == "2. Live detection":
    st.title("Live detection")
    image = handle_image_inputs("det")

    if image:
        col1, col2 = st.columns(2)
        col1.subheader("Original Image")
        col1.image(image, use_container_width=True)

        with st.spinner("Processing"):
            yolo_model = get_yolo()
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_bgr = process_detection(frame_bgr, yolo_model)
            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        col2.subheader("Result")
        col2.image(processed_rgb, use_container_width=True)
elif page == "3. Road segmentation":
    st.title("Segmentation")
    image = handle_image_inputs("seg")

    if image:
        col1, col2 = st.columns(2)
        col1.subheader("Original Image")
        col1.image(image, use_container_width=True)

        with st.spinner("Processing"):
            processor, seg_model = get_segformer()
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_bgr = processor_segmentation(frame_bgr, processor, seg_model)
            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        col2.subheader("Result")
        col2.image(processed_rgb, use_container_width=True)