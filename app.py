import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
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
    source = st.radio(
        "Source:",
        ["Upload photo", "Camera (Snapshot)", "Live Stream (WebRTC)"],
        horizontal=True,
        key=f"{key_prefix}_src"
    )

    if source == "Upload photo":
        file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"], key=f"{key_prefix}_file")
        if file:
            return "static", Image.open(file).convert("RGB")

    elif source == "Camera (Snapshot)":
        cam = st.camera_input("Take a picture", key=f"{key_prefix}_cam")
        if cam:
            return "static", Image.open(cam).convert("RGB")

    elif source == "Live Stream (WebRTC)":
        return "stream", None

    return None, None

def yolo_video_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    model = get_yolo()
    processed_bgr = process_detection(img_bgr, model)
    return av.VideoFrame.from_ndarray(processed_bgr, format="bgr24")

def segformer_video_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    processor, model = get_segformer()
    processed_bgr = processor_segmentation(img_bgr, processor, model)
    return av.VideoFrame.from_ndarray(processed_bgr, format="bgr24")

if page == "1. FAQ":
    st.title("Live detection")
    st.markdown("""
        * **Live detection:** Детекція людей та тварин за допомогою YOLOv8.
        * **Road segmentation:** Сегментація дорожньої інфраструктури за допомогою SegFormer.
        * **Джерела даних:** Підтримується завантаження фото, фотознімок з камери та живий відеопотік (WebRTC).
        """)

elif page == "2. Live detection":
    st.title("Live detection")
    mode, image_or = handle_image_inputs("det")

    if mode == "static" and image_or:
        col1, col2 = st.columns(2)
        col1.subheader("Original Image")
        col1.image(image_or, use_container_width=True)

        with st.spinner("Processing"):
            yolo_model = get_yolo()
            frame_bgr = cv2.cvtColor(np.array(image_or), cv2.COLOR_RGB2BGR)
            processed_bgr = process_detection(frame_bgr, yolo_model)
            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        col2.subheader("Result")
        col2.image(processed_rgb, use_container_width=True)

    elif mode == "stream":
        st.subheader("Real-Time Camera Stream")
        webrtc_streamer(
            key="yolo-stream",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=yolo_video_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

elif page == "3. Road segmentation":
    st.title("Segmentation")
    mode, image = handle_image_inputs("seg")

    if mode == "static" and image:
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
    elif mode == "stream":
        st.subheader("Real-Time Camera Stream")
        webrtc_streamer(
            key="segformer-stream",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=segformer_video_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )