🛠️ Detailed Overview
1. 🌐 Computer Vision Web Hub
An interactive multi-modal web application built with Streamlit that processes both uploaded images and live camera input for real-time scene analysis.

Road Semantic Segmentation: Employs SegFormer (nvidia/segformer-b0-finetuned-cityscapes-1024-1024) fine-tuned on the Cityscapes dataset to perform pixel-level classification (identifying roads, sidewalks, pedestrians, vehicles, and vegetation).

Human & Animal Detection: Runs YOLOv8n filtered specifically for humans and diverse animal classes with visual bounding boxes and confidence scoring.

Performance: Uses Streamlit resource caching (@st.cache_resource) to load deep learning models into memory once and optimize execution speed.
