import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_MAP = {
    0:  (102, 204, 255),  # road (sky)
    1:  (0, 0, 255),  # sidewalk (blue)
    8:  (35, 142, 107),  # vegetation (green)
    11: (255, 0, 0),   # People (red)йй
    13: (204, 204, 0),     # Car (yellow)
}

def load_segformer_model():
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model

def processor_segmentation(frame:np.ndarray, processor, model) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    inputs = processor(images = pil_image, return_tensors = "pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size = frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    mask = np.zeros_like(frame, dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        mask[seg_map == class_id] = color
    return cv2.addWeighted(frame, 0.6, mask, 0.4, 0)

if __name__ == "__main__":
    processor, model = load_segformer_model()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = processor_segmentation(frame, processor, model)
        cv2.imshow("Real-Time Segmentation", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()