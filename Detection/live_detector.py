import cv2
import numpy as np
from ultralytics import YOLO

HUMAN_CLASSES = [0]
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
ALL_TARGETS = HUMAN_CLASSES + ANIMAL_CLASSES

def load_yolo_model():
    return YOLO("yolov8n.pt")

def process_detection(frame:np.ndarray, model) -> np.ndarray:
    output_frame = frame.copy()
    results = model.predict(output_frame, classes=ALL_TARGETS, conf=0.45, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if cls_id in HUMAN_CLASSES:
            label = f"Human. Confidence: {round(confidence * 100, 2)}%"
            color = (0, 255, 0)
        elif cls_id == 14:
            label = f"Bird {round(confidence * 100, 1)}%"
            color = (0, 255, 0)
        elif cls_id == 15:
            label = f"Cat {round(confidence * 100, 1)}%"
            color = (0, 255, 255)
        elif cls_id == 16:
            label = f"Dog {round(confidence * 100, 1)}%"
            color = (255, 0, 0)
        else:
            animal_type = model.names[cls_id]
            label = f"Animal ({animal_type}): {round(confidence * 100, 1)}%"
            color = (76, 0, 135)

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output_frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    model = load_yolo_model()
    model = load_yolo_model()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame = process_detection(frame, model)
        cv2.imshow("Humans vs Animals Detection", process_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()