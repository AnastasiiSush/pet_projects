import cv2
import numpy as np
from ultralytics import YOLO

CAT_ID = 15
CONF_TRESHOLD = 0.65

def export_for_ios():
    model = YOLO("yolov8n.pt")
    model.export(format="coreml")
    print("Файл CoreML успішно створено!")

def load_yolo_model():
    return YOLO("yolov8n.pt")


def process_detection(frame:np.ndarray, model) -> np.ndarray:
    output_frame = frame.copy()
    results = model.predict(output_frame, classes=[CAT_ID], conf=0.45, verbose=False)[0]

    cat_found = False
    best_confidence = 0.0
    cat_box = None

    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        if confidence > best_confidence:
            best_confidence = confidence
            cat_box = (x1,y1,x2,y2)

        ready = confidence >= CONF_TRESHOLD
        color = (0, 255, 0) if ready else (0, 255, 255)
        status_text = "READY TO CATCH!" if ready else "GET CLOSER..."

        label = f"Cat: {round(confidence * 100, 1)}% [{status_text}]"

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(output_frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cat_found = True

    return output_frame, cat_found, best_confidence, cat_box

if __name__ == "__main__":
    export_for_ios()
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # model = load_yolo_model()
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     process_frame, found, conf, _ = process_detection(frame, model)
    #     if found and conf >= CONF_TRESHOLD:
    #         cv2.putText(process_frame, "PRESS [SPACE] TO CATCH!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0), 2)
    #
    #
    #     cv2.imshow("Find the cat", process_frame)
    #
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #     elif key == ord(' '):
    #         print("CATCH!")
    #
    # cap.release()
    # cv2.destroyAllWindows()
