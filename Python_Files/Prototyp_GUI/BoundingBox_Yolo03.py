# BoundingBox_Yolo03.py
from ultralytics import YOLO
import cv2

# --- Modell einmal laden ---
model = YOLO("yolo12m.pt")  # mittlere Version

def capture_frame(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

def get_boxes_and_dimensions(frame, imgsz=1280):
    results = model.predict(frame, imgsz=imgsz)[0]
    boxes_info = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        boxes_info.append({"box": (x1, y1, x2, y2), "width": width, "height": height})
    return boxes_info

def draw_boxes(frame, boxes_info):
    for info in boxes_info:
        x1, y1, x2, y2 = info["box"]
        width = info["width"]
        height = info["height"]

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Breite oben mittig
        width_text = f"{width}px"
        (w_text_width, w_text_height), _ = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, width_text, (x1 + (width - w_text_width) // 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Höhe links mittig
        height_text = f"{height}px"
        (h_text_width, h_text_height), _ = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, height_text, (x1 - h_text_width, y1 + (height + h_text_height) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

if __name__ == "__main__":
    frame = capture_frame()
    if frame is None:
        #print("Fehler: Kein Bild von der Kamera")
        exit()

    boxes_info = get_boxes_and_dimensions(frame)
    frame = draw_boxes(frame, boxes_info)

    cv2.imshow("Boxes", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
