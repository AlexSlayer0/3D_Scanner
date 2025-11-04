from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread("Test_Bild03.png")
results = model.predict(img, imgsz=720)[0]  # single image -> results.boxes

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])   # xyxy coords
    # optional: score = float(box.conf[0]); cls = int(box.cls[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)  # nur Box zeichnen


cv2.imshow("boxes", img)
cv2.waitKey(0)
