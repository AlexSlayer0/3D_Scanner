import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import os
import time
import numpy as np

# --- Funktion zum Einlesen von YOLO Ground-Truth Boxen ---
def load_ground_truth(txt_path, img_shape):
    gt_boxes = []
    h, w = img_shape[:2]
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_c, y_c, bw, bh = map(float, parts)
            x = int((x_c - bw/2) * w)
            y = int((y_c - bh/2) * h)
            width = int(bw * w)
            height = int(bh * h)
            gt_boxes.append((x, y, width, height))
    return gt_boxes

# --- Funktion für Inside-Check ---
def inside(gt, det):
    gx, gy, gw, gh = gt
    dx, dy, dw, dh = det
    cx = dx + dw/2
    cy = dy + dh/2
    return (gx <= cx <= gx+gw) and (gy <= cy <= gy+gh)

# --- Funktion: ROI vorverarbeiten für bessere Decodierung ---
def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)
    
    # Schärfen mit Filter
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    
    # Optional: Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    return thresh

total_gt = 0
total_found = 0
total_decoded = 0

model = YOLO("YOLOV8s_Barcode_Detection.pt")
folder = "InventBar"
#folder = "ParcelBar"
images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

# --- Alle Dateien durchlaufen ---
for img_file in images:
    start_time = time.time()
    base_name = os.path.splitext(img_file)[0]
    txt_file = base_name + ".txt"
    img_path = os.path.join(folder, img_file)
    txt_path = os.path.join(folder, txt_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"{img_file}: Bild konnte nicht geladen werden!")
        continue
    if not os.path.exists(txt_path):
        print(f"{img_file}: TXT-Datei nicht gefunden!")
        continue

    gt_boxes = load_ground_truth(txt_path, img.shape)
    results = model.predict(img)

    yolo_boxes = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            yolo_boxes.append((x1, y1, w, h))

    hits = 0
    decoded_hits = 0
    for gt in gt_boxes:
        found = False
        for det in yolo_boxes:
            if inside(gt, det):
                found = True
                x, y, w, h = det
                roi = img[y:y+h, x:x+w]
                preprocessed = preprocess_roi(roi)
                barcodes = decode(preprocessed)
                if barcodes:
                    decoded_hits += 1
                break
        if found:
            hits += 1

    total_gt += len(gt_boxes)
    total_found += hits
    total_decoded += decoded_hits

    elapsed = time.time() - start_time
    print(f"{img_file} | {hits}/{len(gt_boxes)}, decodiert: {decoded_hits} | {elapsed:.2f}s")

if total_gt > 0:
    percent_found = total_found / total_gt * 100
    percent_decoded = total_decoded / total_gt * 100
    print(f"\nGesamt: erkannt {total_found}/{total_gt} ({percent_found:.2f}%), decodiert {total_decoded}/{total_gt} ({percent_decoded:.2f}%)")
else:
    print("Keine Ground-Truth Boxen gefunden.")
