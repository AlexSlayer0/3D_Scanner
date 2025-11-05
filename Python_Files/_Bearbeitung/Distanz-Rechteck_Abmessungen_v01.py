import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# MiDaS Modell laden
model_type = "DPT_Large"  # Alternativen: "MiDaS_small", "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Transformation vorbereiten
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

def estimate_depth(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    return prediction.cpu().numpy()

# Rechteckerkennung
def find_largest_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = None
    max_area = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > max_area:
                largest = approx
                max_area = area
    return largest

# Kalibrierung
KNOWN_DISTANCE_CM = 50  # Distanz des Referenzobjekts
known_depth_value = None
depth_to_cm_factor = None

# Kamera starten
cap = cv2.VideoCapture(0)
print("Halte ein Rechteck in 50 cm Entfernung zur Kamera und drücke 'c' zur Kalibrierung.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    depth_map = estimate_depth(frame)
    rectangle = find_largest_rectangle(frame)

    if rectangle is not None:
        rect = rectangle.reshape(4, 2)

        # Oberkante und Unterkante
        top_width_px = np.linalg.norm(rect[0] - rect[1])
        bottom_width_px = np.linalg.norm(rect[2] - rect[3])

        # Mittelpunkt berechnen
        cx = int(np.mean(rect[:, 0]))
        cy = int(np.mean(rect[:, 1]))

        # Tiefe aus Tiefenkarte am Mittelpunkt
        depth_value = depth_map[cy, cx]

        if depth_to_cm_factor:
            estimated_distance_cm = depth_value * depth_to_cm_factor
            width_top_cm = top_width_px * (estimated_distance_cm / KNOWN_DISTANCE_CM)
            width_bottom_cm = bottom_width_px * (estimated_distance_cm / KNOWN_DISTANCE_CM)

            # Anzeige im Bild
            text = f"Tiefe: {estimated_distance_cm:.1f}cm"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Breite oben: {width_top_cm:.1f}cm", tuple(rect[0].astype(int) + np.array([0, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Breite unten: {width_bottom_cm:.1f}cm", tuple(rect[2].astype(int) + np.array([0, 20])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Rechteck zeichnen
        cv2.polylines(frame, [rectangle], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Rechteck & Tiefe", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("c") and rectangle is not None:
        rect = rectangle.reshape(4, 2)
        cx = int(np.mean(rect[:, 0]))
        cy = int(np.mean(rect[:, 1]))
        known_depth_value = depth_map[cy, cx]
        depth_to_cm_factor = KNOWN_DISTANCE_CM / known_depth_value
        print(f"Kalibriert: MiDaS-Tiefenwert {known_depth_value:.4f} → {depth_to_cm_factor:.2f} cm/Faktor")

cap.release()
cv2.destroyAllWindows()
