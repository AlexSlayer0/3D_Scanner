# Image_verarbeitung07_autoDetect.py
# ----------------
# Automatische Objekterkennung mit adaptiver Kantenanalyse (Paket/Artikel)
# ----------------
import cv2
import numpy as np
from multiprocessing import shared_memory

# ---------------- PRODUCER: Kameraaufnahme ----------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Kamera-Fehler")

# Shared Memory anlegen
shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
shm_array[:] = frame[:]

print(f"[Producer] Shared Memory erstellt: {shm.name}")
print(f"[Producer] Frame-Shape: {frame.shape}, Dtype: {frame.dtype}")

# ---------------- CONSUMER: Bildverarbeitung ----------------
image = np.ascontiguousarray(shm_array)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Vorverarbeitung – leichte Glättung, um Rauschen zu reduzieren
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------- Variante 1: Sobel + Threshold ----------
grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
grad = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
_, sobel_thresh = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)

# ---------- Variante 2: Adaptive Threshold ----------
adaptive_thresh = cv2.adaptiveThreshold(
    gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 21, 5
)

# ---------- Beste Variante automatisch auswählen ----------
# Wir bewerten die Bildvarianz – mehr Konturen = besserer Kontrast
def contour_score(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)
    return len(contours) * total_area

score_sobel = contour_score(sobel_thresh)
score_adapt = contour_score(adaptive_thresh)
best_mask = sobel_thresh if score_sobel > score_adapt else adaptive_thresh
method = "Sobel" if best_mask is sobel_thresh else "Adaptive"

print(f"[Analyzer] Beste Methode: {method} (Sobel={score_sobel:.0f}, Adaptive={score_adapt:.0f})")

# ---------- Rand entfernen ----------
border = 10
mask = np.zeros_like(best_mask)
cv2.rectangle(mask, (border, border),
              (best_mask.shape[1] - border, best_mask.shape[0] - border),
              255, thickness=-1)
best_mask = cv2.bitwise_and(best_mask, mask)

# ---------- Konturen finden ----------
contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Nur valide Konturen (nicht am Rand)
valid_contours = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if x > border and y > border and (x + w) < (best_mask.shape[1] - border) and (y + h) < (best_mask.shape[0] - border):
        valid_contours.append(c)

cv2.drawContours(image, valid_contours, -1, (0, 0, 255), 1)

# ---------- Größtes Objekt (Artikel/Paket) ----------
max_area = 0
main_contour = None
for c in valid_contours:
    area = cv2.contourArea(c)
    if area > max_area:
        max_area = area
        main_contour = c

if main_contour is not None:
    x, y, width, height = cv2.boundingRect(main_contour)
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Text oben (Breite)
    text_top = f"{width}px"
    text_size_top, _ = cv2.getTextSize(text_top, font, font_scale, thickness)
    text_x_top = x + (width - text_size_top[0]) // 2
    text_y_top = max(20, y - 10)
    cv2.putText(image, text_top, (text_x_top, text_y_top), font, font_scale, (0, 255, 0), thickness)

    # Text links (Höhe)
    text_left = f"{height}px"
    text_size_left, _ = cv2.getTextSize(text_left, font, font_scale, thickness)
    text_x_left = max(5, x - text_size_left[0] - 10)
    text_y_left = y + (height + text_size_left[1]) // 2
    cv2.putText(image, text_left, (text_x_left, text_y_left), font, font_scale, (0, 255, 0), thickness)

    print(f"[Consumer] Hauptobjekt erkannt: Breite={width}px, Höhe={height}px")
else:
    print("[Consumer] Kein Hauptobjekt gefunden.")

# ---------- Anzeige ----------
cv2.imshow("Verarbeitetes Bild (Auto Detection, Rand ignoriert)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- Aufräumen ----------
shm.close()
shm.unlink()
print("[System] Shared Memory freigegeben.")
