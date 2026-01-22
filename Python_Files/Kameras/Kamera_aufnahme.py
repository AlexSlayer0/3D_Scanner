import cv2
import os
from datetime import datetime

# Ordner anlegen
ordner = "camera_photos"
os.makedirs(ordner, exist_ok=True)

# USB-Kamera öffnen (0 = erste Kamera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden")

# Ein Bild lesen
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Kein Bild erhalten")

# Dateiname mit Zeitstempel
filename = datetime.now().strftime("foto_%Y%m%d_%H%M%S.jpg")
pfad = os.path.join(ordner, filename)

# Bild speichern
cv2.imwrite(pfad, frame)

print(f"Foto gespeichert unter: {pfad}")
