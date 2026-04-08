import cv2

cap = cv2.VideoCapture(2)

# Prüfen ob Kamera geöffnet
if not cap.isOpened():
    raise RuntimeError("Kamera nicht gefunden")

# Filter/Parameter setzen (0-1 = Normal)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Helligkeit
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Kontrast
cap.set(cv2.CAP_PROP_SATURATION, 0.5)  # Sättigung
cap.set(cv2.CAP_PROP_GAIN, 0.0)        # Verstärkung
cap.set(cv2.CAP_PROP_EXPOSURE, -6)     # Belichtung (je nach Kamera verschieden)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)  # Weißabgleich Temperatur (manchmal nötig)
