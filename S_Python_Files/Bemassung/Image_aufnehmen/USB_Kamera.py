import cv2

cap = cv2.VideoCapture(1)  # externe Kamera

# Versuche die höchstmögliche Auflösung einzustellen
# Typische maximale Werte (du kannst anpassen)
max_width = 1920
max_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

print("Drücke 'q', um das Programm zu beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Kamerabilds")
        break

    # Bild anzeigen
    cv2.imshow('Live-Kamera', frame)

    # Beenden, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
