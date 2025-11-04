from ultralytics import YOLO
import cv2

model = YOLO("yolo12m.pt")  # mittlere Version

# Kamera öffnen
cap = cv2.VideoCapture(0)  # 0 = Laptopcam, 1 = externe USB-Kamera

ret, frame = cap.read()
cap.release()  # Kamera sofort wieder freigeben -> LED aus

if not ret:
    print("Fehler: Kein Bild erhalten.")
    exit()

results = model.predict(frame, imgsz=1280)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Breite und Höhe berechnen
    width = x2 - x1
    height = y2 - y1

    # Rechteck zeichnen
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Text vorbereiten
    text = f"{width}x{height}px"

    # Text oberhalb der Box anzeigen
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2, cv2.LINE_AA)


# Anzeigen
cv2.imshow("Boxes", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
