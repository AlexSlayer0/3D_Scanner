import cv2

def find_rectangles(image):
    rectangles = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Approximiere Konturen
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 50:  # Mindestfläche, um kleine Artefakte zu ignorieren
                rectangles.append(approx)

    return rectangles

# Kamera öffnen
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden.")
    exit()

print("Drücke 'q' zum Beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rects = find_rectangles(frame)

    # Rechtecke zeichnen
    for rect in rects:
        cv2.polylines(frame, [rect], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Rechteckerkennung", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
