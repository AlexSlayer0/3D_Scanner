import cv2
import mediapipe as mp
import math

# Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Kalibrierwerte
KNOWN_DISTANCE_CM = 20  # bekannte Entfernung bei Kalibrierung (z.B. 20 cm)
KNOWN_WIDTH_PX = None   # wird während Kalibrierung gesetzt

def euclidean_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

# Kamera starten
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    print("Drücke 'c' zum Kalibrieren bei 20cm Abstand")
    print("Drücke 'q' zum Beenden")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # z. B. Abstand zwischen Daumenbasis (0) und kleinem Finger (17)
                x1 = int(hand_landmarks.landmark[0].x * w)
                y1 = int(hand_landmarks.landmark[0].y * h)
                x2 = int(hand_landmarks.landmark[17].x * w)
                y2 = int(hand_landmarks.landmark[17].y * h)

                current_width_px = euclidean_distance((x1, y1), (x2, y2))

                if KNOWN_WIDTH_PX:
                    # Verhältnis zur bekannten Breite berechnen
                    distance_cm = (KNOWN_DISTANCE_CM * KNOWN_WIDTH_PX) / current_width_px
                    cv2.putText(frame, f"Abstand: {distance_cm:.1f} cm", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Zeichne Hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Abstandsmessung", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('c'):
            if result.multi_hand_landmarks:
                KNOWN_WIDTH_PX = current_width_px
                print(f"Kalibriert mit Breite: {KNOWN_WIDTH_PX:.2f} Pixel @ {KNOWN_DISTANCE_CM} cm")

cap.release()
cv2.destroyAllWindows()
