import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Live YOLO Barcode ROI Debugger')
    parser.add_argument('--camera', type=int, default=0, help='Kamera-ID (default: 0)')
    parser.add_argument('--model', type=str, default='YOLOV8s_Barcode_Detection.pt', help='Pfad zum YOLO-Modell')
    parser.add_argument('--conf', type=float, default=0.5, help='Konfidenz-Schwellwert (default: 0.5)')
    parser.add_argument('--padding', type=float, default=0.3, help='Padding-Faktor (default: 0.3)')
    args = parser.parse_args()

    # Modell laden
    if not os.path.exists(args.model):
        print(f"FEHLER: Modell nicht gefunden unter {args.model}")
        return
    model = YOLO(args.model)
    print(f"YOLO-Modell geladen. Konfidenz: {args.conf}, Padding: {args.padding}")

    # Kamera öffnen
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("FEHLER: Kamera konnte nicht geöffnet werden.")
        return

    print("Drücke 'q' zum Beenden, 's' zum Speichern des aktuellen Bildes und der ROIs.")

    # Fenster vorbereiten
    cv2.namedWindow('Live Barcode ROI Debug', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Aktuelle ROI(s)', cv2.WINDOW_NORMAL)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Kamera-Frame konnte nicht gelesen werden.")
            break

        # YOLO benötigt RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, conf=args.conf, verbose=False)

        # Kopie für Anzeige mit Boxen
        display_frame = frame_bgr.copy()
        rois = []  # Liste für extrahierte ROIs (BGR)

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for box in r.boxes:
                # Koordinaten holen (x1, y1, x2, y2)
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                # Padding berechnen
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(box_w * args.padding)
                pad_y = int(box_h * args.padding)

                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame_bgr.shape[1], x2 + pad_x)
                y2_pad = min(frame_bgr.shape[0], y2 + pad_y)

                # ROI aus BGR-Bild extrahieren
                roi = frame_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
                if roi.size == 0:
                    continue
                rois.append(roi)

                # Boxen und Koordinaten ins Bild zeichnen
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # grün: Original YOLO-Box
                cv2.rectangle(display_frame, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 0, 255), 2)  # rot: mit Padding
                cv2.putText(display_frame, f"conf: {box.conf[0]:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # Anzeige der ROIs (falls vorhanden)
        if rois:
            # Einfache Darstellung: alle ROIs in einem Fenster nebeneinander
            # Dazu skalieren wir sie auf eine einheitliche Höhe (z.B. 200px) und hängen sie horizontal an
            vis_rois = []
            target_height = 200
            for r in rois:
                h, w = r.shape[:2]
                if h == 0 or w == 0:
                    continue
                scale = target_height / h
                new_w = int(w * scale)
                resized = cv2.resize(r, (new_w, target_height))
                vis_rois.append(resized)
            if vis_rois:
                combined = np.hstack(vis_rois)
                cv2.imshow('Aktuelle ROI(s)', combined)
        else:
            # Keine ROIs: leeres Fenster oder Hinweis
            blank = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(blank, "Keine Barcodes erkannt", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow('Aktuelle ROI(s)', blank)

        # Hauptbild anzeigen
        cv2.imshow('Live Barcode ROI Debug', display_frame)

        # Tasteneingabe
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Bild speichern
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"debug_frame_{timestamp}.png", display_frame)
            for i, roi in enumerate(rois):
                cv2.imwrite(f"debug_roi_{timestamp}_{i}.png", roi)
            print(f"Bild und {len(rois)} ROIs gespeichert mit Timestamp {timestamp}")

    cap.release()
    cv2.destroyAllWindows()
    print("Programm beendet.")

if __name__ == "__main__":
    main()