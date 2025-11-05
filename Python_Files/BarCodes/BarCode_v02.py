import cv2
import contextlib
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from pyzbar.pyzbar import decode

# --- Umschalter: True = Kamera, False = Bilderordner ---
USE_CAMERA = True  

# --- Pfad nur nötig, wenn Kamera deaktiviert ---
INPUT_DIR = Path("GUI_Anzeige")


# --- Bildvorverarbeitungsfunktionen ---
def unsharp(gray, amount=1.5):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

def clahe_equalize(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def invert(gray):
    return 255 - gray

def adaptive_thresh(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

def scale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)

def try_decode(img_np):
    pil_img = Image.fromarray(img_np)
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        return decode(pil_img)


# --- Decodierung einer ROI mit allen Vorverarbeitungen ---
def process_roi(roi, img_name):
    preprocess_methods = {
        "orig": lambda x: x,
        "unsharp": unsharp,
        "clahe": clahe_equalize,
        "invert": invert,
        "adaptive": adaptive_thresh,
        "clahe+adaptive+inv": lambda x: invert(adaptive_thresh(clahe_equalize(x)))
    }
    scales = [1.0, 1.5, 2.0, 3.0]
    rotations = list(range(-40, 41, 5))
    gray = roi[:,:,2]  # R-Kanal

    for pname, func in preprocess_methods.items():
        img_proc = func(gray)
        for s in scales:
            scaled = scale(img_proc, s)
            for angle in rotations:
                rotated = rotate(scaled, angle)
                dec = try_decode(rotated)
                if dec:
                    # Erstes Ergebnis zurückgeben
                    d = dec[0]
                    val = d.data.decode("utf-8", errors="ignore")
                    typ = d.type
                    return {"found": True, "value": val, "type": typ}
    return {"found": False, "value": None, "type": None}

# --- Hauptfunktion ---
def main():
    model = YOLO("YOLOV8s_Barcode_Detection.pt")

    # Kamera-Modus
    if USE_CAMERA:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Kamera konnte nicht geöffnet werden.")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[ERROR] Kein Frame von Kamera erhalten.")
            return

        img = frame
        results = model.predict(img, verbose=False)

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                roi = img[y1:y2, x1:x2]
                process_roi(roi, "Kamera_Bild")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Kamera Barcode Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Ordner-Modus
    else:
        bilder = [
            f for f in INPUT_DIR.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            and "barcode" in f.stem.lower()
        ]

        if not bilder:
            print(f"Keine passenden Bilder gefunden in {INPUT_DIR}")
            return

        gesamt, erkannt = 0, 0
        for img_path in sorted(bilder):
            gesamt += 1
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"{img_path.name}: Bild konnte nicht geladen werden!")
                continue

            results = model.predict(img, verbose=False)
            yolo_boxes = []
            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    yolo_boxes.append((x1, y1, x2 - x1, y2 - y1))

            for det in yolo_boxes:
                x, y, w, h = det
                roi = img[y:y + h, x:x + w]
                if process_roi(roi, img_path.name):
                    erkannt += 1
                    break

        quote = (erkannt / gesamt * 100) if gesamt > 0 else 0
        print(f"\nTrefferquote: {erkannt}/{gesamt} erkannt ({quote:.2f} %)")


if __name__ == "__main__":
    main()
