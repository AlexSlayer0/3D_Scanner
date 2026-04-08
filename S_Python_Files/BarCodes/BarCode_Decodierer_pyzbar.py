import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
from collections import defaultdict

stats_padding = defaultdict(int)
stats_method = defaultdict(int)
stats_scale = defaultdict(int)
stats_rotation = defaultdict(int)

# INPUT_DIR = Path("ParcelBar")   # <- anpassen
INPUT_DIR = Path("InventBar")   # <- anpassen

# --- Hilfsfunktionen ---
def yolo_to_roi(line, iw, ih, pad=0.3):  # jetzt 30% Standardpadding
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError("Ungültige Koordinatenzeile")
    xc, yc, w, h = map(float, parts[1:5])
    w_px, h_px = int(w * iw), int(h * ih)
    xc_px, yc_px = xc * iw, yc * ih
    x0, y0 = int(xc_px - w_px / 2), int(yc_px - h_px / 2)
    pad_px = int(max(w_px, h_px) * pad)
    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(iw, x0 + w_px + 2 * pad_px)
    y1 = min(ih, y0 + h_px + 2 * pad_px)
    return x0, y0, x1, y1

def unsharp(gray, amount=1.5):
    blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

def clahe_equalize(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def invert(gray):
    return 255 - gray

def adaptive_thresh(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

def scale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_CUBIC)

def try_decode(img_np):
    pil_img = Image.fromarray(img_np)
    import contextlib
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        return decode(pil_img)

# --- Hauptlogik für ein Bild ---
def process_image(img_path):
    txt_path = img_path.with_suffix(".txt")
    if not txt_path.exists():
        print(f"\n{img_path.name}: ⚠️ keine TXT gefunden")
        return False

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"\n{img_path.name}: ❌ Fehler beim Laden")
        return False
    ih, iw = bgr.shape[:2]

    with open(txt_path, "r") as f:
        line = f.readline()

    padding_steps = [0.1, 0.3, 0.4]  # 30/40%
    preprocess_methods = {
        "orig": lambda x: x,
        "clahe": clahe_equalize,
        "unsharp": unsharp,
        "adaptive": adaptive_thresh,
        "invert": invert,
        "clahe+adaptive+inv": lambda x: invert(adaptive_thresh(clahe_equalize(x)))
    }
    scales = [1.0, 1.5, 2.0]  # Originalgröße, 150%, 200%
    rotations = [-135, -90, -45, -15, -5, 0, 5, 15, 45, 90, 135, 180]   # fokussiert auf die erfolgreichen Winkel

    for pad in padding_steps:
        try:
            x0, y0, x1, y1 = yolo_to_roi(line, iw, ih, pad)
        except Exception as e:
            print(f"{img_path.name}: ❌ Fehler bei ROI mit pad={pad}: {e}")
            continue

        roi = bgr[y0:y1, x0:x1]
        # Farbkanal-Extraktion (R-Kanal)
        gray = roi[:,:,2]

        for pname, func in preprocess_methods.items():
            img_proc = func(gray)
            for s in scales:
                scaled = scale(img_proc, s)
                for angle in rotations:
                    rotated = rotate(scaled, angle)
                    dec = try_decode(rotated)
                    if dec:
                        print(f"\n✅ {img_path.name} erkannt! "
                              f"(Padding={pad*100:.0f}%, {pname}, scale={s}, rot={angle}°)")
                        for d in dec:
                            val = d.data.decode("utf-8", errors="ignore")
                            print(f"   Typ: {d.type}, Wert: {val}")
                            stats_padding[pad] += 1
                            stats_method[pname] += 1
                            stats_scale[s] += 1
                            stats_rotation[angle] += 1
                        return True

    print(f"\n❌ {img_path.name} kein Barcode erkannt (alle Varianten getestet)")
    return False

# --- Alle Bilder im Ordner ---
def main():
    bilder = [f for f in INPUT_DIR.iterdir()
              if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff")]
    if not bilder:
        print("⚠️ Keine Bilder gefunden in", INPUT_DIR)
        return

    gesamt, erkannt = 0, 0
    for img_path in sorted(bilder):
        gesamt += 1
        if process_image(img_path):
            erkannt += 1

    quote = (erkannt / gesamt * 100) if gesamt > 0 else 0
    print("\n=== STATISTIK ===")
    print("Padding-Werte (Erfolge):", dict(stats_padding))
    print("Methoden (Erfolge):", dict(stats_method))
    print("Skalierungen (Erfolge):", dict(stats_scale))
    print("Rotationen (Erfolge):", dict(stats_rotation))
    print(f"\nTrefferquote: {erkannt}/{gesamt} erkannt ({quote:.2f} %)")
    print("Top-Paddings:", sorted(stats_padding.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    main()
