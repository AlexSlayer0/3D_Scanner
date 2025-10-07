import cv2
import numpy as np
import torch
import re
import importlib
import time
import traceback

from fastsam import FastSAM

# ----------------- Funktion zum Laden von FastSAM mit dynamischer Allowlist -----------------
def try_load_fastsam_with_allowlist(checkpoint_path, device="cpu", max_attempts=10, delay=0.2):
    attempt = 0
    last_exc = None

    while attempt < max_attempts:
        attempt += 1
        try:
            model = FastSAM(checkpoint_path)
            model.to(device)
            return model
        except Exception as e:
            last_exc = e
            msg = str(e)

            # Pattern: Unsupported global: GLOBAL <full.path.ClassName> was not an allowed global
            m = re.search(r"Unsupported global: GLOBAL\s+([^\s]+)\s+was not an allowed global", msg)
            if not m:
                m2 = re.search(r"WeightsUnpickler error: Unsupported global: GLOBAL\s+([^\s]+)", msg)
                if m2:
                    m = m2

            if not m:
                print("Unerwarteter Ladefehler (nicht 'Unsupported global'):")
                traceback.print_exception(e)
                raise

            fullname = m.group(1)
            print(f"[Versuch {attempt}] Ergänze erlaubte Klasse: {fullname}")

            try:
                module_name, class_name = fullname.rsplit(".", 1)
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name)
            except Exception as ie:
                print(f"Konnte Klasse {fullname} nicht importieren: {ie}")
                traceback.print_exception(ie)
                raise

            # Klasse als safe global registrieren
            try:
                torch.serialization.add_safe_globals([cls])
                print(f"Erfolgreich registriert: {fullname}")
            except Exception as reg_err:
                print("Fehler beim Registrieren als safe global:", reg_err)
                traceback.print_exception(reg_err)
                raise

            time.sleep(delay)

    print("Modell konnte nach mehreren Versuchen nicht geladen werden.")
    raise last_exc

# ----------------- Konfiguration -----------------
IMAGE_PATH = "Test_Bild01.jpg"
OUTPUT_PATH = "Test_Bild01_fastpi.jpg"
DEVICE = "cpu"  # CPU-only (Windows oder Raspberry Pi)

# ----------------- FastSAM Modell laden -----------------
checkpoint = "FastSAM-x.pt"
model = try_load_fastsam_with_allowlist(checkpoint, device=DEVICE)

# ----------------- Bild laden & Vorverarbeitung -----------------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Bild '{IMAGE_PATH}' nicht gefunden.")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Optional: Bild skalieren für schnellere Verarbeitung
scale_factor = 0.5  # kleiner = schneller
img_rgb_small = cv2.resize(img_rgb, (0, 0), fx=scale_factor, fy=scale_factor)


# ----------------- Masken generieren -----------------
masks, scores, boxes = model.predict(img_rgb_small, iou_thresh=0.3, conf_thresh=0.3)
print(f"Masken erzeugt: {len(masks)}")

# ----------------- Zweitgrößte Maske wählen -----------------
if len(masks) < 2:
    chosen_mask = masks[0]
    chosen_box = boxes[0]
else:
    areas = [np.sum(m) for m in masks]
    idx = np.argsort(areas)[-2]  # zweitgrößte
    chosen_mask = masks[idx]
    chosen_box = boxes[idx]

# ----------------- Maske auf Originalbild übertragen -----------------
h, w = img_bgr.shape[:2]
chosen_mask_full = cv2.resize(chosen_mask.astype(np.uint8), (w, h))
chosen_mask_full = (chosen_mask_full > 0).astype(np.uint8) * 255

contours, _ = cv2.findContours(chosen_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
out = img_bgr.copy()
if contours:
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(out, [approx], -1, (0,0,255), 2)  # Umriss rot

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    cv2.rectangle(out, (x,y), (x+w_box,y+h_box), (0,255,0), 2)  # Bounding Box grün

# ----------------- Ergebnis speichern und anzeigen -----------------
cv2.imwrite(OUTPUT_PATH, out)
cv2.imshow("FastSAM Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
