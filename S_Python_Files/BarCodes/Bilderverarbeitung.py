# workers/BarCode_v03_minimal.py
import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_preprocessed_images(image_path: str, output_base: str = "Bilderverarbeitung"):
    """
    Lädt ein Bild und speichert vier Vorverarbeitungsvarianten:
        - original.png (RGB)
        - clahe.png
        - unsharp.png
        - invertiert.png
    """
    if not os.path.exists(image_path):
        logger.error(f"Bilddatei nicht gefunden: {image_path}")
        return

    # Bild laden und in RGB konvertieren
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.error(f"Bild konnte nicht geladen werden: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Ordner für Ausgabe erstellen
    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_base, basename)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Original (RGB)
    cv2.imwrite(os.path.join(out_dir, "original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # 2. CLAHE (Kontrastverstärkung)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    cv2.imwrite(os.path.join(out_dir, "clahe.png"), clahe_img)

    # 3. Unsharp Mask (Schärfung)
    blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
    unsharp_img = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
    cv2.imwrite(os.path.join(out_dir, "unsharp.png"), unsharp_img)

    # 4. Invertierung
    invert_img = 255 - gray
    cv2.imwrite(os.path.join(out_dir, "invertiert.png"), invert_img)

    logger.info(f"Vorverarbeitungsbilder gespeichert in {out_dir}")

if __name__ == "__main__":
    # Beispielaufruf
    image_path = "ProductBarcode496.jpg"   # Pfad anpassen
    save_preprocessed_images(image_path)