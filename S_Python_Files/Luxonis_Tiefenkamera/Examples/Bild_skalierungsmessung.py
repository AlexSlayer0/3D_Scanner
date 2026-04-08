#!/usr/bin/env python3
"""
Objektvermessung im Jet-Farbbild – flexible Suche im gesamten Bild
Ausgabe: Breite und Höhe des Objekts in Pixeln.
Wenn eine Kalibrierungsdatei (calib.json) mit 'mm_per_pixel' existiert,
werden zusätzlich die Maße in Millimetern ausgegeben.
"""

import cv2
import numpy as np
import sys
import json
import os

# ------------------ Konfiguration ------------------
IMAGE_PATH = "saved_frames/disparity_color_1771775464.png"
COLOR_TOLERANCE = 15          # Toleranz für HSV-Maske (Sat+Val)
HUE_TOLERANCE = 5             # Toleranz für Farbton (zirkulär)
MIN_AREA = 50                 # Mindestfläche in Pixeln (zum Filtern von Rauschen)
CALIB_FILE = "calib.json"     # Kalibrierungsdatei (optional, für mm/Pixel)
# ---------------------------------------------------

def load_calibration():
    """
    Lädt den mm_per_pixel-Faktor aus der JSON-Datei.
    Gibt den Faktor zurück oder None, wenn keine gültige Datei vorhanden ist.
    """
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
            mm_per_pixel = data.get("mm_per_pixel")
            if mm_per_pixel is not None:
                print(f"Kalibrierung geladen: {mm_per_pixel:.4f} mm/Pixel")
                return mm_per_pixel
            else:
                print("Kalibrierungsdatei enthält keinen 'mm_per_pixel' Eintrag.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Fehler beim Lesen der Kalibrierungsdatei: {e}")
    else:
        print("Keine Kalibrierungsdatei gefunden. Nur Pixelmaße werden ausgegeben.")
    return None

def get_hsv_mask(hsv_img, ref_hsv, hue_tol=15, sat_tol=50, val_tol=50):
    """Erzeugt HSV-Maske für das gesamte Bild."""
    h, s, v = cv2.split(hsv_img)
    # Zirkuläre Hue-Differenz
    hue_diff = np.abs(h.astype(np.int16) - ref_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    mask_hue = hue_diff <= hue_tol
    mask_sat = np.abs(s - ref_hsv[1]) <= sat_tol
    mask_val = np.abs(v - ref_hsv[2]) <= val_tol
    mask = (mask_hue & mask_sat & mask_val).astype(np.uint8) * 255
    return mask

def main():
    # Kalibrierung laden
    mm_per_pixel = load_calibration()

    # Bild laden
    if not os.path.exists(IMAGE_PATH):
        print(f"Fehler: Bild {IMAGE_PATH} nicht gefunden.")
        sys.exit(1)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        sys.exit(1)

    print("Klicke mit der linken Maustaste auf das Objekt, um die Referenzfarbe auszuwählen.")
    print("Drücke nach dem Klick eine beliebige Taste, um fortzufahren.")

    ref_color_bgr = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal ref_color_bgr
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_color_bgr = img[y, x]
            print(f"Referenzfarbe (BGR) an Position ({x},{y}): {ref_color_bgr}")
            cv2.destroyWindow("Bild - Klick auf Objekt")

    cv2.imshow("Bild - Klick auf Objekt", img)
    cv2.setMouseCallback("Bild - Klick auf Objekt", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ref_color_bgr is None:
        print("Keine Referenzfarbe ausgewählt. Skript wird beendet.")
        return

    # Referenz in HSV umrechnen
    ref_hsv = cv2.cvtColor(np.uint8([[ref_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    print(f"Referenz HSV: {ref_hsv}")

    # HSV-Maske für das gesamte Bild berechnen
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = get_hsv_mask(hsv_img, ref_hsv,
                        hue_tol=HUE_TOLERANCE,
                        sat_tol=COLOR_TOLERANCE,
                        val_tol=COLOR_TOLERANCE)

    # Optional: Maske visualisieren
    cv2.imshow("Maske", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Konturen finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Keine Objektpixel gefunden. Toleranzen oder Referenzfarbe prüfen.")
        return

    # Größte Kontur auswählen (Annahme: das ist das Objekt)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_AREA:
        print(f"Objekt zu klein (Fläche = {area} px). Eventuell Rauschen.")
        return

    # Bounding Box in Pixel
    x, y, w_px, h_px = cv2.boundingRect(largest)

    print("\n--- Ergebnisse ---")
    print(f"Objektbreite: {w_px} Pixel")
    print(f"Objekthöhe:   {h_px} Pixel")
    print(f"Fläche:       {area:.0f} Pixel²")

    # Wenn Kalibrierungsfaktor vorhanden, auch Millimeter ausgeben
    if mm_per_pixel is not None:
        width_mm = w_px * mm_per_pixel
        height_mm = h_px * mm_per_pixel
        print(f"Objektbreite: {width_mm:.1f} mm")
        print(f"Objekthöhe:   {height_mm:.1f} mm")
        print(f"(Umrechnungsfaktor: {mm_per_pixel:.4f} mm/Pixel)")
    else:
        print("\nHinweis: Für eine Umrechnung in mm lege eine Datei 'calib.json' an")
        print("mit dem Inhalt: { \"mm_per_pixel\": DEIN_FAKTOR }")

    # Visualisierung: Objekt im Bild einrahmen
    cv2.rectangle(img, (x, y), (x + w_px, y + h_px), (0, 0, 255), 2)
    label = f"{w_px} px"
    if mm_per_pixel:
        label += f" ({w_px*mm_per_pixel:.1f} mm)"
    cv2.putText(img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Objekt gefunden", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()