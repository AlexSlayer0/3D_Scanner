#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Objektvermessung mit OAK-Kamera – saubere Version mit minimalen Fallbacks

Voraussetzung: Die Datei calib.json muss alle benötigten Werte enthalten:
    z_median, mm_per_pixel_763mm, focal_length_pix, COLOR_TOLERANCE, MIN_AREA, HUE_TOLERANCE (optional)
Fehlende Werte führen zu einer Fehlermeldung und Programmabbruch.

Ablauf:
- Live-Bild mit Disparität und zentraler ROI (Grün).
- In der ROI wird fortlaufend die minimale Tiefe (min_z) und die Höhe über Grund angezeigt.
- Mit Taste 'c' wird das aktuelle Bild eingefroren, der Nutzer klickt auf das zu vermessende Objekt.
- Auf Basis des angeklickten Pixels wird eine HSV-Maske erstellt und das größte Objekt segmentiert.
- Aus der Tiefenmaske wird der Median der Tiefe bestimmt und daraus Breite & Länge in mm berechnet.
- Alle Ergebnisse werden übersichtlich im Terminal ausgegeben.

Keine manuellen Eingaben, keine Standardwerte – die Kalibrierung muss stimmen.
"""

import cv2
import depthai as dai
import numpy as np
import json
import sys
from pathlib import Path

# ---------- Feste Parameter ----------
CALIB_FILE = "calib.json"
ROI_SIZE = 100          # Pixel für die zentrierte ROI (Höhenmessung)
HUE_TOLERANCE_DEFAULT = 10   # falls nicht in Kalibrierung
# --------------------------------------

def load_calibration(required_keys):
    """
    Lädt die Kalibrierung aus CALIB_FILE.
    Prüft, ob alle required_keys vorhanden sind.
    Gibt bei Erfolg das Dictionary zurück, sonst Programmabbruch.
    """
    calib_path = Path(CALIB_FILE)
    if not calib_path.exists():
        print(f"Fehler: {CALIB_FILE} nicht gefunden.")
        sys.exit(1)

    try:
        with calib_path.open() as f:
            data = json.load(f)
    except Exception as e:
        print(f"Fehler beim Lesen von {CALIB_FILE}: {e}")
        sys.exit(1)

    missing = [key for key in required_keys if key not in data]
    if missing:
        print(f"Fehler: In {CALIB_FILE} fehlen folgende Schlüssel: {missing}")
        sys.exit(1)

    # Optional: HUE_TOLERANCE ergänzen
    if "HUE_TOLERANCE" not in data:
        data["HUE_TOLERANCE"] = HUE_TOLERANCE_DEFAULT
        print(f"Hinweis: HUE_TOLERANCE nicht in Kalibrierung, verwende Default {HUE_TOLERANCE_DEFAULT}")

    print("Kalibrierung geladen:")
    for k, v in data.items():
        print(f"   {k} = {v}")
    return data

def create_hsv_mask(hsv_img, ref_hsv, hue_tol, sat_tol, val_tol):
    """Erzeugt eine Binärmaske basierend auf HSV-Toleranzen."""
    h, s, v = cv2.split(hsv_img)
    # HUE zirkulär behandeln (0-180)
    hue_diff = np.abs(h.astype(np.int16) - ref_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    mask_hue = hue_diff <= hue_tol
    mask_sat = np.abs(s - ref_hsv[1]) <= sat_tol
    mask_val = np.abs(v - ref_hsv[2]) <= val_tol
    return (mask_hue & mask_sat & mask_val).astype(np.uint8) * 255

def segment_object_by_click(image_bgr, window_name, hue_tol, color_tol, min_area):
    """
    Zeigt das Bild, wartet auf Mausklick und segmentiert das Objekt.
    Gibt (Maske, BoundingBox (x,y,w,h), Kontur) zurück oder (None,None,None) bei Abbruch/Fehler.
    """
    ref_color_bgr = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal ref_color_bgr
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_color_bgr = image_bgr[y, x]
            print(f"Referenzfarbe (BGR) an ({x},{y}): {ref_color_bgr}")
            cv2.destroyWindow(window_name)

    cv2.imshow(window_name, image_bgr)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("Klicke mit der linken Maustaste auf das Objekt (oder drücke 'q' zum Abbrechen).")

    while ref_color_bgr is None:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyWindow(window_name)
            return None, None, None

    # Kurze Pause, damit das Fenster sicher geschlossen wird
    cv2.waitKey(200)
    cv2.destroyAllWindows()

    # Referenz in HSV umrechnen
    ref_hsv = cv2.cvtColor(np.uint8([[ref_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    print(f"Referenz HSV: {ref_hsv}")

    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = create_hsv_mask(hsv_img, ref_hsv, hue_tol, color_tol, color_tol)

    # Konturen finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Keine Objektpixel gefunden.")
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        print(f"Objekt zu klein (Fläche = {area:.0f} px < {min_area} px).")
        return None, None, None

    x, y, w_px, h_px = cv2.boundingRect(largest)
    print(f"Bounding Box: x={x}, y={y}, w={w_px} px, h={h_px} px, Fläche={area:.0f} px²")
    return mask, (x, y, w_px, h_px), largest

def compute_object_size_mm(bbox_px, depth_mm, focal_length_px):
    """Berechnet Breite und Länge in mm aus Pixelmaßen, Tiefe und Brennweite."""
    w_px, h_px = bbox_px[2], bbox_px[3]
    width_mm = (w_px * depth_mm) / focal_length_px
    height_mm = (h_px * depth_mm) / focal_length_px
    return width_mm, height_mm

def live_mode(calib):
    """Hauptschleife: Live-Ansicht mit ROI und Objektvermessung bei Tastendruck."""
    print("\nLive-Modus gestartet. Drücke:")
    print("   'c' - Objektvermessung (Breite & Länge in mm)")
    print("   'q' - Beenden")
    print("   (Die ROI-Anzeige mit Höhe über Grund läuft dauerhaft)")

    # Kalibrierwerte bequem zuweisen
    z_median_mm = calib["z_median"]
    focal_length = calib["focal_length_pix"]
    color_tol = calib["COLOR_TOLERANCE"]
    hue_tol = calib["HUE_TOLERANCE"]
    min_area = calib["MIN_AREA"]

    # Pipeline aufbauen
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setCamera("left")
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setCamera("right")

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)

    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 100    # mm
    config.postProcessing.thresholdFilter.maxRange = 2000   # mm
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.disparity.link(xout_disparity.input)
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        print(f"📏 Brennweite (aus Kalibrierung): {focal_length:.1f} px")

        q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        max_disp = stereo.initialConfig.getMaxDisparity()

        while True:
            in_disparity = q_disparity.get()
            in_depth = q_depth.get()
            disparity_frame = in_disparity.getFrame()
            depth_frame = in_depth.getFrame()      # in Metern (Tiefe)

            # Disparity in Farbbild umwandeln
            disp_norm = (disparity_frame * (255.0 / max_disp)).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

            # ROI für Höhenmessung
            h, w = disparity_frame.shape
            cx, cy = w // 2, h // 2
            half = ROI_SIZE // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            # Tiefe in mm umrechnen (depth_frame ist in Metern)
            depth_mm = depth_frame * 1000.0
            roi_depth = depth_mm[y1:y2, x1:x2]
            valid = roi_depth[roi_depth > 0]
            min_z = np.min(valid) if valid.size > 0 else np.nan

            # Höhe über Grund (benötigt z_median)
            if not np.isnan(min_z):
                hoehe = z_median_mm - min_z
                text_hoehe = f"Höhe über Grund: {hoehe:.1f} mm"
                current_hoehe = hoehe
            else:
                text_hoehe = "Höhe über Grund: ---"
                current_hoehe = None

            text_minz = f"min_z: {min_z:.1f} mm" if not np.isnan(min_z) else "min_z: ---"
            current_min_z = min_z if not np.isnan(min_z) else None

            # ROI einzeichnen
            cv2.rectangle(disp_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp_color, text_minz, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(disp_color, text_hoehe, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Live - Disparity mit ROI", disp_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Segmentierung starten – aktuelles Bild verwenden
                mask, bbox, contour = segment_object_by_click(
                    disp_color, "Objektvermessung – Bitte anklicken",
                    hue_tol, color_tol, min_area
                )
                if mask is None:
                    continue

                # Tiefenwerte im segmentierten Bereich
                obj_depth = depth_mm[mask > 0]
                valid_obj = obj_depth[(obj_depth > 100) & (obj_depth < 5000)]  # Plausibler Bereich

                if valid_obj.size == 0:
                    print("Keine gültigen Tiefenwerte im Objekt. Messung abgebrochen.")
                    continue

                depth_median_mm = np.median(valid_obj)
                width_mm, height_mm = compute_object_size_mm(bbox, depth_median_mm, focal_length)
                hoehe_obj = z_median_mm - depth_median_mm

                # Ausgabe
                print("\n📐 Pixelmaße: {} x {} px".format(bbox[2], bbox[3]))
                print("\n📍 ROI-Messung (aktuell):")
                if current_min_z is not None:
                    print(f"   min_z = {current_min_z:.1f} mm")
                else:
                    print("   min_z = --- mm")
                if current_hoehe is not None:
                    print(f"   Höhe über Grund (ROI) = {current_hoehe:.1f} mm")
                else:
                    print("   Höhe über Grund (ROI) = --- mm")

                print("\n🔍 *** MESSERGEBNISSE (mm) ***")
                print(f"  Breite: {width_mm:.1f} mm")
                print(f"  Länge:  {height_mm:.1f} mm")
                print(f"  Entfernung (Median): {depth_median_mm:.1f} mm")
                print(f"  Höhe über Grund (Objekt): {hoehe_obj:.1f} mm")
                print("----------------------------------------\n")

    cv2.destroyAllWindows()

def main():
    print("=== Objektvermessung mit OAK-Kamera (saubere Version) ===")
    # Benötigte Kalibrierschlüssel
    required_keys = ["z_median", "mm_per_pixel_763mm", "focal_length_pix",
                     "COLOR_TOLERANCE", "MIN_AREA"]
    calib = load_calibration(required_keys)
    live_mode(calib)

if __name__ == "__main__":
    main()