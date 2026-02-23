#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Objektvermessung mit OAK-Kamera

- Schritt 1: Live-ROI zeigt dauerhaft die Höhe über Grund (aus z_median)
- Schritt 2: Bei Tastendruck 'c' kann ein Objekt durch Mausklick segmentiert und vermessen werden (Breite & Länge in mm)
- Unterstützt mehrere z_median-Werte in einer Liste (Auswahl beim Start)
- mm_per_pixel wird parallel aus einer zweiten Liste geladen
- Tiefenfilter: Nur Objekte, die mindestens GROUND_TOLERANCE_MM über dem Grund liegen, werden erkannt
- Alle Parameter werden aus calib.json geladen
"""

import cv2
import depthai as dai
import numpy as np
import os
import json

# ---------- Konfiguration ----------
CALIB_FILE = "calib.json"
ROI_SIZE = 100  # Pixel für die zentrierte ROI

# Standardwerte, falls in der Kalibrierung nicht vorhanden
DEFAULT_COLOR_TOLERANCE = 25
DEFAULT_MIN_AREA = 25
FALLBACK_FOCAL_LENGTH_PX = 580
DEFAULT_GROUND_TOLERANCE_MM = 5   # 0,5 cm
# -----------------------------------

def load_calibration():
    """Lädt Parameter aus calib.json.
       Erwartet entweder:
         - "z_median_liste" und "mm_per_pixel_liste" (gleich lang)
         - oder einzelne Werte "z_median" und "mm_per_pixel" (für alte Struktur)
       Liefert字典 mit "z_median", "mm_per_pixel", "focal_length_pix", etc.
    """
    if not os.path.exists(CALIB_FILE):
        print(f"{CALIB_FILE} nicht gefunden. Verwende Standardwerte.")
        return {
            "z_median": None,
            "mm_per_pixel": None,
            "focal_length_pix": FALLBACK_FOCAL_LENGTH_PX,
            "COLOR_TOLERANCE": DEFAULT_COLOR_TOLERANCE,
            "MIN_AREA": DEFAULT_MIN_AREA,
            "ground_tolerance_mm": DEFAULT_GROUND_TOLERANCE_MM
        }

    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)

        calib = {
            "focal_length_pix": data.get("focal_length_pix", FALLBACK_FOCAL_LENGTH_PX),
            "COLOR_TOLERANCE": data.get("COLOR_TOLERANCE", DEFAULT_COLOR_TOLERANCE),
            "MIN_AREA": data.get("MIN_AREA", DEFAULT_MIN_AREA),
            "ground_tolerance_mm": data.get("GROUND_TOLERANCE_MM", DEFAULT_GROUND_TOLERANCE_MM)
        }

        # Prüfe, ob Listen existieren
        z_list = data.get("z_median_liste")
        mm_list = data.get("mm_per_pixel_liste")

        if z_list and isinstance(z_list, list) and len(z_list) > 0:
            print("Mehrere z_median Werte gefunden:")
            for i, z in enumerate(z_list):
                print(f"  {i}: {z} mm")

            # Aktiven Index bestimmen
            active = data.get("active_z_median_index")
            if active is not None and 0 <= active < len(z_list):
                selected = active
                print(f"Verwende aktiven Index {selected} aus Datei.")
            else:
                while True:
                    try:
                        sel = input(f"Bitte wählen Sie den gewünschten Index (0-{len(z_list)-1}): ").strip()
                        if sel == "":
                            sel = "0"
                        selected = int(sel)
                        if 0 <= selected < len(z_list):
                            break
                        else:
                            print(f"Index muss zwischen 0 und {len(z_list)-1} liegen.")
                    except ValueError:
                        print("Ungültige Eingabe.")

            calib["z_median"] = z_list[selected]

            # Passenden mm_per_pixel Wert laden (falls Liste vorhanden und gleich lang)
            if mm_list and isinstance(mm_list, list) and len(mm_list) == len(z_list):
                calib["mm_per_pixel"] = mm_list[selected]
            else:
                print("Warnung: mm_per_pixel_liste fehlt oder hat falsche Länge. Fallback auf Einzelwert.")
                # Fallback: Versuche alten Einzelwert "mm_per_pixel" oder "mm_per_pixel_763mm"
                calib["mm_per_pixel"] = data.get("mm_per_pixel") or data.get("mm_per_pixel_763mm")
        else:
            # Alte Struktur: einzelne Werte
            calib["z_median"] = data.get("z_median")
            calib["mm_per_pixel"] = data.get("mm_per_pixel") or data.get("mm_per_pixel_763mm")

        # Ausgabe der geladenen Werte
        print("Kalibrierung geladen:")
        if calib["z_median"] is not None:
            print(f"   z_median = {calib['z_median']:.1f} mm")
        if calib["mm_per_pixel"] is not None:
            print(f"   mm_per_pixel = {calib['mm_per_pixel']:.4f} mm/px")
        print(f"   focal_length_pix = {calib['focal_length_pix']:.1f} px")
        print(f"   COLOR_TOLERANCE = {calib['COLOR_TOLERANCE']}")
        print(f"   MIN_AREA = {calib['MIN_AREA']}")
        print(f"   GROUND_TOLERANCE_MM = {calib['ground_tolerance_mm']} mm")
        return calib

    except Exception as e:
        print(f"Fehler beim Lesen von {CALIB_FILE}: {e}")
        return {
            "z_median": None,
            "mm_per_pixel": None,
            "focal_length_pix": FALLBACK_FOCAL_LENGTH_PX,
            "COLOR_TOLERANCE": DEFAULT_COLOR_TOLERANCE,
            "MIN_AREA": DEFAULT_MIN_AREA,
            "ground_tolerance_mm": DEFAULT_GROUND_TOLERANCE_MM
        }

def create_hsv_mask(hsv_img, ref_hsv, hue_tol, sat_tol, val_tol):
    h, s, v = cv2.split(hsv_img)
    hue_diff = np.abs(h.astype(np.int16) - ref_hsv[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    mask_hue = hue_diff <= hue_tol
    mask_sat = np.abs(s - ref_hsv[1]) <= sat_tol
    mask_val = np.abs(v - ref_hsv[2]) <= val_tol
    mask = (mask_hue & mask_sat & mask_val).astype(np.uint8) * 255
    return mask

def segment_object_by_click(image_bgr, window_name, color_tolerance, min_area):
    """Zeigt Bild, wartet auf Mausklick, segmentiert Objekt (nur farbbasiert) und gibt Maske, BoundingBox zurück."""
    ref_color_bgr = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal ref_color_bgr
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_color_bgr = image_bgr[y, x]
            print(f"Referenzfarbe (BGR) an ({x},{y}): {ref_color_bgr}")
            cv2.destroyWindow(window_name)

    cv2.imshow(window_name, image_bgr)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("Klicke mit der linken Maustaste auf das Objekt.")
    while ref_color_bgr is None:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyWindow(window_name)
            return None, None, None
    cv2.waitKey(200)
    cv2.destroyAllWindows()

    ref_hsv = cv2.cvtColor(np.uint8([[ref_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    print(f"Referenz HSV: {ref_hsv}")

    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    HUE_TOLERANCE = 5  # fest
    mask = create_hsv_mask(hsv_img, ref_hsv, HUE_TOLERANCE, color_tolerance, color_tolerance)

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
    print(f"Bounding Box (Farbe): x={x}, y={y}, w={w_px} px, h={h_px} px, Fläche={area:.0f} px²")
    return mask, (x, y, w_px, h_px), largest

def compute_object_size_mm(bbox_px, depth_value_mm, focal_length_px):
    w_px, h_px = bbox_px[2], bbox_px[3]
    width_mm = (w_px * depth_value_mm) / focal_length_px
    height_mm = (h_px * depth_value_mm) / focal_length_px
    return width_mm, height_mm

def live_mode(calib):
    print("\nLive-Modus gestartet. Drücke:")
    print("   'c' - Objektvermessung (Breite & Länge in mm) mit Tiefenfilter")
    print("   'q' - Beenden")
    print("   (Die ROI-Anzeige mit Höhe über Grund läuft dauerhaft)")

    z_median_mm = calib["z_median"]
    focal_length = calib["focal_length_pix"]
    mm_per_pixel = calib["mm_per_pixel"]   # jetzt einheitlich
    color_tol = calib["COLOR_TOLERANCE"]
    min_area = calib["MIN_AREA"]
    ground_tolerance = calib["ground_tolerance_mm"]

    # Pipeline aufbauen
    pipeline = dai.Pipeline()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

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
    config.postProcessing.thresholdFilter.minRange = 100    # 100 mm
    config.postProcessing.thresholdFilter.maxRange = 2000   # 2000 mm
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.disparity.link(xout_disparity.input)
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        print(f"📏 Brennweite (aus Kalibrierung): {focal_length:.1f} px")
        print(f"🕳️  Tiefentoleranz: {ground_tolerance} mm (Objekte müssen mind. {ground_tolerance} mm über Grund liegen)")

        q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        max_disp = stereo.initialConfig.getMaxDisparity()

        while True:
            in_disparity = q_disparity.get()
            in_depth = q_depth.get()
            disparity_frame = in_disparity.getFrame()
            depth_frame = in_depth.getFrame()  # in mm (wie von depthai geliefert)

            # Disparity in Farbbild umwandeln
            disp_norm = (disparity_frame * (255.0 / max_disp)).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

            # ROI für Höhenmessung (Schritt 1)
            h, w = disparity_frame.shape
            cx, cy = w // 2, h // 2
            half = ROI_SIZE // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            depth_mm = depth_frame  # keine Umrechnung nötig
            roi_depth = depth_mm[y1:y2, x1:x2]
            valid = roi_depth[roi_depth > 0]
            min_z = np.min(valid) if valid.size > 0 else float('nan')

            # Höhe über Grund (Schritt 1)
            if z_median_mm is not None and not np.isnan(min_z):
                hoehe = z_median_mm - min_z
                text_hoehe = f"Höhe über Grund: {hoehe:.1f} mm"
            else:
                text_hoehe = "Höhe über Grund: ---"
            text_minz = f"min_z: {min_z:.1f} mm" if not np.isnan(min_z) else "min_z: ---"

            current_min_z = min_z
            current_hoehe = hoehe if z_median_mm is not None and not np.isnan(min_z) else None

            # ROI einzeichnen und Texte
            cv2.rectangle(disp_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp_color, text_minz, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(disp_color, text_hoehe, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            cv2.imshow("Live - Disparity mit ROI", disp_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Farbbasierte Segmentierung
                mask, bbox, contour = segment_object_by_click(disp_color, "Objektvermessung – Bitte anklicken", color_tol, min_area)
                if mask is None:
                    continue

                # Tiefenfilter anwenden (falls z_median bekannt)
                if z_median_mm is not None:
                    mask_depth = (depth_mm < (z_median_mm - ground_tolerance)) & (depth_mm > 0)
                    mask_depth_uint8 = mask_depth.astype(np.uint8) * 255
                    mask_combined = cv2.bitwise_and(mask, mask_depth_uint8)

                    # Neue Konturen nach Tiefenfilter
                    contours2, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours2:
                        print("Nach Tiefenfilterung keine Objektpixel übrig.")
                        continue
                    largest2 = max(contours2, key=cv2.contourArea)
                    area2 = cv2.contourArea(largest2)
                    if area2 < min_area:
                        print(f"Objekt nach Tiefenfilterung zu klein (Fläche = {area2:.0f} px < {min_area} px).")
                        continue
                    # Aktualisiere Maske, BoundingBox und Kontur
                    x, y, w_px, h_px = cv2.boundingRect(largest2)
                    bbox = (x, y, w_px, h_px)
                    contour = largest2
                    mask = mask_combined
                    obj_depth = depth_mm[mask_combined > 0]
                    print(f"Bounding Box (mit Tiefenfilter): x={x}, y={y}, w={w_px} px, h={h_px} px, Fläche={area2:.0f} px²")
                else:
                    # Kein Tiefenfilter möglich
                    obj_depth = depth_mm[mask > 0]

                                # Tiefenwerte des Objekts auslesen
                valid_obj = obj_depth[(obj_depth > 0) & (obj_depth < 5000)]

                w_px, h_px = bbox[2], bbox[3]
                width_mm = height_mm = None
                depth_median_mm = None
                depth_min_mm = None          # neu: für Höhenberechnung
                hoehe_ueber_grund_obj = None

                if valid_obj.size > 0:
                    depth_median_mm = np.median(valid_obj)
                    depth_min_mm = np.min(valid_obj)      # kleinste Entfernung = höchster Punkt
                    width_mm, height_mm = compute_object_size_mm(bbox, depth_median_mm, focal_length)
                else:
                    print("Keine automatischen Tiefenwerte im Objekt.")
                    if mm_per_pixel is not None:
                        print("Fallback: Berechnung mit mm_per_pixel aus Kalibrierung")
                        width_mm = w_px * mm_per_pixel
                        height_mm = h_px * mm_per_pixel
                    else:
                        manual = input("   Manuelle Eingabe der Objektentfernung in mm (Enter für nur Pixelmaße): ").strip()
                        if manual:
                            try:
                                depth_manual_mm = float(manual)
                                width_mm, height_mm = compute_object_size_mm(bbox, depth_manual_mm, focal_length)
                                depth_median_mm = depth_manual_mm
                                depth_min_mm = depth_manual_mm
                            except ValueError:
                                print("   Ungültige Eingabe - nur Pixelmaße.")

                # Höhe über Grund des Objekts: Verwende ROI-Minimum, falls vorhanden
                if not np.isnan(current_min_z) and z_median_mm is not None:
                    hoehe_ueber_grund_obj = z_median_mm - current_min_z
                else:
                    hoehe_ueber_grund_obj = None


                # Ausgabe
                print("\n📐 Pixelmaße: {} x {} px".format(w_px, h_px))
                print("\n📍 ROI-Messung (aktuell):")
                if not np.isnan(current_min_z):
                    print(f"   min_z = {current_min_z:.1f} mm")
                else:
                    print("   min_z = --- mm")
                if current_hoehe is not None:
                    print(f"   Höhe über Grund (ROI) = {current_hoehe:.1f} mm")
                else:
                    print("   Höhe über Grund (ROI) = --- mm")

                print("\n🔍 *** MESSERGEBNISSE (mm) ***")
                if width_mm is not None and height_mm is not None:
                    print(f"  Breite: {width_mm:.1f} mm")
                    print(f"  Länge:  {height_mm:.1f} mm")
                else:
                    print("  Breite: --- mm")
                    print("  Länge:  --- mm")
                if depth_median_mm is not None:
                    print(f"  Entfernung: {depth_median_mm:.1f} mm")
                else:
                    print("  Entfernung: --- mm")
                if hoehe_ueber_grund_obj is not None:
                    print(f"  Höhe über Grund (Objekt): {hoehe_ueber_grund_obj:.1f} mm")
                else:
                    print("  Höhe über Grund (Objekt): --- mm")
                if z_median_mm is not None:
                    print(f"  (Tiefenfilter: Tiefe < {z_median_mm - ground_tolerance:.1f} mm)")

    cv2.destroyAllWindows()

def main():
    print("=== Objektvermessung mit OAK-Kamera ===")
    calib = load_calibration()
    live_mode(calib)

if __name__ == "__main__":
    main()