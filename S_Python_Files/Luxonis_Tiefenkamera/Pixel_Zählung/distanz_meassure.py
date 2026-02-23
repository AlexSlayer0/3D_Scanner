#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Objektvermessung mit OAK-Kamera

- Schritt 1: Live-ROI zeigt dauerhaft die Höhe über Grund (aus z_median)
- Schritt 2: Bei Tastendruck 'c' kann ein Objekt durch Mausklick segmentiert und vermessen werden (Breite & Länge in mm)
- Kalibrierungsparameter (z_median, mm_per_pixel, focal_length_pix, COLOR_TOLERANCE, MIN_AREA) werden aus calib.json geladen
- Fallbacks: Wenn Kalibrierungswerte fehlen, werden Standardwerte verwendet oder der Nutzer zur manuellen Eingabe aufgefordert
- Alle Ergebnisse (Pixelmaße, mm-Maße, Höhen über Grund) werden übersichtlich im Terminal ausgegeben
"""

import cv2
import depthai as dai
import numpy as np
import os
import json
import time

# ---------- Konfiguration ----------
CALIB_FILE = "calib.json"
ROI_SIZE = 100  # Pixel für die zentrierte ROI

# Standardwerte, falls in der Kalibrierung nicht vorhanden
DEFAULT_COLOR_TOLERANCE = 25
DEFAULT_MIN_AREA = 25
FALLBACK_FOCAL_LENGTH_PX = 580
# -----------------------------------

def load_calibration():
    """Lädt alle Parameter aus calib.json, ersetzt Fehlendes durch Standards."""
    if not os.path.exists(CALIB_FILE):
        print(f"{CALIB_FILE} nicht gefunden. Verwende Standardwerte.")
        return {
            "z_median": None,
            "mm_per_pixel_763mm": None,
            "focal_length_pix": FALLBACK_FOCAL_LENGTH_PX,
            "COLOR_TOLERANCE": DEFAULT_COLOR_TOLERANCE,
            "MIN_AREA": DEFAULT_MIN_AREA
        }
    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)
        calib = {
            "z_median": data.get("z_median"),
            "mm_per_pixel_763mm": data.get("mm_per_pixel_763mm"),
            "focal_length_pix": data.get("focal_length_pix", FALLBACK_FOCAL_LENGTH_PX),
            "COLOR_TOLERANCE": data.get("COLOR_TOLERANCE", DEFAULT_COLOR_TOLERANCE),
            "MIN_AREA": data.get("MIN_AREA", DEFAULT_MIN_AREA)
        }
        print("Kalibrierung geladen:")
        if calib["z_median"] is not None:
            print(f"   z_median = {calib['z_median']:.1f} mm")
        if calib["mm_per_pixel_763mm"] is not None:
            print(f"   mm_per_pixel_763mm = {calib['mm_per_pixel_763mm']:.4f} mm/px")
        print(f"   focal_length_pix = {calib['focal_length_pix']:.1f} px")
        print(f"   COLOR_TOLERANCE = {calib['COLOR_TOLERANCE']}")
        print(f"   MIN_AREA = {calib['MIN_AREA']}")
        return calib
    except Exception as e:
        print(f"Fehler beim Lesen von {CALIB_FILE}: {e}")
        return {
            "z_median": None,
            "mm_per_pixel_763mm": None,
            "focal_length_pix": FALLBACK_FOCAL_LENGTH_PX,
            "COLOR_TOLERANCE": DEFAULT_COLOR_TOLERANCE,
            "MIN_AREA": DEFAULT_MIN_AREA
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
    """Zeigt Bild, wartet auf Mausklick, segmentiert Objekt und gibt Maske, BoundingBox zurück."""
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
    HUE_TOLERANCE = 5  # fest, kann bei Bedarf auch aus Kalibrierung kommen
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
    print(f"Bounding Box: x={x}, y={y}, w={w_px} px, h={h_px} px, Fläche={area:.0f} px²")
    return mask, (x, y, w_px, h_px), largest

def compute_object_size_mm(bbox_px, depth_value_mm, focal_length_px):
    w_px, h_px = bbox_px[2], bbox_px[3]
    width_mm = (w_px * depth_value_mm) / focal_length_px
    height_mm = (h_px * depth_value_mm) / focal_length_px
    return width_mm, height_mm

def live_mode(calib):
    print("\nLive-Modus gestartet. Drücke:")
    print("   'c' - Objektvermessung (Breite & Länge in mm)")
    print("   'q' - Beenden")
    print("   (Die ROI-Anzeige mit Höhe über Grund läuft dauerhaft)")

    z_median_mm = calib["z_median"]
    focal_length = calib["focal_length_pix"]
    mm_per_pixel_763mm = calib["mm_per_pixel_763mm"]
    color_tol = calib["COLOR_TOLERANCE"]
    min_area = calib["MIN_AREA"]

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

        q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        max_disp = stereo.initialConfig.getMaxDisparity()

        while True:
            in_disparity = q_disparity.get()
            in_depth = q_depth.get()
            disparity_frame = in_disparity.getFrame()
            depth_frame = in_depth.getFrame()  # in Metern

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

            depth_mm = depth_frame 
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
                mask, bbox, contour = segment_object_by_click(disp_color, "Objektvermessung – Bitte anklicken", color_tol, min_area)
                if mask is None:
                    continue

                obj_depth = depth_mm[mask > 0]
                valid_obj = obj_depth[(obj_depth > 0) & (obj_depth < 5000)]

                w_px, h_px = bbox[2], bbox[3]
                width_mm = height_mm = None
                depth_median_mm = None
                hoehe_ueber_grund_obj = None

                if valid_obj.size > 0:
                    depth_median_mm = np.median(valid_obj)
                    width_mm, height_mm = compute_object_size_mm(bbox, depth_median_mm, focal_length)
                else:
                    print("Keine automatischen Tiefenwerte im Objekt.")
                    if calib["mm_per_pixel_763mm"] is not None:
                        print("Fallback: Berechnung mit mm_per_pixel_763mm aus Kalibrierung")
                        width_mm = w_px * calib["mm_per_pixel_763mm"]
                        height_mm = h_px * calib["mm_per_pixel_763mm"]
                    else:
                        manual = input("   Manuelle Eingabe der Objektentfernung in mm (Enter für nur Pixelmaße): ").strip()
                        if manual:
                            try:
                                depth_manual_mm = float(manual)
                                width_mm, height_mm = compute_object_size_mm(bbox, depth_manual_mm, focal_length)
                                depth_median_mm = depth_manual_mm
                            except ValueError:
                                print("   Ungültige Eingabe – nur Pixelmaße.")

                if depth_median_mm is not None and z_median_mm is not None:
                    hoehe_ueber_grund_obj = z_median_mm - depth_median_mm

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


    cv2.destroyAllWindows()

def main():
    print("=== Objektvermessung mit OAK-Kamera ===")
    calib = load_calibration()
    live_mode(calib)


if __name__ == "__main__":
    main()