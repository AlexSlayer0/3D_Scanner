#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Objektvermessung mit OAK-Kamera
- Automatisch erkennung von einem einzigen blauen objekt über Grund
- zurückgegeben werden Länge, Breite und Höhe über Grund in mm
- Live-Visualisierung mit blau/weiß Bild (blau = über Grund) soll aus sein bei Interfacev08.py, aber hier zur besseren Orientierung eingebaut bleiben
- ROI Pixelwete gegebenfalls erhöhen wenn nicht 450 mm auf 450 mm erreicht werden
- Kalibrierung vielleicht nochmals anschaugen bei einer distanz von 580 mm, da hier die meisten Messungen stattfinden werden, und die mm/Pixel Werte für diese Distanz besonders wichtig sind
"""

import cv2
import depthai as dai
import numpy as np
import os
import json

CALIB_FILE = "distanz_calibration.json"
ROI_SIZE = 250  # Pixel für die zentrierte ROI

# Standardwerte
DEFAULT_COLOR_TOLERANCE = 25      # nicht mehr für Segmentierung genutzt, aber für Kompatibilität
DEFAULT_MIN_AREA = 25
FALLBACK_FOCAL_LENGTH_PX = 580
DEFAULT_GROUND_TOLERANCE_MM = 10

def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f"{CALIB_FILE} nicht gefunden. Verwende Standardwerte.")
        return {
            "z_median": None,
            "mm_per_pixel": None,
            "focal_length_pix": FALLBACK_FOCAL_LENGTH_PX,
            "COLOR_TOLERANCE": DEFAULT_COLOR_TOLERANCE,
            "MIN_AREA": DEFAULT_MIN_AREA,
            "ground_tolerance_mm": DEFAULT_GROUND_TOLERANCE_MM,
        }

    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)

        calib = {
            "COLOR_TOLERANCE": data.get("COLOR_TOLERANCE", DEFAULT_COLOR_TOLERANCE),
            "MIN_AREA": data.get("MIN_AREA", DEFAULT_MIN_AREA),
            "ground_tolerance_mm": data.get("GROUND_TOLERANCE_MM", DEFAULT_GROUND_TOLERANCE_MM),
        }

        z_list = data.get("z_median_liste")
        mm_list = data.get("mm_per_pixel_liste")
        focal_list = data.get("focal_length_pix")

        print("Mehrere z_median Werte gefunden:")
        for i, z in enumerate(z_list):
            print(f"  {i}: {z} mm")

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
                    print(f"Index muss zwischen 0 und {len(z_list)-1} liegen.")
                except ValueError:
                    print("Ungültige Eingabe.")

        calib["z_median"] = z_list[selected]

        if mm_list and isinstance(mm_list, list) and len(mm_list) == len(z_list):
            calib["mm_per_pixel"] = mm_list[selected]
        else:
            print("Warnung: mm_per_pixel_liste fehlt oder falsche Länge. Fallback auf Einzelwert.")
            calib["mm_per_pixel"] = data.get("mm_per_pixel") or data.get("mm_per_pixel_763mm")

        if isinstance(focal_list, list) and len(focal_list) == len(z_list):
            calib["focal_length_pix"] = focal_list[selected]
        else:
            if focal_list is not None and not isinstance(focal_list, list):
                calib["focal_length_pix"] = focal_list
            else:
                calib["focal_length_pix"] = FALLBACK_FOCAL_LENGTH_PX
                print(f"Warnung: focal_length_pix Liste fehlt. Verwende Fallback {FALLBACK_FOCAL_LENGTH_PX} px.")


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
            "ground_tolerance_mm": DEFAULT_GROUND_TOLERANCE_MM,
        }

def compute_object_size_mm(bbox_px, depth_value_mm, focal_length_px):
    w_px, h_px = bbox_px[2], bbox_px[3]
    width_mm = (w_px * depth_value_mm) / focal_length_px
    height_mm = (h_px * depth_value_mm) / focal_length_px
    return width_mm, height_mm

def direct_mode(calib):
    """Einmalige Messung ohne Benutzerinteraktion - nutzt zentrierte ROI.
       Rückgabe: Dictionary mit Messergebnissen oder Fehlerstatus."""
    print("\n--- Direkter Messmodus ---")
    z_median_mm = calib["z_median"]
    focal_length = calib["focal_length_pix"]
    mm_per_pixel = calib["mm_per_pixel"]
    min_area = calib["MIN_AREA"]
    ground_tolerance = calib["ground_tolerance_mm"]

    # Ergebnis-Dictionary vorbereiten
    result = {
        "success": False,
        "length": 0.0,
        "width": 0.0,
        "height": 0.0,
        "volume": 0.0,
        "abmessung": "0.0 x 0.0 x 0.0",
        "error": None,
        "depth_frame": None
    }

    if z_median_mm is None:
        err_msg = "Kein z_median aus Kalibrierung vorhanden. Messung nicht möglich."
        print("Fehler:", err_msg)
        result["error"] = err_msg
        return result

    # Pipeline aufbauen und Tiefenbild abrufen
    pipeline = dai.Pipeline()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # Weil distanz von ca 40 cm
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # Weil distanz von ca 40 cm
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
    config.postProcessing.thresholdFilter.minRange = 100
    config.postProcessing.thresholdFilter.maxRange = 2000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.disparity.link(xout_disparity.input)
    stereo.depth.link(xout_depth.input)

    try:
        with dai.Device(pipeline) as device:
            print("Starte Device für direkte Messung...")
            q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            in_depth = q_depth.get()
            depth_frame = in_depth.getFrame()          # in mm

            # Tiefenframe für eventuelle Visualisierung merken
            result["depth_frame"] = depth_frame

            h, w = depth_frame.shape

            # ROI in der Mitte (wie im Live-Modus)
            cx, cy = w // 2, h // 2
            half = ROI_SIZE // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            # Nur ROI betrachten
            roi_depth = depth_frame[y1:y2, x1:x2].copy()

            # 1. Grundbestimmung INNERHALB DER ROI
            valid_roi = roi_depth[roi_depth > 0]
            if valid_roi.size == 0:
                err_msg = "Keine gültigen Tiefenwerte in der ROI."
                print("Fehler:", err_msg)
                result["error"] = err_msg
                return result
            ground_z = np.min(valid_roi)
            print(f"Grund (min. Tiefe in ROI): {ground_z:.1f} mm")

            # 2. Maske für Objekte über Grund (nur ROI)
            obj_mask_roi = (roi_depth < (z_median_mm - ground_tolerance)) & (roi_depth > 0)

            # 3. Konturen in der ROI finden
            mask_uint8 = obj_mask_roi.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 4. Konturen nach Mindestfläche filtern
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            if not valid_contours:
                err_msg = "Kein Objekt über Grund in der ROI gefunden."
                print("Fehler:", err_msg)
                result["error"] = err_msg
                return result

            # 5. Objektauswahl: Wenn mehrere, nimm das mit der größten Fläche
            if len(valid_contours) > 1:
                print(f"Warnung: {len(valid_contours)} Objekte in ROI. Verwende das größte.")
                selected_contour = max(valid_contours, key=cv2.contourArea)
            else:
                selected_contour = valid_contours[0]

            # Kontur muss auf das Original-Koordinatensystem zurückversetzt werden
            selected_contour = selected_contour + [x1, y1]   # Verschiebung um ROI-Offset

            # 6. Gedrehte Bounding Box
            rect = cv2.minAreaRect(selected_contour)
            (center_x, center_y), (w_px, h_px), angle = rect
            w_px = abs(w_px)
            h_px = abs(h_px)
            print(f"Pixelmaße (gedrehte Box): {w_px:.1f} x {h_px:.1f} px")

            # 7. Tiefenwerte im gesamten Objekt (nicht nur ROI)
            single_mask = np.zeros_like(depth_frame, dtype=np.uint8)
            cv2.drawContours(single_mask, [selected_contour], -1, 1, thickness=cv2.FILLED)
            single_mask = single_mask.astype(bool)
            obj_depth = depth_frame[single_mask]
            valid_obj = obj_depth[(obj_depth > 0) & (obj_depth < 5000)]

            depth_median_mm = None
            width_mm = height_mm = None

            if valid_obj.size > 0:
                depth_median_mm = np.median(valid_obj)
                width_mm, height_mm = compute_object_size_mm((0, 0, w_px, h_px), depth_median_mm, focal_length)
            else:
                print("Keine automatischen Tiefenwerte im Objekt.")
                if mm_per_pixel is not None:
                    print("Fallback: Berechnung mit mm_per_pixel aus Kalibrierung")
                    width_mm = w_px * mm_per_pixel
                    height_mm = h_px * mm_per_pixel
                else:
                    err_msg = "Kein Fallback verfügbar. Keine metrischen Maße."
                    print(err_msg)
                    result["error"] = err_msg
                    return result

            # 8. Höhe über Grund (bezogen auf ROI-Grund)
            hoehe_ueber_grund_obj = z_median_mm - ground_z

            # 9. Volumen berechnen (Annahme Quader) in mm³ -> cm³
            volume_mm3 = height_mm * width_mm * hoehe_ueber_grund_obj
            volume_cm3 = volume_mm3 / 1000.0

            # 10. Ergebnis zusammenstellen
            result["success"] = True
            result["length"] = height_mm      # erste Seite als Länge
            result["width"] = width_mm         # zweite Seite als Breite
            result["height"] = hoehe_ueber_grund_obj
            result["volume"] = volume_cm3
            result["abmessung"] = f"{height_mm:.1f} x {width_mm:.1f} x {hoehe_ueber_grund_obj:.1f}"

            # Ausgabe (optional, kann für Debug bleiben)
            print("\n*** MESSERGEBNISSE (mm) ***")
            print(f"  (Tiefenfilter: Tiefe < {z_median_mm - ground_tolerance:.1f} mm)")
            print(f"\n  Objektmaße (Länge x Breite x Höhe): {height_mm:.1f} x {width_mm:.1f} x {hoehe_ueber_grund_obj:.1f} mm")
            print(f"  Volumen: {volume_cm3:.1f} cm³")

    except Exception as e:
        err_msg = f"Fehler während der Messung: {str(e)}"
        print(err_msg)
        result["error"] = err_msg
        # depth_frame bleibt None oder bereits gesetzt? Im Fehlerfall nicht.
    finally:
        cv2.destroyAllWindows()

    return result

def main():
    """Hauptfunktion, die von aussen aufgerufen wird.
       Rückgabe: Dictionary mit Messergebnissen."""
    print("=== Objektvermessung mit OAK-Kamera ===")
    calib = load_calibration()
    return direct_mode(calib)

if __name__ == "__main__":
    # Wenn direkt ausgeführt, die Messung starten und Ergebnis ausgeben.
    res = main()
    print("\nFINAL RESULT:", res)