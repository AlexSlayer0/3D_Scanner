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
import sys                          # für Kommandozeilenargumente

CALIB_FILE = "distanz_calibration.json"
ROI_SIZE = 250  # Pixel für die zentrierte ROI

# Standardwerte
DEFAULT_COLOR_TOLERANCE = 25      # nicht mehr für Segmentierung genutzt, aber für Kompatibilität
DEFAULT_MIN_AREA = 25
FALLBACK_FOCAL_LENGTH_PX = 580
DEFAULT_GROUND_TOLERANCE_MM = 5
DEFAULT_HUE_TOLERANCE = 5
DEFAULT_OFFSET_X = 20
DEFAULT_OFFSET_Y = 0

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
            "hue_tolerance": DEFAULT_HUE_TOLERANCE,
            "roi_offset_x": DEFAULT_OFFSET_X,
            "roi_offset_y": DEFAULT_OFFSET_Y
        }

    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)

        calib = {
            "COLOR_TOLERANCE": data.get("COLOR_TOLERANCE", DEFAULT_COLOR_TOLERANCE),
            "MIN_AREA": data.get("MIN_AREA", DEFAULT_MIN_AREA),
            "ground_tolerance_mm": data.get("GROUND_TOLERANCE_MM", DEFAULT_GROUND_TOLERANCE_MM),
            "hue_tolerance": data.get("HUE_TOLERANCE", DEFAULT_HUE_TOLERANCE),
            "roi_offset_x": data.get("roi_offset_x", DEFAULT_OFFSET_X),
            "roi_offset_y": data.get("roi_offset_y", DEFAULT_OFFSET_Y)
        }

        z_list = data.get("z_median_liste")
        mm_list = data.get("mm_per_pixel_liste")
        focal_list = data.get("focal_length_pix")
        offset_x = data.get("roi_offset_x", DEFAULT_OFFSET_X)
        offset_y = data.get("roi_offset_y", DEFAULT_OFFSET_Y)

        if z_list and isinstance(z_list, list) and len(z_list) > 0:
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
            calib["mm_per_pixel"] = mm_list[selected]
            calib["focal_length_pix"] = focal_list[selected]
            calib["roi_offset_x"] = data.get("roi_offset_x", 0)
            calib["roi_offset_y"] = data.get("roi_offset_y", 0)

        else:
            # Alte Struktur
            calib["z_median"] = data.get("z_median")
            calib["mm_per_pixel"] = data.get("mm_per_pixel") or data.get("mm_per_pixel_763mm")
            calib["focal_length_pix"] = data.get("focal_length_pix", FALLBACK_FOCAL_LENGTH_PX)

        print("Kalibrierung geladen:")

        print(f"   z_median = {calib['z_median']:.1f} mm")
        print(f"   mm_per_pixel = {calib['mm_per_pixel']:.4f} mm/px")
        print(f"   focal_length_pix = {calib['focal_length_pix']:.1f} px")
        print(f"   COLOR_TOLERANCE = {calib['COLOR_TOLERANCE']}")
        print(f"   MIN_AREA = {calib['MIN_AREA']}")
        print(f"   GROUND_TOLERANCE_MM = {calib['ground_tolerance_mm']} mm")
        print(f"   HUE_TOLERANCE = {calib['hue_tolerance']}")
        print(f"   ROI Offset X = {calib['roi_offset_x']} px, Y = {calib['roi_offset_y']} px")

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
            "hue_tolerance": DEFAULT_HUE_TOLERANCE
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

def compute_object_size_mm(bbox_px, depth_value_mm, focal_length_px):
    w_px, h_px = bbox_px[2], bbox_px[3]
    width_mm = (w_px * depth_value_mm) / focal_length_px
    height_mm = (h_px * depth_value_mm) / focal_length_px
    return width_mm, height_mm

def direct_mode(calib):
    """Einmalige Messung ohne Benutzerinteraktion – nutzt zentrierte ROI."""
    print("\n--- Direkter Messmodus ---")
    z_median_mm = calib["z_median"]
    focal_length = calib["focal_length_pix"]
    mm_per_pixel = calib["mm_per_pixel"]
    min_area = calib["MIN_AREA"]
    ground_tolerance = calib["ground_tolerance_mm"]

    if z_median_mm is None:
        print("Fehler: Kein z_median aus Kalibrierung vorhanden. Messung nicht möglich.")
        return

    # Pipeline aufbauen (identisch)
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
    config.postProcessing.thresholdFilter.minRange = 100
    config.postProcessing.thresholdFilter.maxRange = 2000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.disparity.link(xout_disparity.input)
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        print("Starte Device für direkte Messung...")
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        in_depth = q_depth.get()
        depth_frame = in_depth.getFrame()          # in mm

        h, w = depth_frame.shape

        # ROI in der Mitte (wie im Live-Modus)
        #cx, cy = w // 2, h // 2 Ohne Offset
        #Mit Offset - da Linke Kamera, als Mittelpunkt des ROIs ausgehend, muss der Offset hier positiv sein, um die ROI nach rechts zu verschieben, und negativ für eine Verschiebung nach links
        cx = w // 2 + calib["roi_offset_x"]
        cy = h // 2 + calib["roi_offset_y"]

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
            print("Fehler: Keine gültigen Tiefenwerte in der ROI.")
            return
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
            print("Kein Objekt über Grund in der ROI gefunden.")
            return

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
                print("Kein Fallback verfügbar. Nur Pixelmaße ausgegeben.")

        # 8. Höhe über Grund (bezogen auf ROI-Grund)
        hoehe_ueber_grund_obj = z_median_mm - ground_z

        # 9. Ausgabe
        print("\n*** MESSERGEBNISSE (mm) ***")
        print(f"  (Tiefenfilter: Tiefe < {z_median_mm - ground_tolerance:.1f} mm)")
        if height_mm is not None and width_mm is not None:
            print(f"\n  Objektmaße (Länge x Breite x Höhe): {height_mm:.1f} x {width_mm:.1f} x {hoehe_ueber_grund_obj:.1f} mm")
        else:
            print("\n  Objektmaße (Länge x Breite x Höhe): ---")


def live_mode(calib):
    print("\nLive-Modus gestartet. Drücke:")
    print("   'c' - Objektvermessung durch Klick im blau/weiß Bild")
    print("   'q' - Beenden")
    print("   (Die ROI-Anzeige mit Hoehe über Grund läuft dauerhaft)")

    z_median_mm = calib["z_median"]
    focal_length = calib["focal_length_pix"]
    mm_per_pixel = calib["mm_per_pixel"]
    min_area = calib["MIN_AREA"]                     # wird für Tiefenmaske verwendet
    ground_tolerance = calib["ground_tolerance_mm"]

    # Pipeline aufbauen (unverändert)
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
    config.postProcessing.thresholdFilter.minRange = 100 # 100 mm = 10 cm, um unrealistische Tiefenwerte zu eliminieren
    config.postProcessing.thresholdFilter.maxRange = 2000  # 2000 mm = 2 m, da wir in diesem Szenario keine größeren Entfernungen erwarten
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.disparity.link(xout_disparity.input)
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        print(f"Brennweite (aus Kalibrierung): {focal_length:.1f} px")
        print(f"Tiefentoleranz: {ground_tolerance} mm (Objekte müssen mind. {ground_tolerance} mm über Grund liegen)")

        q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        max_disp = stereo.initialConfig.getMaxDisparity()

        while True:
            in_disparity = q_disparity.get()
            in_depth = q_depth.get()
            disparity_frame = in_disparity.getFrame()
            depth_frame = in_depth.getFrame()          # in mm

            h, w = disparity_frame.shape
            #cx, cy = w // 2, h // 2 #Ohne Offset

            # Mit Offset - da Linke Kamera, als Mittelpunkt des ROIs ausgehend, muss der Offset hier positiv sein, um die ROI nach rechts zu verschieben, und negativ für eine Verschiebung nach links
            cx = w // 2 + calib["roi_offset_x"]
            cy = h // 2 + calib["roi_offset_y"]


            half = ROI_SIZE // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            depth_mm = depth_frame
            roi_depth = depth_mm[y1:y2, x1:x2]
            valid = roi_depth[roi_depth > 0]
            min_z = np.min(valid) if valid.size > 0 else float('nan')

            # Höhe über Grund (ROI)
            if z_median_mm is not None and not np.isnan(min_z):
                hoehe = z_median_mm - min_z
                text_hoehe = f"Hoehe ueber Grund: {hoehe:.1f} mm"
            else:
                text_hoehe = "Hoehe ueber Grund: ---"
            text_minz = f"min_z: {min_z:.1f} mm" if not np.isnan(min_z) else "min_z: ---"

            # ---------- VISUALISIERUNG (blau/weiß) ----------
            vis_img = np.full((h, w, 3), 255, dtype=np.uint8)   # weißer Hintergrund

            # Fadenkreuz in der Mitte
            cv2.line(vis_img, (cx-10, cy), (cx+10, cy), (0,0,0), 1)
            cv2.line(vis_img, (cx, cy-10), (cx, cy+10), (0,0,0), 1)

            obj_mask = None
            if z_median_mm is not None:
                obj_mask = (depth_mm < (z_median_mm - ground_tolerance)) & (depth_mm > 0)
                vis_img[obj_mask] = (255, 0, 0)                 # Blau für Objektkandidaten

            # ROI und Texte einzeichnen
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, text_minz, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(vis_img, text_hoehe, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(vis_img, "Blau: ueber Grund", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow("Live - Objekterkennung", vis_img)

            # Für den Fall, dass wir das Falschfarbenbild doch mal brauchen (wird hier nicht mehr genutzt)
            disp_norm = (disparity_frame * (255.0 / max_disp)).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if obj_mask is None or z_median_mm is None:
                    print("Keine gültige Tiefenmaske vorhanden.")
                    continue

                # Fenster für Mausklick öffnen
                click_img = vis_img.copy()
                cv2.imshow("Objektauswahl - Bitte auf das Objekt klicken", click_img)
                ref_point = None

                def mouse_cb(event, x, y, flags, param):
                    nonlocal ref_point
                    if event == cv2.EVENT_LBUTTONDOWN:
                        ref_point = (x, y)
                        print(f"Klick bei ({x}, {y})")
                        cv2.destroyWindow("Objektauswahl - Bitte auf das Objekt klicken")

                cv2.setMouseCallback("Objektauswahl - Bitte auf das Objekt klicken", mouse_cb)

                while ref_point is None:
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        cv2.destroyWindow("Objektauswahl - Bitte auf das Objekt klicken")
                        break

                if ref_point is None:
                    continue

                x_click, y_click = ref_point
                if not (0 <= x_click < w and 0 <= y_click < h):
                    print("Klick außerhalb des Bildes")
                    continue

                # In der aktuellen obj_mask die Kontur suchen, die den Punkt enthält
                mask_uint8 = obj_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                selected_contour = None
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, (float(x_click), float(y_click)), False) >= 0:
                        selected_contour = cnt
                        break

                if selected_contour is None:
                    print("Kein Objekt an dieser Stelle (Punkt nicht in einer blauen Region)")
                    continue

                # Bounding Box
                x, y, w_px, h_px = cv2.boundingRect(selected_contour)
                if w_px * h_px < min_area:
                    print(f"Objekt zu klein (Fläche = {w_px*h_px} px < {min_area} px).")
                    continue

                # Gefundene Kontur: gedrehte Bounding Box berechnen
                rect = cv2.minAreaRect(selected_contour)
                (center_x, center_y), (w_px, h_px), angle = rect
                w_px = abs(w_px)   # Sicherheitshalber positive Werte
                h_px = abs(h_px)

                # Optional: Seiten sortieren, falls gewünscht (Länge > Breite)
                #if w_px < h_px:
                #    w_px, h_px = h_px, w_px

                # Box für eventuelle Visualisierung (kann später eingefügt werden)
                box = cv2.boxPoints(rect)
                box = np.int8(box)

                # Mindestfläche prüfen (hier wird die Fläche des gedrehten Rechtecks verwendet)
                if w_px * h_px < min_area:
                    print(f"Objekt zu klein (Fläche = {w_px*h_px:.0f} px² < {min_area} px²).")
                    continue

                # Maske für diese Kontur (unverändert)
                single_mask = np.zeros_like(obj_mask, dtype=np.uint8)
                cv2.drawContours(single_mask, [selected_contour], -1, 1, thickness=cv2.FILLED)
                single_mask = single_mask.astype(bool)

                # Tiefenwerte auslesen (gleich)
                obj_depth = depth_mm[single_mask]
                valid_obj = obj_depth[(obj_depth > 0) & (obj_depth < 5000)]

                width_mm = height_mm = None
                depth_median_mm = None
                hoehe_ueber_grund_obj = None

                if valid_obj.size > 0:
                    depth_median_mm = np.median(valid_obj)
                    # Größenberechnung mit den neuen Pixelmaßen
                    width_mm, height_mm = compute_object_size_mm((0, 0, w_px, h_px), depth_median_mm, focal_length)
                else:
                    # Fallback (unverändert)
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
                                width_mm, height_mm = compute_object_size_mm((0, 0, w_px, h_px), depth_manual_mm, focal_length)
                                depth_median_mm = depth_manual_mm
                            except ValueError:
                                print("   Ungültige Eingabe - nur Pixelmaße.")

                # Höhe über Grund (unverändert)
                if not np.isnan(min_z) and z_median_mm is not None:
                    hoehe_ueber_grund_obj = z_median_mm - min_z
                else:
                    hoehe_ueber_grund_obj = None

                # Ausgabe mit neuen Pixelmaßen
                print("\nPixelmaße (gedrehte Box): {:.1f} x {:.1f} px".format(w_px, h_px))
                print("\n*** MESSERGEBNISSE (mm) ***")
                if z_median_mm is not None:
                    print(f"  (Tiefenfilter: Tiefe < {z_median_mm - ground_tolerance:.1f} mm)")

                if height_mm is not None and width_mm is not None and hoehe_ueber_grund_obj is not None:
                    print(f"\n  Objektmaße (Länge x Breite x Höhe): {height_mm:.1f} x {width_mm:.1f} x {hoehe_ueber_grund_obj:.1f} mm")
                else:
                    print("\n   Objektmaße (Länge x Breite x Höhe): ---")

    cv2.destroyAllWindows()

def main():
    print("=== Objektvermessung mit OAK-Kamera ===")
    calib = load_calibration()

    # Prüfen, ob das Kommandozeilenargument "direkt" übergeben wurde
    if len(sys.argv) > 1 and sys.argv[1] == "direkt":
        direct_mode(calib)
    else:
        live_mode(calib)

if __name__ == "__main__":
    main()