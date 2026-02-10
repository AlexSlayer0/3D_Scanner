#!/usr/bin/env python3

"""
Präzise Objektmessung mit OAK‑D2S (DepthAI 3.x)
Zielgenauigkeit: ca. ±3–5 mm bei ~58 cm Abstand

Voraussetzungen:
- depthai >= 3.0.0
- opencv-python
- numpy

Installation:
    pip install depthai opencv-python numpy
"""

import depthai as dai
import numpy as np
import cv2

# ============================================================
# FESTE KALIBRIER‑WERTE (anpassen nach Referenzmessung!)
# ============================================================

REFERENZ_ABSTAND_MM = 580          # echter Abstand Kamera → Tisch
MESSFELD_BREITE_MM = 500           # reale Breite des Messfeldes
MESSFELD_HOEHE_MM = 500            # reale Höhe des Messfeldes

MIN_OBJEKT_HOEHE_MM = 5            # Objekt muss mind. 5 mm über Tisch
STABIL_FRAMES = 6                  # Frames für stabile Messung

# ROI in Prozent des Bildes
ROI_W = 0.8
ROI_H = 0.8
ROI_CX = 0.5
ROI_CY = 0.5


# ============================================================
# DEPTHAI PIPELINE (v3 API)
# ============================================================

pipeline = dai.Pipeline()

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)

left.requestOutput((640, 400)).link(stereo.left)
right.requestOutput((640, 400)).link(stereo.right)

stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(True)   # wichtig für mm‑Genauigkeit

# Depth‑Stream direkt nutzbar in v3
depth_stream = stereo.depth


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def roi_pixel_coords(frame):
    """Berechnet ROI‑Pixelkoordinaten."""
    h, w = frame.shape

    rw = int(w * ROI_W)
    rh = int(h * ROI_H)

    x1 = int(w * ROI_CX - rw / 2)
    y1 = int(h * ROI_CY - rh / 2)

    return x1, y1, x1 + rw, y1 + rh


def pixel_to_mm(obj_w_px, obj_h_px, roi_w_px, roi_h_px, depth_mm):
    """
    Umrechnung Pixel → mm unter Nutzung echter Projektionsskalierung.
    Näherung: Maßstab proportional zur Tiefe Z.
    """

    scale_x = (MESSFELD_BREITE_MM / roi_w_px) * (depth_mm / REFERENZ_ABSTAND_MM)
    scale_y = (MESSFELD_HOEHE_MM / roi_h_px) * (depth_mm / REFERENZ_ABSTAND_MM)

    return obj_w_px * scale_x, obj_h_px * scale_y


def detect_object(depth_roi):
    """Segmentiert Objekt über Referenzebene."""

    valid = depth_roi[depth_roi > 0]
    if valid.size < 50:
        return None

    table_depth = np.percentile(valid, 90)

    mask = np.zeros_like(depth_roi, np.uint8)
    mask[(depth_roi > 0) & (depth_roi < table_depth - MIN_OBJEKT_HOEHE_MM)] = 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 300:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    obj_depth = depth_roi[y:y + h, x:x + w]
    obj_valid = obj_depth[obj_depth > 0]

    if obj_valid.size == 0:
        return None

    min_depth = np.min(obj_valid)
    height_mm = table_depth - min_depth

    return x, y, w, h, height_mm, min_depth


# ============================================================
# HAUPTPROGRAMM
# ============================================================

last_values = None
stable_counter = 0

print("Starte OAK‑D2S Messsystem...")

with dai.Device(pipeline) as device:

    while True:
        frame = depth_stream.get().getCvFrame()

        x1, y1, x2, y2 = roi_pixel_coords(frame)
        depth_roi = frame[y1:y2, x1:x2]

        obj = detect_object(depth_roi)

        # Visualisierung
        vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        vis = vis.astype("uint8")
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if obj:
            ox, oy, ow, oh, height_mm, depth_mm = obj
            ox += x1
            oy += y1

            roi_w = x2 - x1
            roi_h = y2 - y1

            width_mm, length_mm = pixel_to_mm(ow, oh, roi_w, roi_h, depth_mm)

            cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)

            text = f"H:{height_mm:.1f}mm B:{width_mm:.1f}mm L:{length_mm:.1f}mm"
            cv2.putText(vis, text, (ox, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            values = (round(height_mm, 1), round(width_mm, 1), round(length_mm, 1))

            if values == last_values or last_values is None:
                stable_counter += 1
            else:
                stable_counter = 0

            last_values = values

            if stable_counter == STABIL_FRAMES:
                print(f"Messung stabil → H={values[0]} mm | B={values[1]} mm | L={values[2]} mm")
                stable_counter = 0

        cv2.imshow("OAK‑D2S Präzisionsmessung", vis)

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
