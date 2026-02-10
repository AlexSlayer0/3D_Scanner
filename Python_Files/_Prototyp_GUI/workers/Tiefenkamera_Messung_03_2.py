#!/usr/bin/env python3
"""
OAK-D2S Präzisions-3D-Scanner mit:
- Referenzfläche + Würfel-Kalibrierung
- automatische Tiefenskalierung für 58 cm Abstand
- Punktwolke Export (.xyz)
"""

import depthai as dai
import numpy as np
import cv2

# ===========================================
# MANUELLE REFERENZWERTE
# ===========================================

REFERENZ_ABSTAND_MM = 580  # Abstand der Referenzfläche zur Kamera in mm
MESSFELD_BREITE_MM = 490
MESSFELD_HOEHE_MM = 490
REF_WUERFEL_KANTE_MM = float(input("Kantenlänge Referenzwürfel (mm): "))

# ROI-Einstellungen (relative Koordinaten)
ROI_BREITE = 0.8
ROI_HOEHE = 0.8
ROI_MITTE_X = 0.5
ROI_MITTE_Y = 0.5

# ===========================================
# PIPELINE ERSTELLEN
# ===========================================

pipeline = dai.Pipeline()

cam_left = pipeline.create(dai.node.MonoCamera)
cam_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(True)

cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# ===========================================
# ROI-Berechnung
# ===========================================
topLeft = dai.Point2f(ROI_MITTE_X - ROI_BREITE / 2, ROI_MITTE_Y - ROI_HOEHE / 2)
bottomRight = dai.Point2f(ROI_MITTE_X + ROI_BREITE / 2, ROI_MITTE_Y + ROI_HOEHE / 2)
roi_rect = dai.Rect(topLeft, bottomRight)

# ===========================================
# HILFSFUNKTIONEN
# ===========================================

def normalize_depth(frame):
    vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    vis = vis.astype("uint8")
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def find_largest_object(depth_roi, reference_height_mm):
    valid = depth_roi[depth_roi > 0]
    if valid.size < 50:
        return None

    # Maske für Objekte über Referenzfläche
    mask = np.zeros_like(depth_roi, np.uint8)
    mask[(depth_roi > 0) & (depth_roi < reference_height_mm)] = 255

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
    obj_depth = depth_roi[y:y+h, x:x+w]
    valid_depths = obj_depth[obj_depth > 0]
    if len(valid_depths) == 0:
        return None

    z_min = np.min(valid_depths)
    return x, y, w, h, z_min

# ===========================================
# HAUPTPROGRAMM
# ===========================================

with dai.Device(pipeline) as device:
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    scale_factor = None

    print("Lege Referenzwürfel auf die Fläche zur Kalibrierung…")

    while scale_factor is None:
        depth_frame = depth_queue.get().getCvFrame()

        h, w = depth_frame.shape
        roi_px = roi_rect.denormalize(w, h)
        x1, y1 = int(roi_px.topLeft().x), int(roi_px.topLeft().y)
        x2, y2 = int(roi_px.bottomRight().x), int(roi_px.bottomRight().y)
        depth_roi = depth_frame[y1:y2, x1:x2]

        obj = find_largest_object(depth_roi, REFERENZ_ABSTAND_MM)

        vis = normalize_depth(depth_frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

        if obj:
            ox, oy, ow, oh, z_min = obj
            pixel_size = (ow + oh)/2
            scale_factor = REF_WUERFEL_KANTE_MM / pixel_size
            depth_correction = REFERENZ_ABSTAND_MM / np.percentile(depth_roi[depth_roi>0], 90)
            print(f"Kalibrierung abgeschlossen: mm/px = {scale_factor:.4f}, depth_correction = {depth_correction:.4f}")

        cv2.imshow("Kalibrierung", vis)
        if cv2.waitKey(1) == ord('q'):
            exit()

    # ===========================================
    # 3D-Messung
    # ===========================================
    print("Starte präzise 3D-Messung (q = beenden, s = Punktwolke speichern)…")

    while True:
        depth_frame = depth_queue.get().getCvFrame()
        h, w = depth_frame.shape
        roi_px = roi_rect.denormalize(w, h)
        x1, y1 = int(roi_px.topLeft().x), int(roi_px.topLeft().y)
        x2, y2 = int(roi_px.bottomRight().x), int(roi_px.bottomRight().y)
        depth_roi = depth_frame[y1:y2, x1:x2]

        obj = find_largest_object(depth_roi, REFERENZ_ABSTAND_MM)
        vis = normalize_depth(depth_frame)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

        if obj:
            ox, oy, ow, oh, z_min = obj
            cv2.rectangle(vis, (x1+ox, y1+oy), (x1+ox+ow, y1+oy+oh), (255,0,0), 2)
            # Berechne reale Maße
            height_mm = (REFERENZ_ABSTAND_MM - z_min*depth_correction)
            width_mm = ow * scale_factor
            length_mm = oh * scale_factor

            cv2.putText(vis, f"H:{height_mm:.1f} B:{width_mm:.1f} L:{length_mm:.1f}",
                        (x1+ox, y1+oy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Punktwolke erzeugen
            points = []
            for iy in range(0, oh, 2):
                for ix in range(0, ow, 2):
                    z = depth_roi[oy+iy, ox+ix]
                    if z == 0:
                        continue
                    X = ix * scale_factor
                    Y = iy * scale_factor
                    Z = REFERENZ_ABSTAND_MM - z*depth_correction
                    points.append((X,Y,Z))
            points = np.array(points)

        cv2.imshow("3D-Scanner", vis)
        key = cv2.waitKey(1)
        if key == ord('s') and obj:
            np.savetxt("punktwolke.xyz", points)
            print("Punktwolke gespeichert: punktwolke.xyz")
        if key == ord('q'):
            break

cv2.destroyAllWindows()