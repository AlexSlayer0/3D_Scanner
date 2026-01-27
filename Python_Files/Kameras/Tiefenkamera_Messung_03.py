#!/usr/bin/env python3
#pip3 install depthai opencv-python numpy
#pip install depthai numpy

import cv2
import depthai as dai
import numpy as np

# ===========================================
# FESTE EINSTELLUNGEN (prüfe genau für +-0.5mm)
# ===========================================
KAMERA_AUFLOESUNG = (640, 640)

# Distanz Kamera -> Referenzfläche (mm). Muss exakt gemessen werden.
KAMERA_ABSTAND_REFERENZ_MM = 585.0

# Physische Größe des Messbereichs (bei Referenzdistanz) in mm.
# D.h. bei Abstand KAMERA_ABSTAND_REFERENZ_MM entspricht die ROI-Breite MESSBEREICH_BREITE_MM.
MESSBEREICH_BREITE_MM = 500.0
MESSBEREICH_HOEHE_MM = 500.0

ROI_BREITE = 0.8
ROI_HOEHE = 0.8
ROI_MITTE_X = 0.5
ROI_MITTE_Y = 0.5

OBJEKT_ABSTAND_DELTA_MM = 5  # Objekt muss mind. 5mm näher als Referenz sein

FARBE_MESSBEREICH   = (0, 255, 0)
FARBE_OBJEKT        = (255, 0, 0)
FARBE_TEXT          = (255, 255, 255)

STABIL_FRAMES = 5

# ===========================================
# PIPELINE (unverändert)
# ===========================================
pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)

monoLeftOut = monoLeft.requestOutput(KAMERA_AUFLOESUNG)
monoRightOut = monoRight.requestOutput(KAMERA_AUFLOESUNG)
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

xoutDepth = stereo.depth.createOutputQueue()

# ===========================================
# ROI
# ===========================================
topLeft = dai.Point2f(
    ROI_MITTE_X - ROI_BREITE / 2,
    ROI_MITTE_Y - ROI_HE * 0.5 if False else ROI_MITTE_Y - ROI_HOEHE / 2
)
# (Oben: kleine Korrektur falls Copy-Paste) -> sichere Definition:
topLeft = dai.Point2f(
    ROI_MITTE_X - ROI_BREITE / 2,
    ROI_MITTE_Y - ROI_HOEHE / 2
)
bottomRight = dai.Point2f(
    ROI_MITTE_X + ROI_BREITE / 2,
    ROI_MITTE_Y + ROI_HOEHE / 2
)

roi_rect = dai.Rect(topLeft, bottomRight)

# ===========================================
# HILFSFUNKTIONEN (überarbeitet)
# ===========================================
def berechne_mm_abmessung(depth_roi, box, fx, fy):
    """
    Berechnet Breite und Länge in mm basierend auf jedem Pixel in der Box,
    wobei die Pixel am nächsten zur Kamera als Begrenzung gelten.
    box: 4 Punkte des RotatedRect
    depth_roi: 2D array (ROI)
    fx, fy: focal length in px
    """
    # Maske aus Rotated Rect
    mask = np.zeros_like(depth_roi, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 255, -1)

    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return 0, 0

    depths = depth_roi[ys, xs].astype(np.float32)
    valid = depths > 0
    xs = xs[valid]
    ys = ys[valid]
    depths = depths[valid]

    # Breite: max X - min X in mm, perspektivisch
    breite_mm = np.max((xs - np.min(xs)) / fx * depths)
    # Länge: max Y - min Y in mm, perspektivisch
    laenge_mm = np.max((ys - np.min(ys)) / fy * depths)

    return breite_mm, laenge_mm


def finde_objekt_roi(depth_roi):
    """
    Segmentiert Objekte, die näher sind als die Referenzfläche.
    Liefert robustere Angaben: boundingRect innerhalb ROI, minAreaRect (rotated),
    und median Tiefe in mm (absolut, Sensor-Einheit).
    Koordinaten sind RELATIV zur depth_roi (also 0..roi_w-1, 0..roi_h-1).
    """
    gültig = depth_roi > 0
    if np.count_nonzero(gültig) < 50:
        return None

    objekt_mask = np.zeros_like(depth_roi, np.uint8)
    objekt_mask[
        (depth_roi > 0) &
        (depth_roi < KAMERA_ABSTAND_REFERENZ_MM - OBJEKT_ABSTAND_DELTA_MM)
    ] = 255

    kernel = np.ones((5, 5), np.uint8)
    objekt_mask = cv2.morphologyEx(objekt_mask, cv2.MORPH_OPEN, kernel)
    objekt_mask = cv2.morphologyEx(objekt_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(objekt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 300:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    # Rotated rect (subpixel-robust)
    rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w_px, h_px), angle)
    box = cv2.boxPoints(rect).astype(int)  # relative zu ROI

    # robuste Tiefe: median innerhalb bounding box (nur gültige Werte)
    objekt_tiefen = depth_roi[y:y+h, x:x+w]
    gültige_tiefen = objekt_tiefen[objekt_tiefen > 0]
    if gültige_tiefen.size == 0:
        return None

    median_depth = float(np.median(gültige_tiefen))  # absolute Distanz Kamera->Objekt in mm

    # Falls minDepth gewünscht: min_depth = np.min(gültige_tiefen)
    # Verwende median für Robustheit gegen Ausreißer.

    # Liefere: bounding-rect, rotated rect dims, median depth, box points
    (cx, cy), (w_px, h_px), angle = rect
    return {
        'bbox': (x, y, w, h),
        'rot_rect': (cx, cy, w_px, h_px, angle),
        'box': box,
        'median_depth': median_depth
    }

# ===========================================
# HAUPTPROGRAMM (angepasst)
# ===========================================
letzte_ausgabe = None
stabil_counter = 0

with pipeline:
    pipeline.start()

    while pipeline.isRunning():

        device = dai.Device(pipeline)
        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)
        fx, fy = intrinsics.fx, intrinsics.fy

        depthFrame = xoutDepth.get().getCvFrame()  # uint16 / mm erwartet

        roi_px = roi_rect.denormalize(depthFrame.shape[1], depthFrame.shape[0])
        x1 = int(roi_px.topLeft().x)
        y1 = int(roi_px.topLeft().y)
        x2 = int(roi_px.bottomRight().x)
        y2 = int(roi_px.bottomRight().y)

        # sichere Begrenzung
        x1 = max(0, min(x1, depthFrame.shape[1]-1))
        x2 = max(0, min(x2, depthFrame.shape[1]))
        y1 = max(0, min(y1, depthFrame.shape[0]-1))
        y2 = max(0, min(y2, depthFrame.shape[0]))

        depth_roi = depthFrame[y1:y2, x1:x2]

        objekt = finde_objekt_roi(depth_roi)

        # vis. Aufbereitung (wie vorher)
        roi_w = x2 - x1
        roi_h = y2 - y1

        # depth vis (unschön: clip & colormap)
        roi_valid = depth_roi[depth_roi > 0]
        if roi_valid.size > 0:
            min_d = np.percentile(roi_valid, 5)
            max_d = np.percentile(roi_valid, 95)
        else:
            min_d, max_d = 200, 800

        depth_clipped = np.clip(depthFrame, min_d, max_d)
        depth_vis = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), FARBE_MESSBEREICH, 2)

        if objekt:
            bbox = objekt['bbox']
            box = objekt['box']
            cx, cy, w_px_rect, h_px_rect, angle = objekt['rot_rect']
            depth_mm = objekt['median_depth']  # absolute Distanz Kamera -> Objekt (mm)

            breite_mm, laenge_mm = berechne_mm_abmessung(depth_roi, box, fx, fy)


            # Zeichne Rotated Box (in Frame-Koordinaten)
            box_global = box + np.array([x1, y1])  # offset zur gesamten Frame-Koordinate
            cv2.drawContours(depth_vis, [box_global], 0, FARBE_OBJEKT, 2)

            # Text
            txt = f"Z:{depth_mm:.0f}mm B:{breite_mm:.1f}mm L:{laenge_mm:.1f}mm"
            # Textposition oberhalb der Box (sicher)
            tx = int(min(box_global[:,0]))
            ty = int(min(box_global[:,1]) - 10)
            ty = max(10, ty)
            cv2.putText(depth_vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_TEXT, 1)

            # stabile Terminal-Ausgabe (runde Werte)
            aktuelle_werte = (round(depth_mm,1), round(breite_mm,1), round(laenge_mm,1))
            print(aktuelle_werte)

            if letzte_ausgabe == aktuelle_werte or letzte_ausgabe is None:
                stabil_counter += 1
            else:
                stabil_counter = 1  # reset auf 1 für neuen Wert
            letzte_ausgabe = aktuelle_werte

            if stabil_counter >= STABIL_FRAMES:
                print(
                    f"Objekt erkannt: "
                    f"Distanz={aktuelle_werte[0]} mm, "
                    f"Breite={aktuelle_werte[1]} mm, "
                    f"Länge={aktuelle_werte[2]} mm"
                )
                stabil_counter = 0

        cv2.imshow("Objektmessung 50x50cm", depth_vis)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
