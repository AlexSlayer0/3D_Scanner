#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# ===========================================
# FESTE EINSTELLUNGEN
# ===========================================

KAMERA_AUFLOESUNG = (640, 400)

# Bekannte parallele Referenzfläche unter der Kamera
KAMERA_ABSTAND_REFERENZ_MM = 500

MESSBEREICH_BREITE_MM = 500
MESSBEREICH_HOEHE_MM = 500

ROI_BREITE = 0.8
ROI_HOEHE = 0.8
ROI_MITTE_X = 0.5
ROI_MITTE_Y = 0.5

OBJEKT_ABSTAND_DELTA_MM = 10  # Objekt muss mind. 10mm über Referenz liegen

FARBE_MESSBEREICH = (0, 255, 0)
FARBE_OBJEKT = (255, 0, 0)
FARBE_TEXT = (255, 255, 255)

STABIL_FRAMES = 5

# ===========================================
# PIPELINE
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
#stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

xoutDepth = stereo.depth.createOutputQueue()

# ===========================================
# ROI
# ===========================================

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
# HILFSFUNKTIONEN
# ===========================================

def pixel_zu_mm(w_px, h_px, roi_w_px, roi_h_px):
    mm_pro_px_x = MESSBEREICH_BREITE_MM / roi_w_px
    mm_pro_px_y = MESSBEREICH_HOEHE_MM / roi_h_px
    return w_px * mm_pro_px_x, h_px * mm_pro_px_y


def finde_objekt_roi(depth_roi):
    """
    Segmentiert Objekte, die näher sind als die Referenzfläche
    """
    gültig = depth_roi > 0
    if np.count_nonzero(gültig) < 50:
        return None

    # Maske: Objekt liegt über Referenzfläche
    objekt_mask = np.zeros_like(depth_roi, np.uint8)
    objekt_mask[
        (depth_roi > 0) &
        (depth_roi < KAMERA_ABSTAND_REFERENZ_MM - OBJEKT_ABSTAND_DELTA_MM)
    ] = 255

    # Rauschen entfernen
    kernel = np.ones((5, 5), np.uint8)
    objekt_mask = cv2.morphologyEx(objekt_mask, cv2.MORPH_OPEN, kernel)
    objekt_mask = cv2.morphologyEx(objekt_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        objekt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Größte zusammenhängende Fläche = Objekt
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 300:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    objekt_tiefen = depth_roi[y:y+h, x:x+w]
    gültige_tiefen = objekt_tiefen[objekt_tiefen > 0]

    if len(gültige_tiefen) == 0:
        return None

    min_tiefe = np.min(gültige_tiefen)
    hoehe_mm = KAMERA_ABSTAND_REFERENZ_MM - min_tiefe

    return x, y, w, h, hoehe_mm


# ===========================================
# HAUPTPROGRAMM
# ===========================================

letzte_ausgabe = None
stabil_counter = 0

with pipeline:
    pipeline.start()

    while pipeline.isRunning():
        depthFrame = xoutDepth.get().getCvFrame()

        # ROI Pixelkoordinaten
        roi_px = roi_rect.denormalize(
            depthFrame.shape[1], depthFrame.shape[0]
        )

        x1 = int(roi_px.topLeft().x)
        y1 = int(roi_px.topLeft().y)
        x2 = int(roi_px.bottomRight().x)
        y2 = int(roi_px.bottomRight().y)

        depth_roi = depthFrame[y1:y2, x1:x2]

        objekt = finde_objekt_roi(depth_roi)

        # Anzeige vorbereiten
        depth_vis = cv2.normalize(
            depthFrame, None, 255, 0,
            cv2.NORM_INF, cv2.CV_8UC1
        )
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Messbereich zeichnen
        cv2.rectangle(
            depth_vis, (x1, y1), (x2, y2),
            FARBE_MESSBEREICH, 2
        )

        if objekt:
            ox, oy, ow, oh, hoehe_mm = objekt
            ox += x1
            oy += y1

            roi_w = x2 - x1
            roi_h = y2 - y1

            breite_mm, laenge_mm = pixel_zu_mm(ow, oh, roi_w, roi_h)

            # Objekt zeichnen
            cv2.rectangle(
                depth_vis,
                (ox, oy),
                (ox + ow, oy + oh),
                FARBE_OBJEKT, 2
            )

            cv2.putText(
                depth_vis,
                f"H:{hoehe_mm:.0f}mm B:{breite_mm:.0f}mm L:{laenge_mm:.0f}mm",
                (ox, oy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                FARBE_TEXT,
                1
            )

            # ===== stabile Terminal-Ausgabe =====
            aktuelle_werte = (
                round(hoehe_mm, 1),
                round(breite_mm, 1),
                round(laenge_mm, 1)
            )
            print(aktuelle_werte)
            print(stabil_counter)


            if letzte_ausgabe == aktuelle_werte or letzte_ausgabe is None:
                stabil_counter += 1

            if stabil_counter == STABIL_FRAMES:
                print(
                    f"Objekt erkannt: "
                    f"Höhe={aktuelle_werte[0]} mm, "
                    f"Breite={aktuelle_werte[1]} mm, "
                    f"Länge={aktuelle_werte[2]} mm"
                )
                stabil_counter = 0

        cv2.imshow("Objektmessung 50x50cm", depth_vis)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
