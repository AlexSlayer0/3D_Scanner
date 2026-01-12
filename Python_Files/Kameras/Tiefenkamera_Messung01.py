#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# ===========================================
# FESTE EINSTELLUNGEN
# ===========================================

KAMERA_AUFLOESUNG = (640, 400)
KAMERA_ABSTAND_BODEN_MM = 500
MESSBEREICH_BREITE_MM = 500
MESSBEREICH_HOEHE_MM = 500

ROI_BREITE = 0.8
ROI_HOEHE = 0.8
ROI_MITTE_X = 0.5
ROI_MITTE_Y = 0.5

OBJEKT_GRENZE_MM = 20
FARB_SCHWELLE_MM = 400  # Alles näher als 40cm wird farbig angezeigt

FARBE_MESSBEREICH = (0, 255, 0)
FARBE_OBJEKT = (255, 0, 0)
FARBE_TEXT = (255, 255, 255)

# ===========================================
# PIPELINE SETUP (EXAKT WIE VORHER)
# ===========================================

# Pipeline erstellen (EXAKT WIE IM URSPRÜNGLICHEN FUNKTIONIERENDEN CODE)
pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

# Linking (EXAKT WIE IM URSPRÜNGLICHEN CODE)
monoLeftOut = monoLeft.requestOutput(KAMERA_AUFLOESUNG)
monoRightOut = monoRight.requestOutput(KAMERA_AUFLOESUNG)
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)

# ROI berechnen
topLeft = dai.Point2f(ROI_MITTE_X - ROI_BREITE/2, ROI_MITTE_Y - ROI_HOEHE/2)
bottomRight = dai.Point2f(ROI_MITTE_X + ROI_BREITE/2, ROI_MITTE_Y + ROI_HOEHE/2)

# ROI-Konfiguration
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 600
config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Queues erstellen (EXAKT WIE IM URSPRÜNGLICHEN CODE)
xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()
stereo.depth.link(spatialLocationCalculator.inputDepth)

# ===========================================
# HILFSFUNKTIONEN (NUR DIE NÖTIGSTEN)
# ===========================================

def pixel_zu_mm(pixel_x, pixel_y, bild_breite, bild_hoehe):
    mm_pro_pixel_x = MESSBEREICH_BREITE_MM / (bild_breite * ROI_BREITE)
    mm_pro_pixel_y = MESSBEREICH_HOEHE_MM / (bild_hoehe * ROI_HOEHE)
    roi_offset_x = bild_breite * (1 - ROI_BREITE) / 2
    roi_offset_y = bild_hoehe * (1 - ROI_HOEHE) / 2
    mm_x = (pixel_x - roi_offset_x) * mm_pro_pixel_x
    mm_y = (pixel_y - roi_offset_y) * mm_pro_pixel_y
    return max(0, min(MESSBEREICH_BREITE_MM, mm_x)), max(0, min(MESSBEREICH_HOEHE_MM, mm_y))

def finde_objekte(tiefenbild, boden_tiefe):
    """Einfache Objekterkennung: Alles was näher als Boden - Grenze ist"""
    objekt_schwelle = boden_tiefe - OBJEKT_GRENZE_MM
    
    # Einfache Binärmaske
    objekt_mask = np.where(tiefenbild < objekt_schwelle, 255, 0).astype(np.uint8)
    
    # Konturen finden
    contours, _ = cv2.findContours(objekt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Wenn Konturen gefunden wurden
    if contours:
        # Alle Konturen zu einer großen Kontur kombinieren
        alle_punkte = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(alle_punkte)
        
        # Tiefe in diesem Bereich berechnen
        objekt_bereich = tiefenbild[y:y+h, x:x+w]
        gültige_tiefen = objekt_bereich[objekt_bereich > 0]
        
        if len(gültige_tiefen) > 0:
            durchschnittliche_tiefe = np.mean(gültige_tiefen)
            objekthoehe = KAMERA_ABSTAND_BODEN_MM - durchschnittliche_tiefe
            
            if objekthoehe >= OBJEKT_GRENZE_MM:
                return {
                    'bbox': (x, y, w, h),
                    'tiefe_mm': durchschnittliche_tiefe,
                    'hoehe_mm': objekthoehe
                }
    
    return None

# ===========================================
# HAUPTPROGRAMM (MIT MINIMALEN ÄNDERUNGEN)
# ===========================================

# Starte Pipeline (EXAKT WIE IM URSPRÜNGLICHEN CODE)
with pipeline:
    pipeline.start()
    
    print("=" * 60)
    print("KAMERA VON OBEN - OBJEKTERKENNUNG")
    print("=" * 60)
    print(f"Messbereich: 50x50cm bei {KAMERA_ABSTAND_BODEN_MM/10}cm Höhe")
    print("Tasten: q=Beenden")
    print("=" * 60)
    
    while pipeline.isRunning():
        # Daten holen (EXAKT WIE IM URSPRÜNGLICHEN CODE)
        spatialData = xoutSpatialQueue.get().getSpatialLocations()
        outputDepthImage = outputDepthQueue.get()
        frameDepth = outputDepthImage.getCvFrame()
        
        # 1. SCHWARZ/WEISS BILD ERSTELLEN (wie gewünscht)
        display = np.zeros((frameDepth.shape[0], frameDepth.shape[1], 3), dtype=np.uint8)
        
        # 2. BODENEBENE BESTIMMEN
        boden_tiefe = KAMERA_ABSTAND_BODEN_MM
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=display.shape[1], height=display.shape[0])
            boden_tiefe = depthData.spatialCoordinates.z
            break
        
        # 3. MESSBEREICH ZEICHNEN
        roi_display = config.roi.denormalize(width=display.shape[1], height=display.shape[0])
        x1 = int(roi_display.topLeft().x)
        y1 = int(roi_display.topLeft().y)
        x2 = int(roi_display.bottomRight().x)
        y2 = int(roi_display.bottomRight().y)
        cv2.rectangle(display, (x1, y1), (x2, y2), FARBE_MESSBEREICH, 2)
        
        # 4. OBJEKTE FINDEN (EINFACHE VERSION)
        objekt = finde_objekte(frameDepth, boden_tiefe)
        
        # 5. OBJEKT ZEICHNEN UND MESSEN
        if objekt:
            x, y, w, h = objekt['bbox']
            
            # EIN großes blaues Rechteck um alles
            cv2.rectangle(display, (x, y), (x+w, y+h), FARBE_OBJEKT, 2)
            
            # Maße berechnen
            breite_mm, laenge_mm = pixel_zu_mm(x+w, y+h, display.shape[1], display.shape[0])
            start_x_mm, start_y_mm = pixel_zu_mm(x, y, display.shape[1], display.shape[0])
            tatsaechliche_breite = abs(breite_mm - start_x_mm)
            tatsaechliche_laenge = abs(laenge_mm - start_y_mm)
            
            # Text mit allen Maßen
            hoehe_text = f"Höhe: {objekt['hoehe_mm']/10:.1f}cm"
            breite_text = f"Breite: {tatsaechliche_breite:.0f}mm"
            laenge_text = f"Länge: {tatsaechliche_laenge:.0f}mm"
            
            # Text anzeigen
            cv2.putText(display, hoehe_text, (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_TEXT, 1)
            cv2.putText(display, breite_text, (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_TEXT, 1)
            cv2.putText(display, laenge_text, (x, y+h+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_TEXT, 1)
            
            # Objekt-Nummer
            cv2.putText(display, "Objekt 1", (x, y+h+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_TEXT, 1)
        
        # 6. STATUS ANZEIGEN
        cv2.putText(display, f"Boden: {boden_tiefe/10:.1f}cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FARBE_TEXT, 2)
        
        if objekt:
            cv2.putText(display, f"Objekt erkannt", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FARBE_TEXT, 2)
        else:
            cv2.putText(display, f"Kein Objekt", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FARBE_TEXT, 2)
        
        # 7. LEGENDE
        legend_y = display.shape[0] - 60
        cv2.putText(display, "Grün = Messbereich (50x50cm)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_MESSBEREICH, 1)
        cv2.putText(display, "Blau = Objekt", (10, legend_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, FARBE_OBJEKT, 1)
        
        # 8. BILD ANZEIGEN
        cv2.imshow("Objektmessung 50x50cm", display)
        
        # 9. TASTENSTEUERUNG
        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break

cv2.destroyAllWindows()