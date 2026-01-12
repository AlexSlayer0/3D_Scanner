#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

class VirtualPlane:
    def __init__(self, distance_mm=1000):
        self.distance = distance_mm  # Abstand der virtuellen Ebene in mm
        self.tolerance = 25  # Toleranzbereich in mm
        self.plane_color = (0, 255, 0)  # Grün für die Ebene
        self.object_color = (255, 0, 0)  # Blau für das Objekt
    
    def project_points_to_plane(self, points_3d):
        """Projiziert 3D-Punkte auf die virtuelle Ebene"""
        projected = []
        for point in points_3d:
            if abs(point.z - self.distance) < self.tolerance:
                projected.append(point)
        return projected
    
    def calculate_object_dimensions(self, points_on_plane):
        """Berechnet Abmessungen des Objekts auf der Ebene"""
        if len(points_on_plane) < 2:
            return 0, 0, 0
        
        # Extrahiere X, Y, Z Werte
        x_vals = [p.x for p in points_on_plane]
        y_vals = [p.y for p in points_on_plane]
        z_vals = [p.z for p in points_on_plane]
        
        # Berechne Abmessungen
        width = max(x_vals) - min(x_vals) if x_vals else 0
        height = max(y_vals) - min(y_vals) if y_vals else 0
        depth = max(z_vals) - min(z_vals) if z_vals else 0
        
        return width, height, depth

# Erweiterte Pipeline mit Grid-Detection
pipeline = dai.Pipeline()

# Kameras konfigurieren
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)

# Konfiguration
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# StereoDepth konfigurieren
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(True)

# Spatial Calculator konfigurieren
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN

# Grid-basierte ROIs für detaillierte Messung
grid_size = 5  # 5x5 Grid
roi_size = 0.04  # 4% des Bildes pro ROI
roi_configs = []

for i in range(grid_size):
    for j in range(grid_size):
        roi_config = dai.SpatialLocationCalculatorConfigData()
        roi_config.depthThresholds.lowerThreshold = 100
        roi_config.depthThresholds.upperThreshold = 10000
        roi_config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        
        # ROI-Position im Grid berechnen
        x_min = i * (1.0/grid_size)
        y_min = j * (1.0/grid_size)
        x_max = x_min + roi_size
        y_max = y_min + roi_size
        
        roi_config.roi = dai.Rect(dai.Point2f(x_min, y_min), 
                                   dai.Point2f(x_max, y_max))
        roi_configs.append(roi_config)

# ROIs zur Konfiguration hinzufügen
spatialCalcConfig = dai.SpatialLocationCalculatorConfig()
for cfg in roi_configs:
    spatialCalcConfig.addROI(cfg)

spatialCalc.inputConfig.setWaitForMessage(False)
spatialCalc.initialConfig = spatialCalcConfig

# Verlinkungen
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(spatialCalc.inputDepth)

# Output Queues
xoutSpatial = pipeline.create(dai.node.XLinkOut)
xoutSpatial.setStreamName("spatial")
spatialCalc.out.link(xoutSpatial.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# Virtuelle Ebene
virtual_plane = VirtualPlane(distance_mm=800)  # 800mm entfernte Ebene

# Device starten
with dai.Device(pipeline) as device:
    # Queues
    spatialQueue = device.getOutputQueue(name="spatial", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    # Config Queue für dynamische Änderungen
    configQueue = device.getInputQueue("spatialCalcConfig")
    
    print("Virtual Plane Measurement Active")
    print("Press:")
    print("  'p' +/- : Adjust plane distance")
    print("  't' +/- : Adjust tolerance")
    print("  'a' : Auto-detect object")
    print("  'm' : Manual ROI mode")
    print("  'q' : Quit")
    
    auto_mode = True
    show_grid = True
    
    while True:
        # Daten abrufen
        inDepth = depthQueue.get()
        depthFrame = inDepth.getFrame()
        
        spatialData = spatialQueue.get().getSpatialLocations()
        
        # Tiefenbild für Visualisierung vorbereiten
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        
        # Punkte auf der virtuellen Ebene sammeln
        points_on_plane = []
        
        # Grid anzeigen und Daten sammeln
        if show_grid:
            for idx, depthData in enumerate(spatialData):
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], 
                                     height=depthFrameColor.shape[0])
                
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)
                
                # Prüfe ob Punkt auf virtueller Ebene liegt
                if abs(depthData.spatialCoordinates.z - virtual_plane.distance) < virtual_plane.tolerance:
                    points_on_plane.append(depthData.spatialCoordinates)
                    # Grünes Rechteck für Punkte auf der Ebene
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 
                                virtual_plane.plane_color, 1)
                else:
                    # Blaues Rechteck für Objektpunkte
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 
                                (255, 0, 0), 1)
        
        # Objekt auf der Ebene analysieren
        if points_on_plane and auto_mode:
            # Abmessungen berechnen
            width, height, depth = virtual_plane.calculate_object_dimensions(points_on_plane)
            
            if width > 0 and height > 0:
                # Bounding Box um alle Punkte auf der Ebene
                x_coords = [p.x for p in points_on_plane]
                y_coords = [p.y for p in points_on_plane]
                
                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)
                
                # Skaliere Pixelkoordinaten (annähernd)
                scale_factor = 0.5  # Anpassen basierend auf Kalibrierung
                x1 = int(depthFrameColor.shape[1]/2 + min_x * scale_factor)
                x2 = int(depthFrameColor.shape[1]/2 + max_x * scale_factor)
                y1 = int(depthFrameColor.shape[0]/2 - min_y * scale_factor)
                y2 = int(depthFrameColor.shape[0]/2 - max_y * scale_factor)
                
                # Zeige Objekt-Bounding Box
                cv2.rectangle(depthFrameColor, (x1, y1), (x2, y2), 
                            virtual_plane.object_color, 2)
                
                # Messwerte anzeigen
                info_text = [
                    f"Object Dimensions:",
                    f"Width: {width:.1f} mm",
                    f"Height: {height:.1f} mm",
                    f"Depth: {depth:.1f} mm",
                    f"Plane Distance: {virtual_plane.distance} mm",
                    f"Points on plane: {len(points_on_plane)}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(depthFrameColor, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20
        
        # Virtuelle Ebene als Linie visualisieren
        cv2.line(depthFrameColor, 
                (0, depthFrameColor.shape[0]//2),
                (depthFrameColor.shape[1], depthFrameColor.shape[0]//2),
                virtual_plane.plane_color, 1)
        
        cv2.putText(depthFrameColor, f"Virtual Plane: {virtual_plane.distance}mm",
                   (10, depthFrameColor.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, virtual_plane.plane_color, 1)
        
        # Bild anzeigen
        cv2.imshow("Virtual Plane Measurement", depthFrameColor)
        
        # Tastensteuerung
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Plane Distance anpassen
            adjust = cv2.waitKey(0) & 0xFF
            if adjust == ord('+'):
                virtual_plane.distance += 50
            elif adjust == ord('-'):
                virtual_plane.distance -= 50
            print(f"Plane distance: {virtual_plane.distance}mm")
        elif key == ord('t'):
            # Toleranz anpassen
            adjust = cv2.waitKey(0) & 0xFF
            if adjust == ord('+'):
                virtual_plane.tolerance += 10
            elif adjust == ord('-'):
                virtual_plane.tolerance = max(10, virtual_plane.tolerance - 10)
            print(f"Tolerance: {virtual_plane.tolerance}mm")
        elif key == ord('a'):
            auto_mode = True
            print("Auto-detect mode ON")
        elif key == ord('m'):
            auto_mode = False
            print("Manual mode ON")
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid display: {show_grid}")

cv2.destroyAllWindows()






'''
Tastensteuerung:

p dann +/-: Ebenenabstand anpassen
t dann +/-: Toleranz anpassen
a: Automatische Objekterkennung
m: Manueller Modus
g: Grid ein-/ausblenden
q: Beenden


'''

