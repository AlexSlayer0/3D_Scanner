#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI v3.3.0)
Direkt kompatibel mit Interface_v08.py _run_volume_task()
"""

import cv2
import depthai as dai
import numpy as np
from typing import Dict, Optional, Tuple
import logging

# Logger einrichten
logger = logging.getLogger(__name__)

# ===========================================
# KONFIGURATION - EINFACHE VERSION
# ===========================================

class Config:
    """Einfache Konfiguration für schnelle Integration"""
    # Kameraposition
    REFERENZ_HOEHE_MM = 580.0  # Kameraabstand zur Referenzfläche
    
    # ROI (Region of Interest) - wie in deinem ursprünglichen Code
    ROI_BREITE = 0.8
    ROI_HOEHE = 0.82
    ROI_MITTE_X = 0.5
    ROI_MITTE_Y = 0.5
    
    # Objekterkennung
    MIN_OBJEKT_HOEHE_MM = 10.0
    MESSBEREICH_BREITE_MM = 500
    MESSBEREICH_HOEHE_MM = 500
    
    # Kamerakalibrierung (Standardwerte für OAK-D2S)
    FX = 822.7  # Brennweite x
    FY = 822.7  # Brennweite y
    CX = 321.5  # Hauptpunkt x
    CY = 239.5  # Hauptpunkt y

# ===========================================
# KERN-FUNKTIONEN FÜR 3D-MESSUNG
# ===========================================

class VolumeCalculator:
    """Berechnet Volumen aus 3D-Punktwolken"""
    
    @staticmethod
    def pixel_to_3d(u: int, v: int, depth_mm: float) -> Optional[np.ndarray]:
        """Konvertiert Pixel zu 3D-Weltkoordinaten"""
        if depth_mm <= 0:
            return None
            
        Z = depth_mm
        X = (u - Config.CX) * Z / Config.FX
        Y = (v - Config.CY) * Z / Config.FY
        
        return np.array([X, Y, Z])
    
    @staticmethod
    def calculate_3d_dimensions(points_3d: np.ndarray) -> Dict:
        """Berechnet 3D-Bounding Box aus Punktwolke"""
        if len(points_3d) < 10:
            return None
        
        min_vals = np.min(points_3d, axis=0)
        max_vals = np.max(points_3d, axis=0)
        
        # Echte 3D-Dimensionen (in mm!)
        length = max_vals[0] - min_vals[0]  # X-Richtung
        width = max_vals[1] - min_vals[1]   # Y-Richtung
        height = max_vals[2] - min_vals[2]  # Z-Richtung (HÖHE!)
        
        # Sicherstellen: length >= width
        if length < width:
            length, width = width, length
        
        volume = length * width * height
        
        return {
            'length': length,
            'width': width,
            'height': height,
            'volume': volume
        }

# ===========================================
# OAK-D2 KAMERA-STEUERUNG
# ===========================================

class OakD2Camera:
    """Einfache Steuerung der OAK-D2 Kamera"""
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Einfache Pipeline für DepthAI"""
        self.pipeline = dai.Pipeline()
        
        # Monokameras
        mono_left = self.pipeline.create(dai.MonoCamera)
        mono_right = self.pipeline.create(dai.MonoCamera)
        
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        
        # StereoDepth
        stereo = self.pipeline.create(dai.StereoDepth)
        stereo.setDefaultProfilePreset(dai.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        
        # Verbindungen
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # Output
        xout_depth = self.pipeline.create(dai.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Holt ein Tiefenbild von der Kamera"""
        try:
            if self.device is None:
                self.device = dai.Device(self.pipeline)
            
            depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=True)
            depth_packet = depth_queue.get()
            return depth_packet.getCvFrame()
            
        except Exception as e:
            logger.error(f"Fehler bei Tiefenbildaufnahme: {e}")
            return None
    
    def close(self):
        """Schließt die Kameraverbindung"""
        if self.device:
            self.device.close()

# ===========================================
# HAUPTFUNKTION FÜR INTERFACE_V08.PY
# ===========================================

def get_volume() -> Dict:
    """
    Hauptfunktion für Interface_v08.py - _run_volume_task()
    
    Returns:
        Dictionary im erwarteten Format:
        {
            'success': bool,
            'length': float,    # in mm
            'width': float,     # in mm
            'height': float,    # in mm
            'volume': float,    # in mm³
            'depth_frame': np.ndarray,  # Visualisiertes Bild
            'error': str        # Nur bei Fehlern
        }
    """
    camera = None
    try:
        # 1. Kamera initialisieren
        camera = OakD2Camera()
        
        # 2. Tiefenbild aufnehmen
        depth_frame = camera.get_depth_frame()
        if depth_frame is None:
            return {
                'success': False,
                'length': 0.0,
                'width': 0.0,
                'height': 0.0,
                'volume': 0.0,
                'depth_frame': None,
                'error': 'Kein Tiefenbild empfangen'
            }
        
        # 3. ROI berechnen (wie in deinem Originalcode)
        height, width = depth_frame.shape
        roi_w = int(Config.ROI_BREITE * width)
        roi_h = int(Config.ROI_HOEHE * height)
        roi_x = int((width - roi_w) / 2)
        roi_y = int((height - roi_h) / 2)
        
        x1, y1 = roi_x, roi_y
        x2, y2 = roi_x + roi_w, roi_y + roi_h
        
        depth_roi = depth_frame[y1:y2, x1:x2]
        
        # 4. Referenzebene finden
        valid_depths = depth_roi[depth_roi > 0]
        if len(valid_depths) < 100:
            return {
                'success': False,
                'length': 0.0,
                'width': 0.0,
                'height': 0.0,
                'volume': 0.0,
                'depth_frame': None,
                'error': 'Keine Referenzebene erkannt'
            }
        
        referenz_tiefe = np.median(valid_depths)
        
        # 5. Objekt finden (vereinfachte Version)
        objekt_mask = np.zeros_like(depth_roi, dtype=np.uint8)
        objekt_pixel = (depth_roi > 0) & (depth_roi < referenz_tiefe - Config.MIN_OBJEKT_HOEHE_MM)
        
        if np.sum(objekt_pixel) < 100:
            return {
                'success': False,
                'length': 0.0,
                'width': 0.0,
                'height': 0.0,
                'volume': 0.0,
                'depth_frame': None,
                'error': 'Kein Objekt erkannt'
            }
        
        objekt_mask[objekt_pixel] = 255
        
        # 6. 3D-Punkte extrahieren
        points_3d = []
        for v in range(0, depth_roi.shape[0], 3):  # Jeden 3. Pixel für Geschwindigkeit
            for u in range(0, depth_roi.shape[1], 3):
                if objekt_mask[v, u] > 0:
                    depth = depth_roi[v, u]
                    if depth > 0:
                        point = VolumeCalculator.pixel_to_3d(
                            u + x1, v + y1, depth
                        )
                        if point is not None:
                            points_3d.append(point)
        
        if len(points_3d) < 50:
            return {
                'success': False,
                'length': 0.0,
                'width': 0.0,
                'height': 0.0,
                'volume': 0.0,
                'depth_frame': None,
                'error': 'Zu wenige 3D-Punkte'
            }
        
        points_array = np.array(points_3d)
        
        # 7. 3D-Dimensionen berechnen
        dimensions = VolumeCalculator.calculate_3d_dimensions(points_array)
        
        if dimensions is None:
            return {
                'success': False,
                'length': 0.0,
                'width': 0.0,
                'height': 0.0,
                'volume': 0.0,
                'depth_frame': None,
                'error': '3D-Berechnung fehlgeschlagen'
            }
        
        # 8. Visualisierung erstellen
        vis_frame = _create_visualization(
            depth_frame, x1, y1, x2, y2, 
            objekt_mask, dimensions
        )
        
        # 9. Ergebnis zurückgeben (EXAKTES Format für _run_volume_task)
        return {
            'success': True,
            'length': round(dimensions['length'], 1),
            'width': round(dimensions['width'], 1),
            'height': round(dimensions['height'], 1),
            'volume': round(dimensions['volume'], 1),
            'depth_frame': vis_frame,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Volumenmessungsfehler: {e}")
        return {
            'success': False,
            'length': 0.0,
            'width': 0.0,
            'height': 0.0,
            'volume': 0.0,
            'depth_frame': None,
            'error': str(e)
        }
        
    finally:
        if camera:
            camera.close()

def _create_visualization(depth_frame, x1, y1, x2, y2, mask, dimensions):
    """Erstellt visualisiertes Tiefenbild"""
    # Normalisieren für Anzeige
    depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    
    # ROI zeichnen
    cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Objektkontur zeichnen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_global = contour + np.array([[x1, y1]])
        cv2.drawContours(depth_vis, [contour_global], -1, (255, 0, 0), 2)
    
    # Textinfo
    info = f"L:{dimensions['length']:.0f} W:{dimensions['width']:.0f} H:{dimensions['height']:.0f}mm"
    cv2.putText(depth_vis, info, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return depth_vis

# ===========================================
# TESTFUNKTION
# ===========================================

if __name__ == "__main__":
    print("Teste Volumenmessung...")
    result = get_volume()
    
    if result['success']:
        print(f"✅ Erfolg!")
        print(f"   Länge: {result['length']} mm")
        print(f"   Breite: {result['width']} mm")
        print(f"   Höhe: {result['height']} mm")
        print(f"   Volumen: {result['volume']:.0f} mm³ ({result['volume']/1000:.1f} cm³)")
        
        if result['depth_frame'] is not None:
            cv2.imshow("Volumenmessung", result['depth_frame'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"Fehler: {result.get('error', 'Unbekannt')}")
