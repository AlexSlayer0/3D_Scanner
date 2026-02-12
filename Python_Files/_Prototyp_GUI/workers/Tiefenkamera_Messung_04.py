#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
Optimiert für Raspberry Pi 5 im USB2-Modus.
- PointCloud-Node für direkte Dimensionsmessung
- Feste Referenzhöhe 580 mm (Bodenplatte)
- LED-Beleuchtung für bessere Tiefenqualität
- Keine DeprecationWarnings mehr
"""

import cv2
import depthai as dai
import numpy as np
import serial
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ===========================================
# KONFIGURATION (alles in mm)
# ===========================================
class Config:
    REFERENZ_HOEHE_MM = 580.0       # Abstand Kamera → Bodenplatte
    MIN_OBJEKT_HOEHE_MM = 5.0       # Rauschen unterdrücken
    MAX_OBJEKT_HOEHE_MM = 300.0     # Maximal erwartete Objekthöhe
    MAX_LAENGE_BREITE_MM = 500.0    # Begrenzung für Plausibilität
    
    # PointCloud-Einstellungen (USB2‑tauglich)
    POINTCLOUD_SPARSE = False       # False = dichte Punktwolke (genauer)
    POINTCLOUD_MAX_POINTS = 3000    # Begrenzung für USB2

    # Serielle Schnittstelle für Beleuchtung
    SERIAL_PORT = "/dev/ttyUSB0"    # Ggf. anpassen: /dev/ttyACM0
    SERIAL_BAUDRATE = 9600

# ===========================================
# LED-STEUERUNG (SERIELL)
# ===========================================
def control_light(state: bool):
    """Schaltet die LED-Strips ein/aus"""
    try:
        with serial.Serial(Config.SERIAL_PORT, Config.SERIAL_BAUDRATE, timeout=1) as ser:
            time.sleep(1.5)  # Port-Initialisierung
            ser.write(b"Change\n")
            time.sleep(0.1)
            ser.write(b"a\n" if state else b"0\n")
            time.sleep(0.2)
        logger.info(f"Licht {'EIN' if state else 'AUS'}")
    except Exception as e:
        logger.warning(f"Lichtsteuerung fehlgeschlagen: {e}")

# ===========================================
# KAMERA-PIPELINE MIT POINTCLOUD
# ===========================================
class OakD2Volume:
    def __init__(self):
        self.pipeline = None
        self.device = None
        
    def _build_pipeline(self):
        """Erstellt Pipeline: Mono → Stereo → PointCloud"""
        pipeline = dai.Pipeline()
        
        # ---------- Linker & rechter Mono-Sensor ----------
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        
        # ---------- StereoDepth für Disparität/Tiefe ----------
        stereo = pipeline.create(dai.node.StereoDepth)
        # NEU: DEFAULT statt deprecated HIGH_DENSITY
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        # NEU: Median-Filter über initialConfig
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setRectifyEdgeFillColor(0)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # ---------- PointCloud aus Tiefe ----------
        pointcloud = pipeline.create(dai.node.PointCloud)
        pointcloud.initialConfig.setSparse(Config.POINTCLOUD_SPARSE)
        if not Config.POINTCLOUD_SPARSE:
            pointcloud.initialConfig.setMaxPoints(Config.POINTCLOUD_MAX_POINTS)
        
        stereo.depth.link(pointcloud.inputDepth)
        
        # ---------- Output für Punktwolke ----------
        pc_out = pipeline.create(dai.node.XLinkOut)
        pc_out.setStreamName("pointcloud")
        pointcloud.outputPointCloud.link(pc_out.input)
        
        # ---------- Optional: Tiefenbild für Visualisierung ----------
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)
        
        self.pipeline = pipeline
        return pipeline
    
    def get_measurement(self) -> Dict:
        """
        Nimmt ein Tiefenbild auf, generiert Punktwolke,
        berechnet Bounding Box + Volumen.
        """
        try:
            # Licht EIN für bessere Tiefenqualität
            control_light(True)
            time.sleep(0.3)
            
            pipeline = self._build_pipeline()
            
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)
                
                # Ein Tiefenbild + Punktwolke anfordern
                depth_data = q_depth.get()
                pc_data = q_pc.get()
                
                # ---------- Prüfen, ob Punktwolke gültig ----------
                points = pc_data.getPoints()
                if points is None or len(points) == 0:
                    return self._error_result("Keine Punktwolke empfangen")
                
                # ---------- 3D-Dimensionen aus Punktwolke extrahieren ----------
                # Min/Max in X, Y, Z (in mm)
                min_x = pc_data.getMinX()
                max_x = pc_data.getMaxX()
                min_y = pc_data.getMinY()
                max_y = pc_data.getMaxY()
                min_z = pc_data.getMinZ()       # kleinster Abstand → höchster Punkt!
                
                # Debug-Ausgabe für Fehlersuche
                logger.debug(f"Punktwolke: {len(points)} Punkte")
                logger.debug(f"Z-Bereich: {min_z:.1f} - {pc_data.getMaxZ():.1f} mm")
                
                # Plausibilitätsprüfungen
                if None in (min_x, max_x, min_y, max_y, min_z):
                    return self._error_result("Ungültige Bounding-Box-Werte")
                
                # Länge, Breite (in mm)
                length = max_x - min_x
                width  = max_y - min_y
                if length < width:   # Normierung: Länge ≥ Breite
                    length, width = width, length
                
                # Höhe: Boden (580 mm) – min_z (Abstand des höchsten Punkts)
                height = Config.REFERENZ_HOEHE_MM - min_z
                
                # Plausibilitätsgrenzen
                if height < Config.MIN_OBJEKT_HOEHE_MM:
                    return self._error_result(f"Kein Objekt erkannt (Höhe {height:.1f} mm < {Config.MIN_OBJEKT_HOEHE_MM} mm)")
                if height > Config.MAX_OBJEKT_HOEHE_MM:
                    height = Config.MAX_OBJEKT_HOEHE_MM
                if length > Config.MAX_LAENGE_BREITE_MM:
                    length = Config.MAX_LAENGE_BREITE_MM
                if width > Config.MAX_LAENGE_BREITE_MM:
                    width = Config.MAX_LAENGE_BREITE_MM
                
                # Volumen (vereinfacht, Quader)
                volume = length * width * height
                
                # ---------- Visualisierung vorbereiten ----------
                depth_frame = depth_data.getCvFrame()
                vis_frame = self._create_visualization(depth_frame, pc_data, length, width, height)
                
                # Licht AUS
                control_light(False)
                
                return {
                    'success': True,
                    'length': round(length, 1),
                    'width': round(width, 1),
                    'height': round(height, 1),
                    'volume': round(volume, 1),
                    'depth_frame': vis_frame,
                    'error': None
                }
                
        except Exception as e:
            logger.error(f"Volumenmessung fehlgeschlagen: {e}", exc_info=True)
            control_light(False)  # Licht auch im Fehlerfall ausschalten
            return self._error_result(str(e))
    
    def _create_visualization(self, depth_frame, pc_data, length, width, height):
        """Erstellt visuelles Feedback mit Messwerten"""
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        
        # Messwerte ins Bild schreiben
        volume_cm3 = (length * width * height) / 1000
        info = [
            f"Laenge: {length:.0f} mm",
            f"Breite: {width:.0f} mm",
            f"Hoehe:  {height:.0f} mm",
            f"Volumen: {volume_cm3:.0f} cm³"
        ]
        y0 = 60
        for i, line in enumerate(info):
            y = y0 + i * 40
            cv2.putText(depth_vis, line, (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return depth_vis
    
    def _error_result(self, msg):
        """Einheitliches Fehler-Dictionary"""
        return {
            'success': False,
            'length': 0.0,
            'width': 0.0,
            'height': 0.0,
            'volume': 0.0,
            'depth_frame': None,
            'error': msg
        }
    
    def close(self):
        if self.device:
            self.device.close()
            self.device = None

# ===========================================
# HAUPTFUNKTION FÜR INTERFACE_V08.PY
# ===========================================
def get_volume() -> Dict:
    """
    Wird von ParallelWorker._run_volume_task() aufgerufen.
    Liefert Abmessungen & Volumen in mm / mm³.
    """
    measurer = OakD2Volume()
    try:
        result = measurer.get_measurement()
        return result
    finally:
        measurer.close()

# ===========================================
# DIREKTTEST (WENN SKRIPT AUSGEFÜHRT WIRD)
# ===========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starte 3D-Volumenmessung (PointCloud, USB2, 580 mm Referenz)")
    result = get_volume()
    
    if result['success']:
        print("\n✅ MESSUNG ERFOLGREICH")
        print(f"   📏 Länge:  {result['length']:.1f} mm")
        print(f"   📐 Breite: {result['width']:.1f} mm")
        print(f"   📏 Höhe:  {result['height']:.1f} mm")
        print(f"   📦 Volumen: {result['volume']:.0f} mm³  ({result['volume']/1000:.1f} cm³)")
        
        if result['depth_frame'] is not None:
            cv2.imshow("Volumenmessung", result['depth_frame'])
            print("\nFenster schließen → Ende")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"\n❌ FEHLER: {result.get('error')}")