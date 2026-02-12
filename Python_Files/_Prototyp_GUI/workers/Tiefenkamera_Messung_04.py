#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
Optimiert für Raspberry Pi 5 im USB2-Modus.
- PointCloud-Node (sparse) für Geschwindigkeit
- Feste Referenzhöhe 580 mm
- LED-Beleuchtung integriert
- Keine AttributeError mehr
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
    
    # PointCloud: SPARSE = schneller, sicher für USB2
    POINTCLOUD_SPARSE = True        # True = ca. 1000 Punkte, reicht für BoundingBox
    # KEIN setMaxPoints – in DepthAI 2.29.0 nicht verfügbar

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
            time.sleep(1.5)
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
        """Erstellt Pipeline: Mono → Stereo → PointCloud (sparse)"""
        pipeline = dai.Pipeline()
        
        # ---------- Linker & rechter Mono-Sensor ----------
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        
        # ---------- StereoDepth ----------
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setRectifyEdgeFillColor(0)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # ---------- PointCloud – SPARSE (keine maxPoints) ----------
        pointcloud = pipeline.create(dai.node.PointCloud)
        pointcloud.initialConfig.setSparse(Config.POINTCLOUD_SPARSE)
        # setMaxPoints() existiert in DepthAI 2.29.0 NICHT – daher weggelassen
        
        stereo.depth.link(pointcloud.inputDepth)
        
        # ---------- Outputs ----------
        pc_out = pipeline.create(dai.node.XLinkOut)
        pc_out.setStreamName("pointcloud")
        pointcloud.outputPointCloud.link(pc_out.input)
        
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)
        
        self.pipeline = pipeline
        return pipeline
    
    def get_measurement(self) -> Dict:
        """Führt eine vollständige 3D-Messung durch"""
        try:
            control_light(True)
            time.sleep(0.3)
            
            pipeline = self._build_pipeline()
            
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)
                
                depth_data = q_depth.get()
                pc_data = q_pc.get()
                
                # Punktwolke prüfen
                points = pc_data.getPoints()
                if points is None or len(points) == 0:
                    return self._error_result("Keine Punktwolke empfangen")
                
                # Min/Max in X, Y, Z
                min_x = pc_data.getMinX()
                max_x = pc_data.getMaxX()
                min_y = pc_data.getMinY()
                max_y = pc_data.getMaxY()
                min_z = pc_data.getMinZ()       # höchster Punkt
                
                logger.debug(f"Punkte: {len(points)}, Z: {min_z:.1f} - {pc_data.getMaxZ():.1f} mm")
                
                if None in (min_x, max_x, min_y, max_y, min_z):
                    return self._error_result("Ungültige Bounding-Box")
                
                # Dimensionen
                length = max_x - min_x
                width  = max_y - min_y
                if length < width:
                    length, width = width, length
                
                # Höhe = Boden - höchster Punkt
                height = Config.REFERENZ_HOEHE_MM - min_z
                
                # Plausibilität
                if height < Config.MIN_OBJEKT_HOEHE_MM:
                    return self._error_result(f"Kein Objekt (Höhe {height:.1f} mm)")
                height = min(height, Config.MAX_OBJEKT_HOEHE_MM)
                length = min(length, Config.MAX_LAENGE_BREITE_MM)
                width  = min(width, Config.MAX_LAENGE_BREITE_MM)
                
                volume = length * width * height
                
                # Visualisierung
                depth_frame = depth_data.getCvFrame()
                vis_frame = self._create_visualization(depth_frame, length, width, height)
                
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
            control_light(False)
            return self._error_result(str(e))
    
    def _create_visualization(self, depth_frame, length, width, height):
        """Zeigt Messwerte im Tiefenbild an"""
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        
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


def get_volume() -> Dict:
    """Interface-Funktion für den ParallelWorker"""
    measurer = OakD2Volume()
    try:
        return measurer.get_measurement()
    finally:
        measurer.close()


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