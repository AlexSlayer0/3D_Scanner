#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
Optimiert für Raspberry Pi 5 im USB2-Modus.
Verwendet PointCloud-Node für direkte Dimensionsmessung ohne Pixelrechnung.
Referenzebene (Boden) ist fest bei 580 mm.
"""

import cv2
import depthai as dai
import numpy as np
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
    POINTCLOUD_SPARSE = True        # True = schneller, reicht für BoundingBox
    POINTCLOUD_MAX_POINTS = 5000    # Begrenzung für USB2

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
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Für USB2: weniger Last durch Median-Filter
        stereo.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
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
            pipeline = self._build_pipeline()
            
            # WICHTIG: USB2-Modus für Stabilität erzwingen
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)
                
                # Ein Tiefenbild + Punktwolke anfordern
                depth_data = q_depth.get()      # Tiefenbild (für Visualisierung)
                pc_data = q_pc.get()            # PointCloudData-Objekt
                
                # ---------- 3D-Dimensionen aus Punktwolke extrahieren ----------
                # Min/Max in X, Y, Z (in mm)
                min_x = pc_data.getMinX()
                max_x = pc_data.getMaxX()
                min_y = pc_data.getMinY()
                max_y = pc_data.getMaxY()
                min_z = pc_data.getMinZ()       # kleinster Abstand → höchster Punkt!
                
                # Plausibilitätsprüfungen
                if None in (min_x, max_x, min_y, max_y, min_z):
                    return self._error_result("Keine gültige Punktwolke")
                
                # Länge, Breite (in mm)
                length = max_x - min_x
                width  = max_y - min_y
                if length < width:   # Normierung: Länge ≥ Breite
                    length, width = width, length
                
                # Höhe: Boden (580 mm) – min_z (Abstand des höchsten Punkts)
                height = Config.REFERENZ_HOEHE_MM - min_z
                
                # Plausibilitätsgrenzen
                if height < Config.MIN_OBJEKT_HOEHE_MM:
                    return self._error_result("Kein Objekt erkannt (Höhe zu klein)")
                if height > Config.MAX_OBJEKT_HOEHE_MM:
                    height = Config.MAX_OBJEKT_HOEHE_MM
                if length > Config.MAX_LAENGE_BREITE_MM:
                    length = Config.MAX_LAENGE_BREITE_MM
                if width > Config.MAX_LAENGE_BREITE_MM:
                    width = Config.MAX_LAENGE_BREITE_MM
                
                # Volumen (vereinfacht, Quader)
                volume = length * width * height
                
                # ---------- Visualisierung vorbereiten ----------
                depth_frame = depth_data.getCvFrame()  # Roh-Tiefenbild
                vis_frame = self._create_visualization(
                    depth_frame, pc_data, length, width, height
                )
                
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
            return self._error_result(str(e))
    
    def _create_visualization(self, depth_frame, pc_data, length, width, height):
        """Erstellt ein visuelles Feedback mit Bounding Box"""
        # Tiefenbild normalisieren und einfärben
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        
        # 2D-Projektion der Bounding Box (vereinfacht)
        # – sinnvoller wäre eine 3D-Reprojektion, aber für Debug reicht Text
        h, w = depth_frame.shape
        cv2.rectangle(depth_vis, (20, 20), (w-20, h-20), (0, 255, 0), 2)
        
        # Messwerte ins Bild schreiben
        info = [
            f"L: {length:.0f} mm",
            f"B: {width:.0f} mm",
            f"H: {height:.0f} mm",
            f"Vol: {volume/1000:.0f} cm³"
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
    print("🚀 Starte 3D-Volumenmessung (PointCloud, USB2, 580 mm Referenz)")
    result = get_volume()
    
    if result['success']:
        print("\n✅ MESSUNG ERFOLGREICH")
        print(f"   📏 Länge:  {result['length']:.1f} mm")
        print(f"   📐 Breite: {result['width']:.1f} mm")
        print(f"   📏 Höhe:  {result['height']:.1f} mm")
        print(f"   📦 Volumen: {result['volume']:.0f} mm³  ({result['volume']/1000:.1f} cm³)")
        
        if result['depth_frame'] is not None:
            cv2.imshow("Volumenmessung (Bounding Box)", result['depth_frame'])
            print("\nBeliebiges Fenster schließen → Ende")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"\n❌ FEHLER: {result.get('error')}")