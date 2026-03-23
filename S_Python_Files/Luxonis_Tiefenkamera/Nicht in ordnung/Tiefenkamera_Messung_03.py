#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
- Windows-kompatibel (Lichtsteuerung deaktiviert)
- ROI auf 500 mm × 500 mm beschränkt (anpassbar)
- Filterung der Punktwolke auf diesen Bereich
"""

import cv2
import depthai as dai
import numpy as np
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    # Messbereich in Weltkoordinaten (mm) – an Ihre Box anpassen!
    ROI_X_MIN = -250      # linker Rand der Box (in mm)
    ROI_X_MAX = 250       # rechter Rand
    ROI_Y_MIN = -250      # unterer Rand (in mm) – je nach Ausrichtung
    ROI_Y_MAX = 250       # oberer Rand

    # Höhen-Toleranzen
    MIN_OBJEKT_HOEHE_MM = 5.0
    MAX_OBJEKT_HOEHE_MM = 300.0
    MAX_LAENGE_BREITE_MM = 500.0   # Box maximal 500 mm

    POINTCLOUD_SPARSE = True        # Schnell, reicht für BoundingBox

    # Keine serielle Schnittstelle unter Windows – Lichtsteuerung auskommentiert

# ===== KAMERA =====
class OakD2Volume:
    def __init__(self):
        self.pipeline = None

    def _build_pipeline(self):
        pipeline = dai.Pipeline()

        # Mono-Kameras
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        # StereoDepth
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setRectifyEdgeFillColor(0)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # PointCloud (sparse)
        pointcloud = pipeline.create(dai.node.PointCloud)
        pointcloud.initialConfig.setSparse(Config.POINTCLOUD_SPARSE)
        stereo.depth.link(pointcloud.inputDepth)

        # Outputs
        pc_out = pipeline.create(dai.node.XLinkOut)
        pc_out.setStreamName("pointcloud")
        pointcloud.outputPointCloud.link(pc_out.input)

        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)

        return pipeline

    def get_measurement(self) -> Dict:
        try:
            # Lichtsteuerung unter Windows deaktiviert (kein Fehler mehr)
            # control_light(True) auskommentiert

            pipeline = self._build_pipeline()

            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)

                depth_data = q_depth.get()
                pc_data = q_pc.get()

                # ===== TIEFENBILD =====
                depth_frame = depth_data.getCvFrame()
                valid_depths = depth_frame[depth_frame > 0]
                if len(valid_depths) == 0:
                    return self._error_result("Keine gültigen Tiefenwerte")

                # Dynamische Referenz: Median der unteren Bildhälfte (wo der Boden ist)
                h, w = depth_frame.shape
                bottom_half = depth_frame[h//2:, :]  # untere 50%
                bottom_valid = bottom_half[bottom_half > 0]
                if len(bottom_valid) == 0:
                    referenz_mm = float(np.median(valid_depths))  # Fallback
                else:
                    referenz_mm = float(np.median(bottom_valid))
                logger.info(f"📏 Referenzhöhe (Boden): {referenz_mm:.1f} mm")

                # ===== PUNKTWOLKE =====
                points = pc_data.getPoints()  # numpy array (N, 3)
                if points is None or len(points) == 0:
                    return self._error_result("Keine Punktwolke empfangen")

                # Extrahieren der Koordinaten (mm)
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                z_coords = points[:, 2]

                # Nur Punkte mit gültiger Tiefe (>0)
                valid_mask = (z_coords > 0) & (z_coords < referenz_mm + 100)  # nur bis knapp unter Referenz
                valid_mask &= (x_coords > Config.ROI_X_MIN) & (x_coords < Config.ROI_X_MAX)
                valid_mask &= (y_coords > Config.ROI_Y_MIN) & (y_coords < Config.ROI_Y_MAX)

                if np.sum(valid_mask) < 10:
                    return self._error_result("Keine Punkte im Messbereich")

                x_valid = x_coords[valid_mask]
                y_valid = y_coords[valid_mask]
                z_valid = z_coords[valid_mask]

                logger.info(f"📊 Punktwolke: {len(z_valid)} gültige Punkte im ROI")
                logger.info(f"   Z: min={np.min(z_valid):.1f}, max={np.max(z_valid):.1f}, mean={np.mean(z_valid):.1f} mm")

                # ===== BOUNDING BOX =====
                min_x = np.min(x_valid)
                max_x = np.max(x_valid)
                min_y = np.min(y_valid)
                max_y = np.max(y_valid)
                min_z = np.min(z_valid)   # kleinster Z = höchster Punkt (näher an Kamera)
                max_z = np.max(z_valid)

                logger.info(f"   X: {min_x:.1f} .. {max_x:.1f} mm")
                logger.info(f"   Y: {min_y:.1f} .. {max_y:.1f} mm")
                logger.info(f"   Z: {min_z:.1f} .. {max_z:.1f} mm")

                # ===== DIMENSIONEN =====
                length = max_x - min_x
                width  = max_y - min_y
                if length < width:
                    length, width = width, length

                height = referenz_mm - min_z
                if height < Config.MIN_OBJEKT_HOEHE_MM:
                    return self._error_result(f"Objekt zu flach: {height:.1f} mm")

                height = min(height, Config.MAX_OBJEKT_HOEHE_MM)
                length = min(length, Config.MAX_LAENGE_BREITE_MM)
                width  = min(width, Config.MAX_LAENGE_BREITE_MM)

                volume = length * width * height

                # ===== VISUALISIERUNG =====
                vis_frame = self._create_visualization(depth_frame, length, width, height, referenz_mm, min_z)

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

    def _create_visualization(self, depth_frame, length, width, height, referenz, min_z):
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        volume_cm3 = (length * width * height) / 1000
        info = [
            f"Boden: {referenz:.0f} mm",
            f"Objekt-Höhe: {min_z:.0f} mm (Kamera)",
            f"Maße: {length:.0f} x {width:.0f} x {height:.0f} mm",
            f"Volumen: {volume_cm3:.0f} cm³"
        ]
        y0 = 40
        for i, line in enumerate(info):
            y = y0 + i * 35
            cv2.putText(depth_vis, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

def get_volume() -> Dict:
    measurer = OakD2Volume()
    try:
        return measurer.get_measurement()
    finally:
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("\n🚀 Starte 3D-Volumenmessung (ROI 500x500 mm, dynamische Referenz, USB2)\n")
    result = get_volume()

    if result['success']:
        print("\n✅ MESSUNG ERFOLGREICH")
        print(f"   📏 Länge:  {result['length']:.1f} mm")
        print(f"   📐 Breite: {result['width']:.1f} mm")
        print(f"   📏 Höhe:  {result['height']:.1f} mm")
        print(f"   📦 Volumen: {result['volume']:.0f} mm³ ({result['volume']/1000:.1f} cm³)")

        if result['depth_frame'] is not None:
            cv2.imshow("Volumenmessung", result['depth_frame'])
            print("\n🔲 Fenster schließen → Ende")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"\n❌ FEHLER: {result.get('error')}")