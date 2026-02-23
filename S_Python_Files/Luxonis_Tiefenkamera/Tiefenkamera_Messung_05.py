#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
- ROI-begrenzt auf z.B. 500x500 mm
- Kalibrierung des leeren Raums zur Unterdrückung von Bodenrauschen
- Windows-kompatibel (keine Lichtsteuerung)
"""

import cv2
import depthai as dai
import numpy as np
import json
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class Config:
    # ROI in Weltkoordinaten (mm) – an Ihre Box anpassen!
    ROI_X_MIN = -250
    ROI_X_MAX = 250
    ROI_Y_MIN = -250
    ROI_Y_MAX = 250

    # Höhen-Toleranzen
    MIN_OBJEKT_HOEHE_MM = 5.0
    MAX_OBJEKT_HOEHE_MM = 300.0
    MAX_LAENGE_BREITE_MM = 500.0

    POINTCLOUD_SPARSE = True
    
    # Intrinsische Kameraparameter (geschätzt für OAK-D2, bitte ggf. kalibrieren)
    FX = 822.7   # Brennweite in x-Richtung (Pixel)
    FY = 822.7   # Brennweite in y-Richtung (Pixel)
    CX = 321.5   # Hauptpunkt x (Pixel)
    CY = 239.5   # Hauptpunkt y (Pixel)


    # Kalibrierungsdatei
    CALIB_FILE = "t_calibration.json"

class OakD2Volume:
    def __init__(self):
        self.pipeline = None
        self.calibration = self.load_calibration()

    def _build_pipeline(self):
        pipeline = dai.Pipeline()

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setRectifyEdgeFillColor(0)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        pointcloud = pipeline.create(dai.node.PointCloud)
        pointcloud.initialConfig.setSparse(Config.POINTCLOUD_SPARSE)
        stereo.depth.link(pointcloud.inputDepth)

        pc_out = pipeline.create(dai.node.XLinkOut)
        pc_out.setStreamName("pointcloud")
        pointcloud.outputPointCloud.link(pc_out.input)

        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)

        return pipeline

    def capture_pointcloud(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Nimmt eine Punktwolke und ein Tiefenbild auf."""
        pipeline = self._build_pipeline()
        try:
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)

                depth_data = q_depth.get()
                pc_data = q_pc.get()

                depth_frame = depth_data.getCvFrame()
                points = pc_data.getPoints()  # numpy array (N, 3)
                return depth_frame, points
        except Exception as e:
            logger.error(f"Fehler bei Aufnahme: {e}")
            return None, None

    def calibrate_empty_space(self):
        """Führt eine Kalibrierung ohne Objekt durch und speichert die Bodenstatistik."""
        print("\n=== KALIBRIERUNG: BITTE OBJEKT ENTFERNEN ===")
        input("Drücken Sie Enter, wenn die Box leer ist...")

        depth_frame, points = self.capture_pointcloud()
        if points is None:
            print("❌ Fehler bei Aufnahme")
            return False

        # Punkte im ROI filtern
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        mask = (z > 0) & \
               (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
               (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

        if np.sum(mask) < 100:
            print("❌ Zu wenig Punkte im ROI. ROI prüfen oder Kamera ausrichten.")
            return False

        z_roi = z[mask]

        # Statistik des Bodens (wir nehmen an, dass im leeren Raum nur Boden ist)
        z_median = float(np.median(z_roi))
        z_std = float(np.std(z_roi))
        z_min = float(np.min(z_roi))
        z_max = float(np.max(z_roi))

        calibration = {
            "z_median": z_median,
            "z_std": z_std,
            "z_min": z_min,
            "z_max": z_max,
            "roi": {
                "x_min": Config.ROI_X_MIN,
                "x_max": Config.ROI_X_MAX,
                "y_min": Config.ROI_Y_MIN,
                "y_max": Config.ROI_Y_MAX
            }
        }

        with open(Config.CALIB_FILE, "w") as f:
            json.dump(calibration, f, indent=2)

        print(f"\n✅ Kalibrierung erfolgreich!")
        print(f"   Bodenmedian: {z_median:.1f} mm")
        print(f"   Standardabweichung: {z_std:.1f} mm")
        print(f"   Spanne: {z_min:.1f} - {z_max:.1f} mm")
        return True

    def load_calibration(self):
        """Lädt die Kalibrierungsdaten, falls vorhanden."""
        if os.path.exists(Config.CALIB_FILE):
            with open(Config.CALIB_FILE, "r") as f:
                return json.load(f)
        else:
            return None

    def get_measurement(self) -> Dict:
        """Führt eine Volumenmessung mit Objekt durch."""
        if self.calibration is None:
            return self._error_result("Keine Kalibrierung vorhanden. Bitte zuerst leeren Raum kalibrieren.")

        try:
            depth_frame, points = self.capture_pointcloud()
            if points is None:
                return self._error_result("Keine Punktwolke empfangen")

            # Punkte im ROI und mit gültiger Tiefe
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            mask = (z > 0) & \
                   (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
                   (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

            if np.sum(mask) == 0:
                return self._error_result("Keine Punkte im ROI")

            x_valid = x[mask]
            y_valid = y[mask]
            z_valid = z[mask]

            # Kalibrierungswerte
            z_median = self.calibration["z_median"]
            z_std = self.calibration["z_std"]
            tolerance = 3 * z_std  # z.B. 3 Sigma

            # Bodenpunkte entfernen (alles innerhalb Toleranz um Median)
            object_mask = np.abs(z_valid - z_median) > tolerance

            if np.sum(object_mask) < 10:
                return self._error_result("Kein Objekt erkannt (nach Bodenfilterung)")

            x_obj = x_valid[object_mask]
            y_obj = y_valid[object_mask]
            z_obj = z_valid[object_mask]

            logger.info(f"📊 Objektpunkte: {len(z_obj)} von {len(z_valid)} ROI-Punkten")

            # Bounding Box des Objekts
            min_x = np.min(x_obj)
            max_x = np.max(x_obj)
            min_y = np.min(y_obj)
            max_y = np.max(y_obj)
            min_z = np.min(z_obj)

            # Dimensionen
            length = max_x - min_x
            width  = max_y - min_y
            if length < width:
                length, width = width, length

            # Höhe = Bodenmedian - min_z (da min_z der höchste Punkt ist)
            height = z_median - min_z
            if height < Config.MIN_OBJEKT_HOEHE_MM:
                return self._error_result(f"Objekt zu flach: {height:.1f} mm")

            height = min(height, Config.MAX_OBJEKT_HOEHE_MM)
            length = min(length, Config.MAX_LAENGE_BREITE_MM)
            width  = min(width, Config.MAX_LAENGE_BREITE_MM)

            volume = length * width * height

            # Visualisierung
            # Punkte im ROI (vor der Filterung) als Nx3-Array
            points_roi = np.column_stack((x_valid, y_valid, z_valid))

            vis_frame = self._create_visualization(
                depth_frame,
                points_roi,
                object_mask,          # boolean-Maske: True = Objektpunkt
                length, width, height,
                z_median, min_z,
                len(z_valid),         # Anzahl ROI-Punkte
                np.sum(object_mask)   # Anzahl Objektpunkte
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

    def _create_visualization(self, depth_frame, points_roi, object_mask,
                            length, width, height, z_median, min_z,
                            num_roi_points, num_obj_points):

        # ---------- Tiefenbild farbig ----------
        depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        # ---------- Bild skalieren ----------
        target_w = 900
        scale = target_w / depth_color.shape[1]
        target_h = int(depth_color.shape[0] * scale)
        vis = cv2.resize(depth_color, (target_w, target_h))

        # ---------- ROI halbtransparent ----------
        overlay = vis.copy()
        cv2.rectangle(overlay, (40, 40), (target_w - 40, target_h - 40), (0, 255, 0), -1)
        vis = cv2.addWeighted(overlay, 0.08, vis, 0.92, 0)

        # ---------- Objektpunkte groß & sichtbar ----------
        fx, fy = Config.FX, Config.FY
        cx, cy = Config.CX, Config.CY

        obj_points = points_roi[object_mask]

        for x, y, z in obj_points:
            if z <= 0:
                continue

            u = (x * fx / z) + cx
            v = (y * fy / z) + cy

            if 0 <= u < depth_color.shape[1] and 0 <= v < depth_color.shape[0]:
                u_rs = int(u * scale)
                v_rs = int(v * scale)

                # größerer Punkt mit weißem Rand
                cv2.circle(vis, (u_rs, v_rs), 5, (255, 255, 255), -1)
                cv2.circle(vis, (u_rs, v_rs), 7, (0, 0, 0), 2)

        # ---------- ROI-Rahmen ----------
        cv2.rectangle(vis, (40, 40), (target_w - 40, target_h - 40), (0, 255, 0), 3)
        cv2.putText(vis, "MESSBEREICH", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ---------- Messwerte-Panel ----------
        panel_w = 320
        panel = np.zeros((target_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        volume_cm3 = (length * width * height) / 1000.0

        lines = [
            ("LAENGE",  f"{length:.1f} mm"),
            ("BREITE",  f"{width:.1f} mm"),
            ("HOEHE",   f"{height:.1f} mm"),
            ("VOLUMEN", f"{volume_cm3:.0f} cm3"),
            ("", ""),
            ("ROI Punkte", str(num_roi_points)),
            ("Objektpunkte", str(num_obj_points)),
        ]

        y = 60
        for title, value in lines:
            if title == "":
                y += 20
                continue

            cv2.putText(panel, title, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.putText(panel, value, (20, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y += 70

        # ---------- Tiefenbild + Panel kombinieren ----------
        vis = np.hstack((vis, panel))

        return vis


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
    """Hauptfunktion für Interface_v08.py"""
    measurer = OakD2Volume()
    return measurer.get_measurement()

# ===== KALIBRIERUNGSFUNKTION =====
def calibrate():
    """Führt die Kalibrierung durch (kann separat aufgerufen werden)."""
    measurer = OakD2Volume()
    measurer.calibrate_empty_space()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate()
    else:
        print("\n🚀 Starte 3D-Volumenmessung (ROI 500x500 mm, dynamische Referenz, USB2)")
        print("   (Falls keine Kalibrierung vorhanden, bitte zuerst mit 'calibrate' ausführen)\n")
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