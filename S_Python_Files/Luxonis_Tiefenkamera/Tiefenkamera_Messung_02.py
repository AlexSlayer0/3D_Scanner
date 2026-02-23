#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
- ROI-begrenzt auf z.B. 500x500 mm
- Kalibrierung des leeren Raums zur Unterdrückung von Bodenrauschen
- Dynamisches Auslesen der intrinsischen Kameraparameter
- Visualisierung mit Objektpunkten und Bounding Box
- Optionale LED-Lichtsteuerung (seriell) für Raspberry Pi
"""

import cv2
import depthai as dai
import numpy as np
import json
import os
import time
from typing import Dict, Optional, Tuple
import logging
import sys

# Versuch, pyserial zu importieren (optional, für Licht)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

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

    # Punktwolken-Dichte (True = schnell, False = dicht)
    POINTCLOUD_SPARSE = True

    # Fallback-Kameraparameter (nur falls Auslesen fehlschlägt)
    FX_FALLBACK = 822.7
    FY_FALLBACK = 822.7
    CX_FALLBACK = 321.5
    CY_FALLBACK = 239.5

    # Kalibrierungsdatei
    CALIB_FILE = "t_calibration.json"

    # Serielle Schnittstelle für LED (auf Windows ignoriert)
    SERIAL_PORT = "/dev/ttyUSB0"   # ggf. anpassen (z.B. COM3 unter Windows, aber dort deaktiviert)
    SERIAL_BAUDRATE = 9600

    x_offset = -587,2 
    y_offset = 313,1

# ===== LICHTSTEUERUNG (nur auf Linux/Pi aktiv) =====
def control_light(state: bool):
    """Schaltet LED ein/aus (nur wenn pyserial verfügbar und Port existiert)."""
    if not SERIAL_AVAILABLE:
        logger.debug("Lichtsteuerung deaktiviert (pyserial nicht installiert)")
        return
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

# ===== KAMERA-KLASSE =====
class OakD2Volume:
    def __init__(self):
        self.pipeline = None
        self.calibration = self.load_calibration()
        self.fx = self.fy = self.cx = self.cy = None  # werden später aus Gerät gelesen

    def _build_pipeline(self):
        """Erstellt die DepthAI-Pipeline mit Mono, Stereo und PointCloud."""
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

    def _get_intrinsics(self, device):
        """Liest die intrinsischen Parameter aus der Kamera-Kalibrierung."""
        try:
            calib = device.readCalibration()
            # Für Mono-Kamera links (CAM_B) – da wir Tiefe aus Stereo verwenden,
            # sind die intrinsischen der Mono-Kameras relevant.
            # Wir nehmen CAM_B, da die Punktwolke in diesem Koordinatensystem vorliegt.
            intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
            self.fx = intrinsics[0][0]
            self.fy = intrinsics[1][1]
            self.cx = intrinsics[0][2]
            self.cy = intrinsics[1][2]
            logger.info(f"Intrinsics geladen: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        except Exception as e:
            logger.warning(f"Konnte intrinsische Parameter nicht auslesen, verwende Fallback: {e}")
            self.fx = Config.FX_FALLBACK
            self.fy = Config.FY_FALLBACK
            self.cx = Config.CX_FALLBACK
            self.cy = Config.CY_FALLBACK

    def capture_pointcloud(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Nimmt eine Punktwolke und ein Tiefenbild auf. Liefert Punkte in mm."""
        pipeline = self._build_pipeline()
        try:
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                if self.fx is None:
                    self._get_intrinsics(device)

                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)

                depth_data = q_depth.get()
                pc_data = q_pc.get()

                depth_frame = depth_data.getCvFrame()
                points = pc_data.getPoints()          # Rohdaten in 0,1 mm?
                
                # --- Debug: Rohdaten anzeigen ---
                print("\n[DEBUG] Rohdaten (erste 5 Punkte):")
                for i in range(min(5, len(points))):
                    print(f"  {points[i]}")
                print(f"[DEBUG] Mittelwert z (roh): {np.mean(points[:,2]):.2f}")

                # Umrechnung von 0,1 mm -> mm
                points = points / 10.0

                print("[DEBUG] Nach Umrechnung (/10) – erste 5 Punkte in mm:")
                for i in range(min(5, len(points))):
                    print(f"  {points[i]}")
                print(f"[DEBUG] Mittelwert z (mm): {np.mean(points[:,2]):.2f}")
                # ---------------------------------

                return depth_frame, points
        except Exception as e:
            logger.error(f"Fehler bei Aufnahme: {e}")
            return None, None

    def calibrate_empty_space(self):
        """Kalibriert den leeren Raum, speichert Bodenstatistik + Offset für die Mitte."""
        print("\n=== KALIBRIERUNG: BITTE OBJEKT ENTFERNEN ===")
        input("Drücken Sie Enter, wenn die Box leer ist...")

        control_light(True)
        time.sleep(0.3)

        depth_frame, points = self.capture_pointcloud()   # points sind jetzt in mm
        control_light(False)

        if points is None:
            print("❌ Fehler bei Aufnahme")
            return False

        # Alle Punkte mit gültiger Tiefe (z > 0)
        valid = points[:, 2] > 0
        if np.sum(valid) < 100:
            print("❌ Zu wenig gültige Tiefenpunkte.")
            return False

        x = points[valid, 0]
        y = points[valid, 1]
        z = points[valid, 2]

        # Debug: Bereich anzeigen
        print(f"\n[DEBUG] x-Bereich: {np.min(x):.1f} .. {np.max(x):.1f} mm")
        print(f"[DEBUG] y-Bereich: {np.min(y):.1f} .. {np.max(y):.1f} mm")
        print(f"[DEBUG] z-Bereich: {np.min(z):.1f} .. {np.max(z):.1f} mm")

        # ROI-Grenzen aus den tatsächlichen Punkten ableiten (mit 20 mm Rand)
        x_min = float(np.min(x)) - 20
        x_max = float(np.max(x)) + 20
        y_min = float(np.min(y)) - 20
        y_max = float(np.max(y)) + 20

        # Mitte der Box berechnen
        x_mitte = (x_min + x_max) / 2
        y_mitte = (y_min + y_max) / 2

        print(f"\n[DEBUG] Berechnete Box-Mitte: x = {x_mitte:.1f} mm, y = {y_mitte:.1f} mm")

        # Statistik des Bodens (z-Werte)
        z_median = float(np.median(z))
        z_std = float(np.std(z))
        z_min = float(np.min(z))
        z_max = float(np.max(z))

        calibration = {
            "z_median": z_median,
            "z_std": z_std,
            "z_min": z_min,
            "z_max": z_max,
            "roi": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max
            },
            "offset": {                     # NEU: Verschiebung zur Mitte
                "x": x_mitte,
                "y": y_mitte
            }
        }

        with open(Config.CALIB_FILE, "w") as f:
            json.dump(calibration, f, indent=2)

        print(f"\n✅ Kalibrierung erfolgreich!")
        print(f"   Bodenmedian: {z_median:.1f} mm")
        print(f"   Standardabweichung: {z_std:.1f} mm")
        print(f"   Spanne: {z_min:.1f} - {z_max:.1f} mm")
        print(f"   ROI x: {x_min:.1f} .. {x_max:.1f} mm")
        print(f"   ROI y: {y_min:.1f} .. {y_max:.1f} mm")
        print(f"   Offset (x, y): ({x_mitte:.1f}, {y_mitte:.1f}) mm")
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
            control_light(True)
            time.sleep(0.3)

            depth_frame, points = self.capture_pointcloud()   # points in mm
            control_light(False)

            if points is None:
                return self._error_result("Keine Punktwolke empfangen")

            # Offset anwenden, falls vorhanden
            if "offset" in self.calibration:
                offset_x = self.calibration["offset"]["x"]
                offset_y = self.calibration["offset"]["y"]
                points[:, 0] -= offset_x
                points[:, 1] -= offset_y
                print(f"\n[DEBUG] Offset angewendet: x - {offset_x:.1f}, y - {offset_y:.1f}")
            else:
                print("[DEBUG] Kein Offset in Kalibrierung – verwende Rohdaten")

            # Jetzt ROI aus Config (fest ±250 mm) verwenden
            x_min_roi = Config.ROI_X_MIN   # -250
            x_max_roi = Config.ROI_X_MAX   #  250
            y_min_roi = Config.ROI_Y_MIN   # -250
            y_max_roi = Config.ROI_Y_MAX   #  250

            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            # Debug: Bereich nach Offset
            print(f"[DEBUG] Nach Offset  x-Bereich: {np.min(x):.1f} .. {np.max(x):.1f} mm")
            print(f"[DEBUG] Nach Offset  y-Bereich: {np.min(y):.1f} .. {np.max(y):.1f} mm")

            mask = (z > 0) & (x >= x_min_roi) & (x <= x_max_roi) & (y >= y_min_roi) & (y <= y_max_roi)

            if np.sum(mask) == 0:
                return self._error_result("Keine Punkte im ROI")

            x_valid = x[mask]
            y_valid = y[mask]
            z_valid = z[mask]

            print(f"[DEBUG] ROI-Punkte: {len(z_valid)}")

            z_median = self.calibration["z_median"]
            z_std = self.calibration["z_std"]
            tolerance = 3 * z_std

            # Boden entfernen
            object_mask = np.abs(z_valid - z_median) > tolerance
            if np.sum(object_mask) < 10:
                return self._error_result("Kein Objekt erkannt (nach Bodenfilterung)")

            x_obj = x_valid[object_mask]
            y_obj = y_valid[object_mask]
            z_obj = z_valid[object_mask]

            logger.info(f"📊 Objektpunkte: {len(z_obj)} von {len(z_valid)} ROI-Punkten")

            # Bounding Box
            min_x = np.min(x_obj)
            max_x = np.max(x_obj)
            min_y = np.min(y_obj)
            max_y = np.max(y_obj)
            min_z = np.min(z_obj)

            length = max_x - min_x
            width  = max_y - min_y
            if length < width:
                length, width = width, length

            height = z_median - min_z
            if height < Config.MIN_OBJEKT_HOEHE_MM:
                return self._error_result(f"Objekt zu flach: {height:.1f} mm")

            height = min(height, Config.MAX_OBJEKT_HOEHE_MM)
            length = min(length, Config.MAX_LAENGE_BREITE_MM)
            width  = min(width, Config.MAX_LAENGE_BREITE_MM)

            volume = length * width * height

            print(f"\n[DEBUG] Gemessene Abmessungen (vor Beschneidung):")
            print(f"  Länge: {max_x - min_x:.1f} mm, Breite: {max_y - min_y:.1f} mm, Höhe: {z_median - min_z:.1f} mm")

            # Visualisierung
            points_roi = np.column_stack((x_valid, y_valid, z_valid))
            vis_frame = self._create_visualization(
                depth_frame, points_roi, object_mask,
                min_x, max_x, min_y, max_y, min_z,
                length, width, height,
                z_median, min_z,
                len(z_valid), np.sum(object_mask)
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
            control_light(False)
            return 0

# ===== SCHNITTSTELLE FÜR DAS HAUPTINTERFACE =====
def get_volume() -> Dict:
    """Hauptfunktion für Interface_v08.py"""
    measurer = OakD2Volume()
    return measurer.get_measurement()

def calibrate():
    """Führt die Kalibrierung durch."""
    measurer = OakD2Volume()
    measurer.calibrate_empty_space()

if __name__ == "__main__":
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