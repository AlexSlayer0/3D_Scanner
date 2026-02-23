#!/usr/bin/env python3
"""
3D-Volumenmessung mit OAK-D2 (DepthAI 2.29.0)
- ROI-begrenzt auf 500x500 mm
- Kalibrierung des leeren Raums zur Unterdrückung von Bodenrauschen
- Dynamisches Auslesen der intrinsischen Kameraparameter
- EINFACHE VISUALISIERUNG: Tiefenbild in Grau + JET-Farbskala (wie DepthAI-Beispiel)
- LED-Lichtsteuerung (seriell) für Raspberry Pi
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
    ROI_X_MIN = -550
    ROI_X_MAX = -50
    ROI_Y_MIN = -250
    ROI_Y_MAX = 50

    # Höhen-Toleranzen
    MIN_OBJEKT_HOEHE_MM = 5.0
    MAX_OBJEKT_HOEHE_MM = 300.0
    MAX_LAENGE_BREITE_MM = 500.0

    # Punktwolken-Dichte (True = schnell, False = dicht)
    POINTCLOUD_SPARSE = True   # False = dichte Punktwolke

    # Fallback-Kameraparameter (nur falls Auslesen fehlschlägt)
    FX_FALLBACK = 822.7
    FY_FALLBACK = 822.7
    CX_FALLBACK = 321.5
    CY_FALLBACK = 239.5

    # Kalibrierungsdatei
    CALIB_FILE = "distanz_calibration.json"

    # Serielle Schnittstelle für LED (auf Windows ignoriert)
    SERIAL_PORT = "/dev/ttyUSB0"
    SERIAL_BAUDRATE = 9600
    
    # Für Visualisierung: maximaler Tiefenwert für konsistente Farben (mm)
    MAX_VIS_DEPTH_MM = 2000

# ===== LICHTSTEUERUNG (nur auf Linux/Pi aktiv) =====
def control_light(state: bool):
    """Schaltet LED ein/aus (nur wenn pyserial verfügbar und Port existiert)."""
    if not SERIAL_AVAILABLE:
        logger.debug("Lichtsteuerung deaktiviert (pyserial nicht installiert)")
        return
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





# ===== KAMERA-KLASSE =====
class OakD2Volume:
    def __init__(self):
        self.pipeline = None
        self.calibration = self.load_calibration()
        self.fx = self.fy = self.cx = self.cy = None

    def _build_pipeline(self):
        """Erstellt die DepthAI-Pipeline mit Mono, Stereo und PointCloud.
        Verwendet die Filtereinstellungen aus dem DepthAI-Disparity-Beispiel.
        """
        pipeline = dai.Pipeline()

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)

        # Post-Processing wie im DepthAI-Beispiel
        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = False
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.thresholdFilter.minRange = 200   # mm
        config.postProcessing.thresholdFilter.maxRange = 800 # mm
        config.postProcessing.decimationFilter.decimationFactor = 2
        stereo.initialConfig.set(config)

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
        """
        Nimmt eine Punktwolke und ein Tiefenbild auf.
        WICHTIG: Laut Luxonis-Dokumentation werden Punktkoordinaten standardmäßig in MILLIMETER geliefert!
        """
        pipeline = self._build_pipeline()
        try:
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                if self.fx is None:
                    self._get_intrinsics(device)

                q_pc = device.getOutputQueue("pointcloud", maxSize=1, blocking=True)
                q_depth = device.getOutputQueue("depth", maxSize=1, blocking=True)

                depth_data = q_depth.get()
                pc_data = q_pc.get()

                depth_frame = depth_data.getCvFrame()  # bereits in mm (16-bit)
                points = pc_data.getPoints()  # Offizielle Doku: in mm! [citation:1]
                
                return depth_frame, points
                
        except Exception as e:
            logger.error(f"Fehler bei Aufnahme: {e}")
            return None, None

    def calibrate_empty_space(self):
        """Führt eine Kalibrierung ohne Objekt durch und speichert die Bodenstatistik."""
        print("\n=== KALIBRIERUNG: BITTE OBJEKT ENTFERNEN ===")
        print("Drücken Sie Enter, wenn die Box leer ist...")
        input()

        #control_light(True)
        time.sleep(0.3)

        depth_frame, points = self.capture_pointcloud()
        #control_light(False)

        if points is None:
            print("Fehler bei Aufnahme")
            return False

        # Punkte im ROI filtern
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        mask = (z > 0) & \
               (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
               (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

        if np.sum(mask) < 100:
            print("Zu wenig Punkte im ROI. ROI prüfen oder Kamera ausrichten.")
            return False

        z_roi = z[mask]

        # Statistik des Bodens
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

        print(f"\nKalibrierung erfolgreich!")
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

    def _create_simple_visualization(self, depth_frame, length=None, width=None, height=None,
                                    min_z=None, ref_dist=None, error_msg=None):
        """
        Erstellt eine JET-Visualisierung aus dem Tiefenbild.
        Kann mit oder ohne Messwerte aufgerufen werden.
        """
        if depth_frame is None:
            # Fallback: schwarzes Bild mit Meldung
            vis = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(vis, "Kein Bild", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return vis

        # Tiefenbild auf festen Bereich clippen und normalisieren (0-2000 mm → 0-255)
        max_depth = 2000  # mm, anpassbar
        depth_clipped = np.clip(depth_frame, 0, max_depth)
        depth_norm = (depth_clipped * 255.0 / max_depth).astype(np.uint8)

        # JET-Farbmap anwenden
        vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Auf einheitliche Größe skalieren
        h, w = vis.shape[:2]
        target_w = 900
        scale = target_w / w
        target_h = int(h * scale)
        vis = cv2.resize(vis, (target_w, target_h))

        # Texte zusammenstellen
        lines = []
        if error_msg:
            lines.append(f"FEHLER: {error_msg}")
        else:
            if length is not None:
                lines.append(f"Laenge: {length:.1f} mm")
                lines.append(f"Breite: {width:.1f} mm")
                lines.append(f"Hoehe: {height:.1f} mm")
                lines.append(f"Volumen: {length * width * height / 1000:.1f} cm³")
                lines.append(f"min Z: {min_z:.1f} mm  (Referenz: {ref_dist:.1f} mm)")
            else:
                lines.append("Keine Messwerte")

        # Text einblenden (schwarzer Rand + weiße Schrift)
        y0 = 40
        for i, line in enumerate(lines):
            y = y0 + i * 35
            cv2.putText(vis, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(vis, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return vis


    def get_measurement(self) -> Dict:
        """Führt eine Volumenmessung mit Objekt durch."""
        if self.calibration is None:
            return self._error_result("Keine Kalibrierung vorhanden. Bitte zuerst leeren Raum kalibrieren.")

        try:
            # 1. Punktwolke aufnehmen
            depth_frame, points = self.capture_pointcloud()
            if points is None or len(points) == 0:
                return self._error_result("Keine Punktwolke erhalten.")

            print(f"📸 Aufnahme abgeschlossen: {len(points)} Punkte erhalten")
            print(f"max points {max(points[:, 2])} mm")
            print(f"min points {min(points[:, 2])} mm")

            # 2. ROI aus Kalibrierung laden
            ROI_X_MIN = self.calibration["roi"]["x_min"]
            ROI_X_MAX = self.calibration["roi"]["x_max"]
            ROI_Y_MIN = self.calibration["roi"]["y_min"]
            ROI_Y_MAX = self.calibration["roi"]["y_max"]
            x_min, x_max = ROI_X_MIN, ROI_X_MAX
            y_min, y_max = ROI_Y_MIN, ROI_Y_MAX

            # 3. Punkte im ROI filtern (räumlich)
            mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            roi_points = points[mask]

            if len(roi_points) == 0:
                return self._error_result("Keine Punkte im ROI gefunden. Bitte Objekt positionieren.")

            # 4. Höhenfilter: Nur Punkte oberhalb des Bodens (Objektpunkte)
            z_boden = self.calibration["z_median"]          # gemittelte Bodenhöhe
            schwellwert = 20                                  # mm Toleranz für Rauschen
            obj_mask = roi_points[:, 2] < (z_boden - schwellwert)
            obj_points = roi_points[obj_mask]

            if len(obj_points) == 0:
                return self._error_result("Keine Objektpunkte nach Höhenfilter gefunden.")

            print(f"max objekt points {max(obj_points[:, 2])} mm")
            print(f"min objekt points {min(obj_points[:, 2])} mm")
            print(f"🎯 Objektpunkte nach Höhenfilter: {len(obj_points)}")

            # 4b. Optional: Clustering
            from sklearn.cluster import DBSCAN
            if len(obj_points) > 10:
                clustering = DBSCAN(eps=25, min_samples=5).fit(obj_points[:, :2])
                labels = clustering.labels_
                unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique) > 0:
                    main_cluster = unique[np.argmax(counts)]
                    obj_points = obj_points[labels == main_cluster]
                    print(f"🔍 Hauptcluster: {len(obj_points)} Punkte")
                else:
                    return self._error_result("Kein klares Objektcluster gefunden.")

            # 5. Höhe berechnen
            z_min_obj = np.min(obj_points[:, 2])             # kleinste Z-Koordinate (höchster Punkt)
            hoehe = z_boden - z_min_obj                       # Objekthöhe
            if hoehe < 0:
                print("⚠️  Höhe negativ - wird korrigiert.")
                hoehe = abs(hoehe)

            # 6. Länge und Breite (robust gegen Ausreißer)
            laenge = np.percentile(obj_points[:, 0], 99) - np.percentile(obj_points[:, 0], 1)
            breite = np.percentile(obj_points[:, 1], 99) - np.percentile(obj_points[:, 1], 1)

            # 7. Volumen
            volumen_mm3 = laenge * breite * hoehe

            # --- Visualisierung mit Markierung des höchsten Punktes ---
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 10))

            # Gesamte Punktwolke (hellgrau)
            plt.scatter(points[:, 0], points[:, 1], c='lightgray', s=1, alpha=0.3, label='Alle Punkte')

            # ROI-Punkte (alle, hellrot) – optional, aber hilfreich
            plt.scatter(roi_points[:, 0], roi_points[:, 1], c='lightcoral', s=1, alpha=0.3, label='ROI (alle)')

            # Objektpunkte (rot)
            plt.scatter(obj_points[:, 0], obj_points[:, 1], c='red', s=2, label=f'Objekt ({len(obj_points)} Punkte)')

            # ROI-Rechteck
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label='ROI')
            plt.gca().add_patch(rect)

            # **Markierung des höchsten Punktes (min z)**
            idx_highest = np.argmin(obj_points[:, 2])          # Index des Punktes mit kleinster z-Koordinate
            highest_point = obj_points[idx_highest]
            plt.scatter(highest_point[0], highest_point[1], c='gold', s=100, marker='*',
                        edgecolors='black', linewidths=1, label=f'Höchster Punkt ({highest_point[2]:.1f} mm)')

            # **Optional: Markierung des tiefsten Punktes auf dem Objekt (max z)**
            idx_lowest = np.argmax(obj_points[:, 2])
            lowest_point = obj_points[idx_lowest]
            plt.scatter(lowest_point[0], lowest_point[1], c='lime', s=80, marker='s',
                        edgecolors='black', linewidths=1, label=f'Tiefster Punkt ({lowest_point[2]:.1f} mm)')

            # Achsenbegrenzung auf ROI (mit etwas Rand)
            plt.xlim(x_min - 10, x_max + 10)
            plt.ylim(y_min - 10, y_max + 10)
            plt.gca().set_aspect('equal')

            plt.xlabel('x [mm]')
            plt.ylabel('y [mm]')
            plt.title('Objekterkennung mit markiertem höchsten Punkt')
            plt.legend(loc='upper right', fontsize='small')
            plt.grid(True, linestyle=':', alpha=0.6)

            # Ergebnisse als Text einblenden
            textstr = f'Länge: {laenge:.1f} mm\nBreite: {breite:.1f} mm\nHöhe: {hoehe:.1f} mm\nVolumen: {volumen_mm3:.0f} mm³'
            plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.show()
            # --- Ende Visualisierung ---

            return {
                'success': True,
                'length': laenge,
                'width': breite,
                'height': hoehe,
                'volume': volumen_mm3,
                'depth_frame': depth_frame,
                'error': None
            }

        except Exception as e:
            logger.error(f"Volumenmessung fehlgeschlagen: {e}", exc_info=True)
            return self._error_result(str(e))



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
        print("\nStarte 3D-Volumenmessung (ROI 500x500 mm, einfache Visualisierung)")
        print("   (Falls keine Kalibrierung vorhanden, bitte zuerst mit 'calibrate' ausführen)\n")
        result = get_volume()
        if result['success']:
            print("\nMESSUNG ERFOLGREICH")
            print(f"   Länge:  {result['length']:.1f} mm")
            print(f"   Breite: {result['width']:.1f} mm")
            print(f"   Höhe:  {result['height']:.1f} mm")
            print(f"   Volumen: {result['volume']:.0f} mm³ ({result['volume']/1000:.1f} cm³)")

            if result['depth_frame'] is not None:
                cv2.imshow("Volumenmessung (JET)", result['depth_frame'])
                print("\nFenster schliessen → Ende")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"\nFEHLER: {result.get('error')}")