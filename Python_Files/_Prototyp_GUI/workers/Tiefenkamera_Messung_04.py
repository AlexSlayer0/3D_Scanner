#!/usr/bin/env python3
"""
3D Volumenmessung mit OAK-D2 - Komplettlösung ohne Open3D
Kompatibel mit Python 3.13 - Nutzt nur depthai, numpy und opencv
"""

import cv2
import depthai as dai
import numpy as np
import sys
import time
import json
import os

class VolumeMeasurer:
    def __init__(self):
        # Feste Parameter
        self.REFERENCE_HEIGHT_MM = 580.0  # 58cm Kameraabstand
        self.MEASURE_AREA_MM = 500.0      # 500x500mm Bereich
        self.MIN_OBJECT_HEIGHT_MM = 20.0  # Mindesthöhe für Objekterkennung
        
        # Kameraparameter für OAK-D2 (640x400)
        self.FX = 822.7
        self.FY = 822.7
        self.CX = 321.5
        self.CY = 239.5
        
        # Kalibrierung
        self.calibration_file = "calibration.json"
        self.calibration = {
            'scale_x': 1.0, 'scale_y': 1.0, 'scale_z': 1.0,
            'offset_x': 0.0, 'offset_y': 0.0, 'offset_z': 0.0
        }
        self.load_calibration()
    
    def load_calibration(self):
        """Lädt Kalibrierung aus JSON-Datei"""
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                self.calibration = json.load(f)
            print(f"✅ Kalibrierung geladen: {self.calibration_file}")
    
    def save_calibration(self):
        """Speichert Kalibrierung in JSON-Datei"""
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration, f, indent=4)
        print(f"💾 Kalibrierung gespeichert: {self.calibration_file}")
    
    def setup_camera(self):
        """Richtet die OAK-D2 Kamera ein"""
        pipeline = dai.Pipeline()
        
        # Monokameras konfigurieren
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        
        # StereoDepth konfigurieren
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setConfidenceThreshold(200)
        
        # Verbindungen herstellen
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # Ausgabe konfigurieren
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        return pipeline
    
    def get_stable_depth_frame(self, num_frames=5):
        """Holt mehrere Tiefenbilder und mittelt sie für Stabilität"""
        try:
            pipeline = self.setup_camera()
            with dai.Device(pipeline) as device:
                depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=True)
                frames = []
                
                for _ in range(num_frames):
                    frame = depth_queue.get().getCvFrame()
                    if frame is not None:
                        frames.append(frame.astype(np.float32))
                    time.sleep(0.05)
                
                if not frames:
                    return None
                
                # Median für Rauschreduktion
                frames_array = np.array(frames)
                median_frame = np.median(frames_array, axis=0)
                
                return median_frame.astype(np.uint16)
                
        except Exception as e:
            print(f"❌ Kamera Fehler: {e}")
            return None
    
    def depth_to_pointcloud(self, depth_frame):
        """Konvertiert Tiefenbild in 3D-Punktwolke (mm)"""
        height, width = depth_frame.shape
        
        # Effiziente Erstellung von Gitterkoordinaten
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Nur gültige Tiefenwerte
        valid_mask = depth_frame > 0
        if not np.any(valid_mask):
            return np.array([])
        
        # Extraktion der gültigen Werte
        u_valid = u[valid_mask].flatten()
        v_valid = v[valid_mask].flatten()
        z_valid = depth_frame[valid_mask].flatten().astype(np.float32)
        
        # Umrechnung in 3D-Koordinaten (mm)
        x_points = (u_valid - self.CX) * z_valid / self.FX
        y_points = (v_valid - self.CY) * z_valid / self.FY
        
        # Zusammenfügen der Punkte
        points = np.column_stack((x_points, y_points, z_valid))
        return points
    
    def extract_object_points(self, depth_frame):
        """Extrahiert Punkte, die zum Objekt gehören"""
        # Vollständige Punktwolke erstellen
        all_points = self.depth_to_pointcloud(depth_frame)
        
        if len(all_points) == 0:
            return np.array([])
        
        # Objektpunkte: alles unterhalb der Referenzhöhe
        object_threshold = self.REFERENCE_HEIGHT_MM - self.MIN_OBJECT_HEIGHT_MM
        object_mask = all_points[:, 2] < object_threshold
        
        return all_points[object_mask]
    
    def filter_measurement_area(self, points_3d):
        """Filtert Punkte außerhalb des 500x500mm Bereichs"""
        if len(points_3d) == 0:
            return points_3d
        
        half_size = self.MEASURE_AREA_MM / 2
        mask = (
            (np.abs(points_3d[:, 0]) <= half_size) &  # X: ±250mm
            (np.abs(points_3d[:, 1]) <= half_size)    # Y: ±250mm
        )
        
        return points_3d[mask]
    
    def find_main_cluster(self, points_3d, eps=30.0, min_samples=15):
        """Findet das Hauptobjekt mit DBSCAN-Clustering"""
        if len(points_3d) < min_samples:
            return points_3d
        
        from sklearn.cluster import DBSCAN
        
        try:
            # Cluster nur basierend auf X/Y-Koordinaten
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d[:, :2])
            labels = clustering.labels_
            
            # Größten Cluster finden (ignoriere Rauschen mit Label -1)
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            
            if len(unique_labels) == 0:
                return points_3d
            
            main_label = unique_labels[np.argmax(counts)]
            main_cluster = points_3d[labels == main_label]
            
            return main_cluster if len(main_cluster) > min_samples else points_3d
            
        except Exception:
            # Fallback: Alle Punkte zurückgeben, wenn Clustering fehlschlägt
            return points_3d
    
    def calculate_dimensions(self, points_3d):
        """Berechnet 3D-Dimensionen aus Punktwolke"""
        if len(points_3d) < 10:
            return None
        
        # Min/Max finden
        min_vals = np.min(points_3d, axis=0)
        max_vals = np.max(points_3d, axis=0)
        
        # Basis-Dimensionen
        length_raw = max_vals[0] - min_vals[0]  # X-Richtung
        width_raw = max_vals[1] - min_vals[1]   # Y-Richtung
        
        # Höhe: Referenz - niedrigster Punkt
        min_z = min_vals[2]
        height_raw = self.REFERENCE_HEIGHT_MM - min_z
        
        # Sicherstellen: Länge ≥ Breite
        if length_raw < width_raw:
            length_raw, width_raw = width_raw, length_raw
        
        # Kalibrierung anwenden
        length = length_raw * self.calibration['scale_x'] + self.calibration['offset_x']
        width = width_raw * self.calibration['scale_y'] + self.calibration['offset_y']
        height = height_raw * self.calibration['scale_z'] + self.calibration['offset_z']
        
        volume = length * width * height
        
        return {
            'length': length,
            'width': width,
            'height': height,
            'volume': volume,
            'center_x': (min_vals[0] + max_vals[0]) / 2,
            'center_y': (min_vals[1] + max_vals[1]) / 2,
            'points_count': len(points_3d)
        }
    
    def calculate_calibration_factors(self, target_values, measured_values):
        """Berechnet Kalibrierungsfaktoren"""
        if any(v <= 0 for v in measured_values):
            return False
        
        self.calibration = {
            'scale_x': target_values[0] / measured_values[0],
            'scale_y': target_values[1] / measured_values[1],
            'scale_z': target_values[2] / measured_values[2],
            'offset_x': 0.0,
            'offset_y': 0.0,
            'offset_z': 0.0
        }
        
        print(f"🔧 Kalibrierungsfaktoren berechnet:")
        print(f"   X: {self.calibration['scale_x']:.4f}")
        print(f"   Y: {self.calibration['scale_y']:.4f}")
        print(f"   Z: {self.calibration['scale_z']:.4f}")
        
        return True
    
    def create_visualization(self, depth_frame, points_3d, dimensions):
        """Erstellt Visualisierung mit OpenCV"""
        # Tiefenbild für Anzeige normalisieren
        vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        height, width = depth_frame.shape
        
        # Messbereich anzeigen (500x500mm)
        center_x, center_y = width // 2, height // 2
        pixels_per_mm = self.FX / self.REFERENCE_HEIGHT_MM
        radius_px = int(250 * pixels_per_mm)  # 250mm Radius
        
        cv2.circle(vis, (center_x, center_y), radius_px, (0, 255, 255), 2)
        cv2.putText(vis, "500x500mm", (center_x-50, center_y-radius_px-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Objektposition (falls Punkte vorhanden)
        if len(points_3d) > 0 and 'center_x' in dimensions:
            obj_x = dimensions['center_x']
            obj_y = dimensions['center_y']
            
            # Näherungsweise Rückrechnung zu Pixel
            u = int(obj_x * self.FX / self.REFERENCE_HEIGHT_MM + self.CX)
            v = int(obj_y * self.FY / self.REFERENCE_HEIGHT_MM + self.CY)
            
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(vis, (u, v), 8, (255, 0, 0), -1)
                cv2.putText(vis, "Object", (u+10, v),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Messwerte anzeigen
        y_offset = 30
        cv2.putText(vis, f"Length: {dimensions['length']:.1f}mm", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Width: {dimensions['width']:.1f}mm", 
                   (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Height: {dimensions['height']:.1f}mm", 
                   (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Volume: {dimensions['volume']/1000:.1f}cm³", 
                   (20, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Kalibrierungsstatus
        if self.calibration['scale_x'] != 1.0:
            cv2.putText(vis, "Calibrated", (width - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def measure(self, calibration_mode=False, target_values=None):
        """Hauptmessfunktion"""
        print("\n📷 Erfasse Tiefenbild...")
        depth_frame = self.get_stable_depth_frame(num_frames=7)
        
        if depth_frame is None:
            return {'success': False, 'error': 'Kein Tiefenbild erhalten'}
        
        print(f"✅ Tiefenbild: {depth_frame.shape}")
        
        # 3D-Punkte extrahieren
        raw_points = self.extract_object_points(depth_frame)
        
        if len(raw_points) < 50:
            return {'success': False, 'error': f'Zu wenige Punkte ({len(raw_points)})'}
        
        print(f"📊 {len(raw_points)} 3D-Punkte extrahiert")
        
        # Auf Messbereich filtern
        filtered_points = self.filter_measurement_area(raw_points)
        
        if len(filtered_points) < 20:
            return {'success': False, 'error': 'Objekt außerhalb Messbereich'}
        
        print(f"🎯 {len(filtered_points)} Punkte im Messbereich")
        
        # Hauptobjekt finden
        main_object_points = self.find_main_cluster(filtered_points)
        
        if len(main_object_points) < 10:
            return {'success': False, 'error': 'Kein klares Objekt erkannt'}
        
        print(f"📦 Hauptobjekt: {len(main_object_points)} Punkte")
        
        # Dimensionen berechnen
        dimensions = self.calculate_dimensions(main_object_points)
        
        if dimensions is None:
            return {'success': False, 'error': 'Dimensionen konnten nicht berechnet werden'}
        
        # Kalibrierung bei Bedarf
        if calibration_mode and target_values is not None:
            measured = (dimensions['length'], dimensions['width'], dimensions['height'])
            if self.calculate_calibration_factors(target_values, measured):
                self.save_calibration()
        
        # Visualisierung erstellen
        vis = self.create_visualization(depth_frame, main_object_points, dimensions)
        
        return {
            'success': True,
            'length': round(dimensions['length'], 1),
            'width': round(dimensions['width'], 1),
            'height': round(dimensions['height'], 1),
            'volume': round(dimensions['volume'], 1),
            'position_x': round(dimensions['center_x'], 1),
            'position_y': round(dimensions['center_y'], 1),
            'points': dimensions['points_count'],
            'image': vis,
            'error': None
        }


def main():
    print("=" * 60)
    print("3D-VOLUMENMESSUNG - OAK-D2 KAMERA")
    print("=" * 60)
    print("Objektposition egal - Nur innerhalb 500x500mm Bereich!")
    print("=" * 60)
    
    measurer = VolumeMeasurer()
    
    # Kommandozeilen-Argumente verarbeiten
    if len(sys.argv) >= 2 and sys.argv[1] == "calibrate":
        if len(sys.argv) != 5:
            print("\n🔧 KALIBRIERUNGSMODUS:")
            print("  python script.py calibrate <Laenge> <Breite> <Hoehe>")
            print("  Beispiel: python script.py calibrate 300 200 150")
            print("\n📦 Objekt IRGENDWO im gelben Kreis platzieren!")
            return
        
        try:
            length = float(sys.argv[2])
            width = float(sys.argv[3])
            height = float(sys.argv[4])
            
            print(f"\n🎯 Kalibrierung gestartet")
            print(f"   Soll-Werte: {length} x {width} x {height} mm")
            print("\n📦 Objekt im Messbereich platzieren...")
            print("Drücke Enter wenn bereit...")
            input()
            
            result = measurer.measure(calibration_mode=True, 
                                    target_values=(length, width, height))
            
            if result['success']:
                print(f"\n✅ KALIBRIERUNG ERFOLGREICH!")
                print(f"   Position: X={result['position_x']:.0f}mm, Y={result['position_y']:.0f}mm")
                print(f"   Gemessen (vor Kalibrierung):")
                print(f"     Laenge: {result['length']:.1f} mm")
                print(f"     Breite: {result['width']:.1f} mm")
                print(f"     Hoehe: {result['height']:.1f} mm")
                print(f"     Punkte: {result['points']}")
                
                if result['image'] is not None:
                    cv2.imshow("Kalibrierung", result['image'])
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
            else:
                print(f"\n❌ Kalibrierung fehlgeschlagen: {result.get('error')}")
                
        except ValueError:
            print("❌ Fehler: Ungültige Zahlenwerte")
            
    elif len(sys.argv) >= 2 and sys.argv[1] == "info":
        print("\n📊 SYSTEMINFORMATIONEN:")
        print("=" * 40)
        print(f"Referenzhoehe: {measurer.REFERENCE_HEIGHT_MM}mm")
        print(f"Messbereich: {measurer.MEASURE_AREA_MM}x{measurer.MEASURE_AREA_MM}mm")
        print(f"Brennweite: fx={measurer.FX}, fy={measurer.FY}")
        print(f"Hauptpunkt: cx={measurer.CX}, cy={measurer.CY}")
        print(f"\n🔧 Kalibrierungsfaktoren:")
        for key, val in measurer.calibration.items():
            print(f"  {key}: {val:.6f}")
            
    elif len(sys.argv) >= 2 and sys.argv[1] == "reset":
        measurer.calibration = {
            'scale_x': 1.0, 'scale_y': 1.0, 'scale_z': 1.0,
            'offset_x': 0.0, 'offset_y': 0.0, 'offset_z': 0.0
        }
        measurer.save_calibration()
        print("✅ Kalibrierung zurückgesetzt")
        
    else:
        print("\n📏 NORMALER MESSMODUS")
        print("Objekt IRGENDWO im Messbereich platzieren")
        print("(Innerhalb des gelben Kreises)")
        print("\nDrücke Enter wenn bereit...")
        input()
        
        result = measurer.measure()
        
        if result['success']:
            print(f"\n✅ MESSUNG ERFOLGREICH!")
            print(f"   Position: X={result['position_x']:.0f}mm, Y={result['position_y']:.0f}mm")
            print(f"   Laenge: {result['length']:.1f} mm")
            print(f"   Breite: {result['width']:.1f} mm")
            print(f"   Hoehe: {result['height']:.1f} mm")
            print(f"   Volumen: {result['volume']:.0f} mm³ ({result['volume']/1000:.1f} cm³)")
            print(f"   3D-Punkte: {result['points']}")
            
            if measurer.calibration['scale_x'] != 1.0:
                print(f"\n🔧 Kalibrierung aktiv")
            else:
                print(f"\n⚠️  Unkalibriert (Standardwerte)")
            
            if result['image'] is not None:
                cv2.imshow("3D-Scanner - Volumenmessung", result['image'])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"\n❌ Messung fehlgeschlagen: {result.get('error')}")


if __name__ == "__main__":
    main()
