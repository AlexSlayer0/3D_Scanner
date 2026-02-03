#!/usr/bin/env python3
"""
Kamera_aufnahmen.py - Alle Kamerafunktionen für 3D-Scanner
Enthält USB-Kameras und OAK-D2 mit DepthAI
"""

import os
import time
import logging
import platform
import numpy as np
import cv2
import serial
from typing import List, Optional, Dict, Any, Tuple

# OpenCV-Warnungen reduzieren
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'V4L2'

# DepthAI optional importieren
try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    dai = None

# Logging
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
class CameraConfig:
    """Kamera-Konfiguration"""
    NUM_CAMERAS = 4  # 3 USB + 1 OAK-D2
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    USB0 = "/dev/ttyUSB0"
    BAURATE = 9600
    DEBUG_SINGLE_CAMERA = False

# ==================== OAK-D2 MANAGER ====================
class OakCameraManager:
    """Spezialisierter Manager für OAK-D2 mit DepthAI"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.is_running = False
        self.device = None
        self.pipeline = None
        self.q_rgb = None
        self.q_depth = None
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.last_measurements = None
        
    def initialize(self) -> bool:
        """Initialisiert die OAK-D2 Kamera"""
        if not DEPTHAI_AVAILABLE:
            logger.error("DepthAI nicht verfügbar - bitte installieren: pip install depthai")
            return False
        
        try:
            # Prüfe ob Geräte verfügbar sind
            device_infos = dai.Device.getAllAvailableDevices()
            if len(device_infos) == 0:
                logger.warning("Keine OAK-D2 Geräte gefunden")
                return False
            
            # Pipeline erstellen
            self.pipeline = dai.Pipeline()
            
            # RGB-Kamera
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            # Stereo-Kameras für Tiefenmessung
            mono_left = self.pipeline.create(dai.node.MonoCamera)
            mono_right = self.pipeline.create(dai.node.MonoCamera)
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            
            # StereoDepth Node
            stereo = self.pipeline.create(dai.node.StereoDepth)
            stereo.setRectification(True)
            stereo.setExtendedDisparity(True)
            stereo.setLeftRightCheck(True)
            
            # Links
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)
            
            # Ausgabe-Streams
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)
            
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)
            
            # Gerät verbinden
            self.device = dai.Device(self.pipeline)
            
            # Queues
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            self.is_running = True
            logger.info(f"OAK-D2 initialisiert: {self.device.getDeviceInfo().getMxId()}")
            return True
            
        except Exception as e:
            logger.error(f"OAK-D2 Initialisierung fehlgeschlagen: {e}")
            return False
    
    def get_rgb_frame(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        """Holt das neueste RGB-Bild"""
        if not self.is_running or self.q_rgb is None:
            return None
        
        try:
            frame_data = self.q_rgb.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                self.latest_rgb_frame = frame
                return frame
        except Exception as e:
            logger.error(f"Fehler beim Holen des RGB-Frames: {e}")
        
        return self.latest_rgb_frame
    
    def get_depth_measurement(self) -> Dict[str, Any]:
        """Führt eine Tiefenmessung durch (vereinfachte Version)"""
        if not self.is_running:
            return {"error": "OAK-D2 nicht initialisiert"}
        
        try:
            # Versuche Tiefendaten zu holen
            if self.q_depth is not None:
                depth_data = self.q_depth.tryGet()
                if depth_data is not None:
                    depth_frame = depth_data.getCvFrame()
                    self.latest_depth_frame = depth_frame
                    
                    # Einfache Messung im ROI
                    if depth_frame is not None:
                        return self._analyze_depth_simple(depth_frame)
            
            return {"error": "Keine Tiefendaten verfügbar"}
            
        except Exception as e:
            logger.error(f"Tiefenmessung fehlgeschlagen: {e}")
            return {"error": str(e)}
    
    def _analyze_depth_simple(self, depth_frame: np.ndarray) -> Dict[str, Any]:
        """Vereinfachte Tiefenanalyse (anpassbar)"""
        try:
            # ROI definieren (mittlerer Bereich)
            h, w = depth_frame.shape
            roi_x1 = int(w * 0.3)
            roi_y1 = int(h * 0.3)
            roi_x2 = int(w * 0.7)
            roi_y2 = int(h * 0.7)
            
            depth_roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Nur gültige Werte (nicht 0)
            valid_mask = depth_roi > 0
            if np.sum(valid_mask) == 0:
                return {"error": "Keine gültigen Tiefendaten"}
            
            valid_depths = depth_roi[valid_mask]
            
            # Basis-Messungen
            min_depth = float(np.min(valid_depths))
            max_depth = float(np.max(valid_depths))
            avg_depth = float(np.mean(valid_depths))
            std_depth = float(np.std(valid_depths))
            
            # Einfache Objekterkennung: Suche nach Vordergrundobjekten
            # (Objekte sind näher als Hintergrund)
            background_estimate = np.percentile(valid_depths, 75)
            object_mask = depth_roi < (background_estimate * 0.8)
            
            object_pixels = np.sum(object_mask)
            total_pixels = depth_roi.size
            
            self.last_measurements = {
                'min_depth_mm': min_depth,
                'max_depth_mm': max_depth,
                'avg_depth_mm': avg_depth,
                'std_depth_mm': std_depth,
                'object_coverage': float(object_pixels / total_pixels) if total_pixels > 0 else 0,
                'roi_width_px': roi_x2 - roi_x1,
                'roi_height_px': roi_y2 - roi_y1,
                'valid_pixels': int(np.sum(valid_mask))
            }
            
            return self.last_measurements
            
        except Exception as e:
            logger.error(f"Tiefenanalyse fehlgeschlagen: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Schließt die OAK-D2 Verbindung"""
        if self.device:
            self.device.close()
        self.is_running = False
        logger.info("OAK-D2 geschlossen")

# ==================== HAUPT CAMERA MANAGER ====================
class CameraManager:
    """Haupt-Kamera-Manager für alle Kameras (USB + OAK-D2)"""
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        
        # USB-Kamera Variablen
        self.port = self.config.USB0
        self.baudrate = self.config.BAURATE
        self.debug_single_camera = self.config.DEBUG_SINGLE_CAMERA
        
        # Kamera-Listen
        self.available_cameras: List[int] = []
        self.oak_available: bool = False
        self.oak_index: Optional[int] = None
        
        # OAK-D2 Manager
        self.oak_manager: Optional[OakCameraManager] = None
        
        # Initialisiere Kameras
        self._find_cameras()
        logger.info(f"USB-Kameras gefunden: {self.available_cameras}")
        logger.info(f"OAK-D2 verfügbar: {self.oak_available}")
    
    def _find_cameras(self):
        """Findet alle verfügbaren Kameras"""
        usb_cameras = []
        
        # Suche USB-Kameras (nur gerade Indizes für bessere Performance)
        for i in range(0, 10, 2):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        usb_cameras.append(i)
                        logger.info(f"USB-Kamera gefunden: Index {i}")
                    cap.release()
            except Exception:
                pass
        
        # Falls weniger als 3, suche auch ungerade
        if len(usb_cameras) < 3:
            for i in [1, 3, 5, 7, 9]:
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            usb_cameras.append(i)
                            logger.info(f"USB-Kamera gefunden: Index {i}")
                        cap.release()
                except Exception:
                    pass
        
        # Sortiere und begrenze auf 3 Kameras
        usb_cameras.sort()
        self.available_cameras = usb_cameras[:3]
        
        # Prüfe ob OAK-D2 verfügbar ist (über DepthAI)
        if DEPTHAI_AVAILABLE:
            try:
                device_infos = dai.Device.getAllAvailableDevices()
                if len(device_infos) > 0:
                    self.oak_available = True
                    logger.info(f"OAK-D2 erkannt: {device_infos[0].getMxId()}")
            except Exception as e:
                logger.warning(f"OAK-D2 nicht verfügbar: {e}")
    
    def _get_camera_backend(self) -> int:
        """Bestimmt den passenden Camera Backend"""
        if platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_V4L2
    
    def _make_placeholder(self, camera_id: int = -1) -> np.ndarray:
        """Erstellt ein Platzhalterbild"""
        img = np.zeros((self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3), dtype=np.uint8)
        if camera_id == 3:
            text = "OAK-D2 nicht verfügbar"
        else:
            text = f"Kamera {camera_id} nicht verfügbar"
        
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        return img
    
    def initialize_oak(self) -> bool:
        """Initialisiert die OAK-D2 Kamera (wenn verfügbar)"""
        if not self.oak_available or self.oak_manager is not None:
            return False
        
        try:
            self.oak_manager = OakCameraManager(self.config)
            success = self.oak_manager.initialize()
            if success:
                logger.info("OAK-D2 erfolgreich initialisiert")
            return success
        except Exception as e:
            logger.error(f"OAK-D2 Initialisierung fehlgeschlagen: {e}")
            self.oak_available = False
            return False
    
    def take_picture(self, camera_id: int) -> np.ndarray:
        """Nimmt ein Bild mit der angegebenen Kamera auf"""
        # OAK-D2 (Kamera 3)
        if camera_id == 3:
            return self._take_oak_picture()
        
        # USB-Kameras (0, 1, 2)
        return self._take_usb_picture(camera_id)
    
    def _take_usb_picture(self, camera_id: int) -> np.ndarray:
        """Nimmt ein Bild von einer USB-Kamera auf"""
        if camera_id >= len(self.available_cameras):
            return self._make_placeholder(camera_id)
        
        system_index = self.available_cameras[camera_id]
        backend = self._get_camera_backend()
        
        try:
            cap = cv2.VideoCapture(system_index, backend)
            if not cap.isOpened():
                logger.error(f"USB-Kamera {camera_id} (Index {system_index}) konnte nicht geöffnet werden")
                cap.release()
                return self._make_placeholder(camera_id)
            
            # Kamera-Einstellungen
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.IMAGE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.IMAGE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            # Ein paar Frames verwerfen
            for _ in range(3):
                cap.read()
            
            # Beleuchtung einschalten
            try:
                ser = serial.Serial(self.port, self.baudrate, timeout=1)
                time.sleep(0.1)
                
                def send_command(command):
                    ser.write((command + "\n").encode('utf-8'))
                    time.sleep(0.05)
                
                send_command("Change")
                send_command("a")  # Alle an
                time.sleep(0.3)
                
                ret, frame = cap.read()
                
                send_command("Change")
                send_command("0")  # Alle aus
                ser.close()
            except Exception as e:
                logger.warning(f"Beleuchtung fehlgeschlagen: {e}")
                ret, frame = cap.read()
            
            cap.release()
            
            if ret and frame is not None:
                logger.info(f"Bild von USB-Kamera {camera_id} aufgenommen")
                return frame
            else:
                logger.error(f"Bildaufnahme von USB-Kamera {camera_id} fehlgeschlagen")
                return self._make_placeholder(camera_id)
                
        except Exception as e:
            logger.error(f"Fehler bei USB-Kamera {camera_id}: {e}")
            return self._make_placeholder(camera_id)
    
    def _take_oak_picture(self) -> np.ndarray:
        """Nimmt ein Bild von der OAK-D2 auf"""
        if not self.oak_available:
            logger.warning("OAK-D2 nicht verfügbar")
            return self._make_placeholder(3)
        
        # OAK initialisieren wenn nötig
        if self.oak_manager is None:
            if not self.initialize_oak():
                return self._make_placeholder(3)
        
        if self.oak_manager is None:
            return self._make_placeholder(3)
        
        try:
            # Bild von OAK holen
            frame = self.oak_manager.get_rgb_frame(timeout_ms=2000)
            
            if frame is not None:
                logger.info("Bild von OAK-D2 aufgenommen")
                # Auf Standardgröße resizen
                if frame.shape[:2] != (self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH):
                    frame = cv2.resize(frame, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
                return frame
            else:
                logger.warning("Kein Bild von OAK-D2 erhalten")
                return self._make_placeholder(3)
                
        except Exception as e:
            logger.error(f"Fehler bei OAK-D2 Bildaufnahme: {e}")
            return self._make_placeholder(3)
    
    def take_all_pictures(self) -> List[np.ndarray]:
        """Nimmt Bilder von allen Kameras auf"""
        images = []
        
        if self.debug_single_camera:
            logger.debug("Debug-Modus: Verwende eine Kamera für alle Bilder")
            for i in range(self.config.NUM_CAMERAS):
                if i == 3 and self.oak_available:
                    img = self._take_oak_picture()
                else:
                    img = self._take_usb_picture(0)
                images.append(img)
        else:
            # Normale Aufnahme
            for i in range(self.config.NUM_CAMERAS):
                if i < 3:  # USB-Kameras
                    if i < len(self.available_cameras):
                        img = self._take_usb_picture(i)
                    else:
                        img = self._make_placeholder(i)
                else:  # OAK-D2 (Kamera 3)
                    if self.oak_available:
                        img = self._take_oak_picture()
                    else:
                        img = self._make_placeholder(i)
                images.append(img)
        
        # Statistik
        successful = sum(1 for img in images if img is not None and not np.all(img == 0))
        logger.info(f"Bildaufnahme abgeschlossen: {successful}/{self.config.NUM_CAMERAS} erfolgreich")
        
        return images
    
    def get_oak_depth_measurement(self) -> Optional[Dict[str, Any]]:
        """Holt eine Tiefenmessung von der OAK-D2"""
        if not self.oak_available or self.oak_manager is None:
            return None
        
        try:
            # Stelle sicher dass OAK initialisiert ist
            if self.oak_manager is None:
                if not self.initialize_oak():
                    return None
            
            # Hole Tiefenmessung
            measurements = self.oak_manager.get_depth_measurement()
            
            if "error" in measurements:
                logger.warning(f"OAK-D2 Messung fehlgeschlagen: {measurements['error']}")
                return None
            
            return measurements
            
        except Exception as e:
            logger.error(f"Fehler bei OAK-D2 Messung: {e}")
            return None
    
    def check_camera_status(self) -> Dict[str, Any]:
        """Gibt den Status aller Kameras zurück"""
        return {
            "usb_cameras": len(self.available_cameras),
            "usb_indices": self.available_cameras,
            "oak_available": self.oak_available,
            "oak_initialized": self.oak_manager is not None,
            "total_cameras": self.config.NUM_CAMERAS
        }
    
    def cleanup(self):
        """Ressourcen freigeben"""
        if self.oak_manager:
            self.oak_manager.close()
        logger.info("Kamera-Ressourcen freigegeben")

# ==================== HELPER FUNCTIONS ====================
def test_all_cameras(config: Optional[CameraConfig] = None):
    """Testfunktion für alle Kameras"""
    print("=== KAMERA TEST ===")
    
    manager = CameraManager(config)
    status = manager.check_camera_status()
    
    print(f"USB-Kameras: {status['usb_cameras']} (Indizes: {status['usb_indices']})")
    print(f"OAK-D2: {'Verfügbar' if status['oak_available'] else 'Nicht verfügbar'}")
    
    # Teste Bilder
    print("\nTeste Bildaufnahme...")
    images = manager.take_all_pictures()
    
    for i, img in enumerate(images):
        if img is not None and not np.all(img == 0):
            print(f"  Kamera {i}: OK ({img.shape})")
        else:
            print(f"  Kamera {i}: FEHLER")
    
    # Teste OAK-D2 Messung
    if status['oak_available']:
        print("\nTeste OAK-D2 Tiefenmessung...")
        measurements = manager.get_oak_depth_measurement()
        if measurements:
            print(f"  Messung erfolgreich: {measurements}")
        else:
            print("  Messung fehlgeschlagen")
    
    manager.cleanup()
    print("\nTest abgeschlossen")

if __name__ == "__main__":
    # Direkter Test
    test_all_cameras()