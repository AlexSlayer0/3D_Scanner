# workers/BarCode_v03.py
import cv2
import contextlib
from PIL import Image
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from typing import List, Dict, Optional, Any
import numpy as np
import logging

# Logging setup
logger = logging.getLogger(__name__)

class BarcodeDetector:
    """Barcode-Detection-Klasse mit YOLO-Unterstützung"""
    
    def __init__(self, model_path: str = "models/YOLOV8s_Barcode_Detection.pt"):
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO-Modell geladen von {model_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des YOLO-Modells: {e}")
            raise 
        self.detected_barcodes: List[Dict[str, Any]] = []
        
    def detect_barcodes_in_image(self, image: np.ndarray, image_index: int = 0, image_name: str = "") -> List[Dict[str, Any]]:
        """Erkennt alle Barcodes in einem Bild"""
        if image is None:
            logger.warning(f"Bild {image_index} ist None")
            return []
        
        try:
            # Konvertiere Bild zu RGB für YOLO
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # YOLO Vorhersage - CONFIDENZ erhöhen für bessere Ergebnisse
            results = self.model.predict(image_rgb, conf=0.5, verbose=False)
            image_barcodes = []
            
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        # Bounding Box Koordinaten
                        # Padding (z.B. 30% der Box-Größe)
                        pad_factor = 0.3
                        box_w = x2 - x1
                        box_h = y2 - y1
                        pad_x = int(box_w * pad_factor)
                        pad_y = int(box_h * pad_factor)

                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(image_rgb.shape[1], x2 + pad_x)
                        y2 = min(image_rgb.shape[0], y2 + pad_y)
                        
                        # Extrahiere ROI
                        roi = image_rgb[y1:y2, x1:x2]
                        
                        # Überspringe zu kleine ROIs
                        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                            continue
                        
                        # Barcode dekodieren
                        barcode_data = self._decode_roi(roi, f"image_{image_index}")
                        
                        if barcode_data["found"]:
                            barcode_info = {
                                "image_index": image_index,
                                "image_name": image_name,
                                "value": barcode_data["value"],
                                "type": barcode_data["type"],
                                "confidence": float(box.conf[0]) if hasattr(box, 'conf') else 0.0,
                                "bbox": [x1, y1, x2, y2],  # ÄNDERUNG: Liste statt Tuple
                                "cropped_image": roi.copy()
                            }
                            image_barcodes.append(barcode_info)
                            logger.info(f"Barcode erkannt in Bild {image_index}: {barcode_data['value']}")
            
            return image_barcodes
            
        except Exception as e:
            logger.error(f"Fehler bei Barcode-Erkennung Bild {image_index}: {e}")
            return []
    
    def _decode_roi(self, roi: np.ndarray, img_name: str) -> Dict[str, Any]:
        """Dekodiert einen Barcode in einer ROI"""
        if roi is None or roi.size == 0:
            return {"found": False, "value": None, "type": None}
        
        # REDUZIERT: Weniger Vorverarbeitungsmethoden für bessere Performance
        preprocess_methods = {
            "original": lambda x: x,
            "clahe": self._clahe_equalize,
            "unsharp": self._unsharp, 
            "invert": self._invert,
        }
        
        # Skalierungen und Rotationen
        scales = [1.0, 1.5, 2.0]  # Originalgröße, 150%, 200%
        rotations = rotations = [-135, -90, -45, -15, -5, 0, 5, 15, 45, 90, 135, 180]   # fokussiert auf die erfolgreichen Winkel laut Statistik
        
        # Konvertiere zu Grau
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Versuche verschiedene Vorverarbeitungen
        for pname, func in preprocess_methods.items():
            img_proc = func(gray)
            
            # Versuche verschiedene Skalierungen
            for s in scales:
                scaled = self._scale(img_proc, s)
                
                # Versuche verschiedene Rotationen
                for angle in rotations:
                    rotated = self._rotate(scaled, angle)
                    decoded_barcodes = self._try_decode(rotated)
                    
                    if decoded_barcodes:
                        for d in decoded_barcodes:
                            try:
                                val = d.data.decode("utf-8", errors="ignore").strip()
                                typ = d.type
                                
                                if val:
                                    return {"found": True, "value": val, "type": typ}
                            except:
                                continue
        
        return {"found": False, "value": None, "type": None}
    
    # Hilfsmethoden für Bildverarbeitung - OPTIMIERT
    @staticmethod
    def _clahe_equalize(gray):
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        except:
            return gray
    
    @staticmethod
    def _unsharp(gray):
        blurred = cv2.GaussianBlur(gray, (5,5), 1.0)
        return cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)  # amount=1.5

    @staticmethod
    def _invert(gray):
        return 255 - gray
    
    @staticmethod
    def _rotate(img, angle):
        if img is None or img.size == 0 or angle == 0:
            return img
            
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    
    @staticmethod
    def _scale(img, factor):
        if img is None or img.size == 0 or factor == 1.0:
            return img
            
        h, w = img.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)
        
        if new_w <= 0 or new_h <= 0:
            return img
            
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def _try_decode(img_np):
        if img_np is None or img_np.size == 0:
            return []
            
        try:
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
                
            pil_img = Image.fromarray(img_np)
            with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                return decode(pil_img)
        except Exception:
            return []
    
    def get_detected_barcodes(self) -> List[Dict[str, Any]]:
        """Gibt alle erkannten Barcodes zurück"""
        return self.detected_barcodes
    
    def reset(self):
        """Setzt den Detektor zurück"""
        self.detected_barcodes.clear()


# NEU: Fehlende Funktion für die Hauptseite
def detect_barcodes(images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    Hauptfunktion zur Barcode-Erkennung.
    Wird von der Hauptseite aufgerufen.
    
    Args:
        images: Liste von Bildern als numpy arrays
        
    Returns:
        Liste von Barcode-Ergebnissen
    """
    try:
        detector = BarcodeDetector()
        all_barcodes = []
        
        for idx, img in enumerate(images):
            if img is None:
                continue
                
            # Bildnamen basierend auf Position
            image_names = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]
            img_name = image_names[idx] if idx < len(image_names) else f"Bild_{idx}"
            
            # Barcodes erkennen
            barcodes = detector.detect_barcodes_in_image(img, idx, img_name)
            all_barcodes.extend(barcodes)
        
        logger.info(f"Insgesamt {len(all_barcodes)} Barcodes erkannt")
        return all_barcodes
        
    except Exception as e:
        logger.error(f"Fehler in detect_barcodes: {e}")
        return []