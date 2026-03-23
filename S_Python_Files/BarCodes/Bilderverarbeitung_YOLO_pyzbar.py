# workers/BarCode_v03.py
import cv2
import contextlib
from PIL import Image
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import os
import json

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Barcode-Detektor-Klasse
# ----------------------------
class BarcodeDetector:
    """Barcode-Detection-Klasse mit YOLO-Unterstützung"""
    
    def __init__(self, model_path: str = "YOLOV8s_Barcode_Detection.pt"):
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
            
            # YOLO Vorhersage
            results = self.model.predict(image_rgb, conf=0.5, verbose=False)
            image_barcodes = []
            box_counter = 0
            
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        # Originalbox (ohne Padding)
                        x1_orig, y1_orig, x2_orig, y2_orig = map(int, box.xyxy[0])
                        box_w = x2_orig - x1_orig
                        box_h = y2_orig - y1_orig
                        
                        # Padding (30%)
                        pad_factor = 0.3
                        pad_x = int(box_w * pad_factor)
                        pad_y = int(box_h * pad_factor)
                        x1 = max(0, x1_orig - pad_x)
                        y1 = max(0, y1_orig - pad_y)
                        x2 = min(image_rgb.shape[1], x2_orig + pad_x)
                        y2 = min(image_rgb.shape[0], y2_orig + pad_y)
                        
                        # ROI mit Padding
                        roi_padded = image_rgb[y1:y2, x1:x2]
                        if roi_padded.size == 0 or roi_padded.shape[0] < 10 or roi_padded.shape[1] < 10:
                            continue
                        
                        # ROI ohne Padding (für spätere Referenz)
                        roi_original = image_rgb[y1_orig:y2_orig, x1_orig:x2_orig]
                        
                        # Eindeutige ID
                        roi_id = f"{image_name}_box_{box_counter}" if image_name else f"image_{image_index}_box_{box_counter}"
                        box_counter += 1
                        
                        # Barcode dekodieren und Vorverarbeitungsbilder speichern
                        barcode_data = self._decode_roi(roi_padded, roi_original, roi_id)
                        
                        if barcode_data["found"]:
                            barcode_info = {
                                "image_index": image_index,
                                "image_name": image_name,
                                "value": barcode_data["value"],
                                "type": barcode_data["type"],
                                "confidence": float(box.conf[0]) if hasattr(box, 'conf') else 0.0,
                                "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                                "cropped_image": roi_padded.copy(),
                                "decoding_details": barcode_data.get("decoding_details", {}),
                                "all_attempts": barcode_data.get("all_attempts", []),
                                "roi_id": roi_id
                            }
                            image_barcodes.append(barcode_info)
                            logger.info(f"Barcode erkannt in Bild {image_index}: {barcode_data['value']}")
            
            return image_barcodes
            
        except Exception as e:
            logger.error(f"Fehler bei Barcode-Erkennung Bild {image_index}: {e}")
            return []
    
    def _decode_roi(self, roi_padded: np.ndarray, roi_original: np.ndarray, roi_id: str) -> Dict[str, Any]:
        """Dekodiert einen Barcode und protokolliert alle getesteten Kombinationen"""
        if roi_padded is None or roi_padded.size == 0:
            return {"found": False, "value": None, "type": None}
        
        # Konvertiere zu Grau
        if len(roi_padded.shape) == 3:
            gray = cv2.cvtColor(roi_padded, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi_padded
        
        # --- Speichere die vier Vorverarbeitungsstufen ---
        self._save_preprocessed_images(roi_padded, gray, roi_id)
        
        # --- Dekodierungsversuche mit deutschen Methodennamen ---
        preprocess_methods = {
            "Original": lambda x: x,
            "CLAHE": self._clahe_equalize,
            "Unsharp": self._unsharp, 
            "Invertiert": self._invert,
        }
        scales = [1.0, 1.5, 2.0]
        rotations = [-135, -90, -45, -15, -5, 0, 5, 15, 45, 90, 135, 180]
        
        attempts = []
        success_attempt = None
        
        for pname, func in preprocess_methods.items():
            img_proc = func(gray)
            for s in scales:
                scaled = self._scale(img_proc, s)
                for angle in rotations:
                    rotated = self._rotate(scaled, angle)
                    decoded_barcodes = self._try_decode(rotated)
                    found = False
                    value = None
                    typ = None
                    if decoded_barcodes:
                        for d in decoded_barcodes:
                            try:
                                val = d.data.decode("utf-8", errors="ignore").strip()
                                t = d.type
                                if val:
                                    found = True
                                    value = val
                                    typ = t
                                    break
                            except:
                                continue
                    attempt = {
                        "preprocess": pname,
                        "scale": s,
                        "angle": angle,
                        "found": found,
                        "value": value,
                        "type": typ
                    }
                    attempts.append(attempt)
                    if found and success_attempt is None:
                        success_attempt = attempt
                        logger.info(f"Barcode erkannt in {roi_id} mit {pname}, Skalierung={s}, Rotation={angle}°: {value}")
        
        # JSON mit allen Versuchen speichern
        self._save_decoding_json(roi_id, attempts, success_attempt)
        
        if success_attempt:
            return {
                "found": True,
                "value": success_attempt["value"],
                "type": success_attempt["type"],
                "decoding_details": success_attempt,
                "all_attempts": attempts
            }
        else:
            return {"found": False, "value": None, "type": None, "all_attempts": attempts}
    
    def _save_preprocessed_images(self, roi_rgb: np.ndarray, gray: np.ndarray, roi_id: str):
        """Speichert die vier Vorverarbeitungsstufen, die auch in der Erkennung verwendet werden."""
        try:
            base_path = "Bilderverarbeitung"
            folder = os.path.join(base_path, roi_id)
            os.makedirs(folder, exist_ok=True)
            
            # 1. Original (ROI mit Padding, in Farbe)
            cv2.imwrite(os.path.join(folder, "original.png"), cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
            
            # 2. CLAHE (Kontrastverstärkung)
            clahe_img = self._clahe_equalize(gray)
            cv2.imwrite(os.path.join(folder, "clahe.png"), clahe_img)
            
            # 3. Unsharp (Schärfung)
            unsharp_img = self._unsharp(gray)
            cv2.imwrite(os.path.join(folder, "unsharp.png"), unsharp_img)
            
            # 4. Invertiert (Invertierung)
            invert_img = self._invert(gray)
            cv2.imwrite(os.path.join(folder, "invertiert.png"), invert_img)
            
            logger.info(f"Vorverarbeitungsbilder gespeichert in {folder}")
        except Exception as e:
            logger.warning(f"Konnte Vorverarbeitungsbilder nicht speichern: {e}")
    
    def _save_decoding_json(self, roi_id: str, attempts: List[Dict], success_attempt: Optional[Dict] = None):
        """Speichert alle Versuche und das erfolgreiche Ergebnis als JSON"""
        try:
            base_path = "Bilderverarbeitung"
            folder = os.path.join(base_path, roi_id)
            os.makedirs(folder, exist_ok=True)
            data = {
                "roi_id": roi_id,
                "success": success_attempt is not None,
                "successful_attempt": success_attempt,
                "all_attempts": attempts
            }
            json_path = os.path.join(folder, "decoding_info.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON-Datei mit {len(attempts)} Versuchen gespeichert in {json_path}")
        except Exception as e:
            logger.warning(f"Konnte JSON nicht speichern: {e}")
    
    # Hilfsmethoden für Bildverarbeitung
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
        return cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)

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
        return self.detected_barcodes
    
    def reset(self):
        self.detected_barcodes.clear()


# ----------------------------
# Hauptfunktion für Einzelbild
# ----------------------------
def main():
    IMAGE_FILE = "ProductBarcode007.jpg"   # Hier den Dateinamen anpassen
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), IMAGE_FILE)
    
    if not os.path.exists(IMAGE_PATH):
        logger.error(f"Bilddatei nicht gefunden: {IMAGE_PATH}")
        return
    
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        logger.error(f"Bild konnte nicht geladen werden: {IMAGE_PATH}")
        return

    detector = BarcodeDetector()
    barcodes = detector.detect_barcodes_in_image(image, image_index=0, image_name=os.path.basename(IMAGE_PATH))
    
    print("\n" + "="*80)
    if not barcodes:
        logger.info("Keine Barcodes erkannt.")
        print("Die Vorverarbeitungsbilder und JSON-Dateien befinden sich im Ordner 'Bilderverarbeitung'.")
    else:
        print(f"Gefundene Barcodes ({len(barcodes)}):")
        for i, b in enumerate(barcodes, start=1):
            print(f"\nBarcode {i}: Wert={b['value']}, Typ={b['type']}")
            print(f"Erfolgreiche Vorverarbeitung: {b['decoding_details']}")
            print(f"Verarbeitungsordner: Bilderverarbeitung/{b['roi_id']}")
    print("="*80)

if __name__ == "__main__":
    main()