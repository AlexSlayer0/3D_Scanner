#Interface_v08.py
"""=======TODO-Liste v0.8=======
Objekt-Detection muss verbessert werden/Mit Bemassung der Distanz von der LIDAR-Kamera      !!!!!!!!!
Gewicht-Messung muss implementiert werden                                                   !!!!!!!!!
SAP-Integration                 (Platzhalter-Button/optional)
Lokal speichern Integration     (Formatierung?)

================================"""

import os
import csv
import sys
import cv2
import json
import logging
import platform
import numpy as np
from datetime import datetime 
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, QProgressBar, QComboBox
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal

# ==================== Konfiguration ====================
@dataclass
class AppConfig:
    """Konfigurationsklasse für die Anwendung"""
    NUM_CAMERAS: int = 4
    IMAGE_WIDTH: int = 640
    IMAGE_HEIGHT: int = 480
    DEBUG_SINGLE_CAMERA: bool = True
    DEFAULT_LANGUAGE: str = "de"
    GUI_RESOURCES_PATH: str = "GUI_Anzeige"
    LOG_LEVEL: str = "INFO"
    YOLO_MODEL_PATH: str = "models/YOLOV8s_Barcode_Detection.pt"
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json"):
        """Lädt Konfiguration aus JSON-Datei"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    return cls(**config_data)
        except Exception as e:
            logging.warning(f"Konfigurationsdatei konnte nicht geladen werden: {e}")
        return cls()

# ==================== Logging Setup ====================
def setup_logging(level: str = "INFO"):
    """Initialisiert das Logging-System"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('3d_scanner.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== Globale Konstante ====================
CONFIG = AppConfig.load_from_file()
logger = setup_logging(CONFIG.LOG_LEVEL)

# ==================== Translation Manager ====================
class TranslationManager:
    """Verwaltet mehrsprachige Texte"""
    
    def __init__(self):
        # Struktur: (Deutsch, Englisch, Italienisch)
        self.translations = {
            "start": {
                "title": ("3D Scanner Interface", "3D Scanner Interface", "3D Scanner Interfaccia"),
                "subtitle": ("Interface um den 3D-Scanner zu bedienen", "Interface to operate the 3D scanner", "Interfaccia per gestire lo scanner 3D"),
                "instruction1": ("Bitte lege den Artikel der gescannt werden soll in die Box ein", "Please place the item to be scanned in the box", "Si prega di posizionare l'articolo nella scatola"),
                "instruction2": ("Stellen Sie sicher, dass der Artikel vollständig im Sichtfeld aller Kameras liegt", "Make sure the item is completely in the field of view of all cameras", "Assicurarsi che l'articolo sia completamente nel campo visivo di tutte le telecamere"),
                "instruction3": ("Maximale Größe: 50x50x50 cm", "Maximum size: 50x50x50 cm", "Dimensione massima: 50x50x50 cm"),
                "instruction4": ("Maximales Gewicht: 20 kg", "Maximum weight: 20 kg", "Peso massimo: 20 kg"),
                "scan_btn": ("Scan Starten", "Start Scan", "Avvia Scan"),
                "save_btn": ("Lokal speichern", "Save Locally", "Salva localmente"),
                "status_title": ("System Status", "System Status", "Stato del Sistema"),
                "camera_status": ("Kamera System", "Camera System", "Sistema Fotocamera"),
                "light_status": ("Beleuchtung", "Lighting", "Illuminazione"),
                "measure_status": ("Mess-System", "Measurement System", "Sistema di Misura"),
                "scale_status": ("Waage", "Scale", "Bilancia"),
                "storage_status": ("Speicher", "Storage", "Memoria"),
                "ready": ("Bereit", "Ready", "Pronto"),
                "active": ("Aktiv", "Active", "Attivo"),
                "calibrated": ("Kalibriert", "Calibrated", "Calibrato"),
                "connected": ("Verbunden", "Connected", "Connesso"),
                "available": ("Verfügbar", "Available", "Disponibile"),
                "refresh_btn": ("Status aktualisieren", "Refresh Status", "Aggiorna Stato"),
                "check_camera": ("Kamera prüfen", "Check Camera", "Controlla Fotocamera"),
                "check_light": ("Beleuchtung prüfen", "Check Lighting", "Controlla Illuminazione"),
                "check_measure": ("Mess-System prüfen", "Check Measurement", "Controlla Sistema Misura"),
                "calibrate_scale": ("Waage kalibrieren", "Calibrate Scale", "Calibra Bilancia"),
                "check_storage": ("Speicher prüfen", "Check Storage", "Controlla Memoria")
            },
            "photo": {
                "title": ("Foto-Auswahl", "Photo Selection", "Selezione Foto"),
                "retry_btn": ("Wiederholen", "Retake", "Ripeti"),
                "discard_btn": ("Verwerfen", "Discard", "Scarta")
            },
            "overview": {
                "title": ("Kamera-Übersicht", "Camera Overview", "Panoramica Fotocamera"),
                "dimensions": ("Abmessungen:", "Dimensions:", "Dimensioni:"),
                "weight": ("Gewicht:", "Weight:", "Peso:"),
                "mm": ("mm", "mm", "mm"),
                "kg": ("kg", "kg", "kg")
            },
            "storage": {
                "title": ("Speicher Option", "Storage Options", "Opzioni di Memorizzazione"),
                "no_barcodes": ("Keine Barcodes erkannt", "No barcodes detected", "Nessun codice a barre rilevato"),
                "sap_btn": ("SAP-Eintrag", "SAP Entry", "SAP Entry"),
                "save_btn": ("Lokal speichern", "Save Locally", "Salva localmente"),
                "restart_btn": ("Neu Beginnen", "Restart", "Riavvia"),
                "add_barcode_btn": ("Weiteren Barcode hinzufügen", "Add another barcode", "Aggiungi altro codice"),
                "barcode_label": ("Barcode:", "Barcode:", "Codice a barre:"),
                "article_number_label": ("Artikelnummer:", "Article number:", "Numero articolo:"),
                "type_label": ("Typ:", "Type:", "Tipo:"),
                "source_label": ("Quelle:", "Source:", "Fonte:"),
                "manual_entry": ("Manuelle Eingabe", "Manual Entry", "Ingresso Manuale"),
                "detected": ("Erkannt", "Detected", "Rilevato"),
                "manual": ("Manuell", "Manual", "Manuale"),
                "for_ean13": ("(EAN13 als Barcode)", "(EAN13 as barcode)", "(EAN13 come codice)"),
                "for_other": ("(andere als Artikelnummer)", "(other as article number)", "(altro come numero articolo)")
            },
            "messagebox": {
                "camera_error": ("Kamerafehler", "Camera Error", "Errore Fotocamera"),
                "measurement_error": ("Messfehler", "Measurement Error", "Errore di Misura"),
                "storage_error": ("Speicherfehler", "Storage Error", "Errore di Memoria"),
                "data_loss_confirm": ("Datenverlust bestätigen", "Confirm Data Loss", "Conferma Perdita Dati"),
                "data_loss_message": ("Möchten Sie wirklich zurück zur Startseite? Alle erfassten Daten gehen verloren.", "Do you really want to go back to the start page? All captured data will be lost.", "Vuoi davvero tornare alla pagina iniziale? Tutti i dati acquisiti saranno persi."),
                "cancel_confirm": ("Abbrechen", "Cancel", "Annulla"),
                "scan_aborted_title": ("Scan abgebrochen", "Scan Aborted", "Scansione Annullata"),
                "scan_aborted_message": ("Der Scan wurde abgebrochen.", "The scan has been aborted.", "La scansione è stata annullata."),
                "scan_completed_title": ("Scan abgeschlossen", "Scan Completed", "Scansione Completata"),
                "scan_completed_message": ("Der Scan war erfolgreich!\nDie Daten stehen nun zur Verfügung.", "The scan was successful!\nThe data is now available.", "La scansione è stata completata con successo!\nI dati sono ora disponibili."),
                "no_images_title": ("Keine Bilder", "No Images", "Nessuna Immagine"),
                "no_images_message": ("Bitte nehmen Sie zuerst Bilder auf, bevor Sie fortfahren.", "Please take pictures first before continuing.", "Per favore scatta prima le foto prima di continuare."),
                "no_barcodes_title": ("Keine Barcodes", "No Barcodes", "Nessun Codice a Barre"),
                "no_barcodes_message": ("Es wurden keine Barcodes zum Speichern gefunden.", "No barcodes were found to save.", "Non è stato trovato alcun codice a barre da salvare."),
                "save_error_title": ("Speicherfehler", "Save Error", "Errore di Salvataggio"),
                "save_error_message": ("Fehler beim Speichern der Daten.", "Error saving data.", "Errore durante il salvataggio dei dati."),
                "save_success_title": ("Erfolgreich gespeichert", "Save Successful", "Salvataggio Riuscito"),
                "save_success_message": ("{count} Barcode(s) wurden lokal gespeichert.", "{count} barcode(s) have been saved locally.", "{count} codice(i) a barre sono stati salvati localmente."),
                "sap_integration_title": ("SAP-Integration", "SAP Integration", "Integrazione SAP"),
                "sap_integration_message": ("SAP-Integration würde jetzt gestartet werden...", "SAP Integration would now be started...", "L'integrazione SAP verrà ora avviata..."),
                "save_local_title": ("Lokales Speichern", "Local Save", "Salvataggio Locale"),
                "save_local_message": ("Daten würden jetzt lokal gespeichert werden...", "Data would now be saved locally...", "I dati verrebbero ora salvati localmente...")
            }
        }
        
        # Sprach-Mapping: 0=Deutsch, 1=Englisch, 2=Italienisch
        self.language_map: Dict[str, int] = {"de": 0, "en": 1, "it": 2}
    
    def get_text(self, language: str, page: str, key: str) -> str:
        """Holt übersetzten Text für gegebene Sprache, Seite und Schlüssel"""
        lang_index = self.language_map.get(language, 0)  # Default zu Deutsch
        page_dict = self.translations.get(page, {})
        text_tuple = page_dict.get(key, ("[FEHLER]", "[ERROR]", "[ERRORE]"))
        
        # Sicherstellen, dass wir immer einen String zurückgeben
        if isinstance(text_tuple, tuple) and len(text_tuple) > lang_index:
            return text_tuple[lang_index]
        return f"[{key}]"

# ==================== Camera Manager ====================
class CameraManager:
    """Verwaltet Kamerazugriff und Bildaufnahme"""
    
    def __init__(self, debug_single_camera: bool = CONFIG.DEBUG_SINGLE_CAMERA):
        self.debug_single_camera = debug_single_camera
        self.available_cameras = self._find_cameras()
        logger.info(f"Verfügbare Kameras gefunden: {self.available_cameras}")
    
    def _get_camera_backend(self) -> int:
        """Bestimmt den passenden Camera Backend für das Betriebssystem"""
        if platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY
    
    def _find_cameras(self) -> List[int]:
        """Findet verfügbare Kameras"""
        available: List[int] = []
        backend = self._get_camera_backend()
        
        for i in range(CONFIG.NUM_CAMERAS):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                logger.warning(f"Fehler beim Zugriff auf Kamera {i}: {e}")
        
        return available
    
    def _enable_flash(self):
        """Aktiviert den Blitz (Platzhalter)"""
        # BLITZ-IMPLEMENTIERUNG HIER EINFÜGEN
        # GPIO, serielle Schnittstelle
        pass
    
    def _disable_flash(self):
        """Deaktiviert den Blitz (Platzhalter)"""
        pass
    
    def _make_placeholder(self, camera_id: int = -1) -> np.ndarray:
        """Erstellt ein Platzhalterbild für fehlende Kameras"""
        img = np.zeros((CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 3), dtype=np.uint8)
        text = f"Kamera {camera_id} nicht verfügbar" if camera_id >= 0 else "BILD NICHT AUFGENOMMEN"
        cv2.putText(img, text, 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 255), 2)
        return img
    
    def take_picture(self, camera_id: int) -> np.ndarray:
        """Nimmt ein Bild mit der angegebenen Kamera auf"""
        if camera_id not in self.available_cameras:
            logger.warning(f"Kamera {camera_id} nicht verfügbar")
            return self._make_placeholder(camera_id)
        
        backend = self._get_camera_backend()
        
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if not cap.isOpened():
                logger.error(f"Kamera {camera_id} konnte nicht geöffnet werden")
                cap.release()
                return self._make_placeholder(camera_id)
            
            self._enable_flash()
            ret, frame = cap.read()
            self._disable_flash()
            cap.release()
            
            if ret:
                logger.info(f"Bild erfolgreich von Kamera {camera_id} aufgenommen")
                return frame
            else:
                logger.error(f"Bildaufnahme von Kamera {camera_id} fehlgeschlagen")
                return self._make_placeholder(camera_id)
                
        except Exception as e:
            logger.error(f"Fehler bei Bildaufnahme von Kamera {camera_id}: {e}")
            return self._make_placeholder(camera_id)
        
    def take_all_pictures(self) -> List[np.ndarray]:
        """Nimmt Bilder von allen Kameras auf"""
        images: List[np.ndarray] = []
        
        if self.debug_single_camera:
            # Debug: Eine Kamera für alle Bilder
            logger.debug("Debug-Modus: Verwende eine Kamera für alle Bilder")
            for i in range(CONFIG.NUM_CAMERAS):
                img = self.take_picture(0)
                images.append(img)
        else:
            # Normal: Jede Kamera macht ein Bild
            for i in range(CONFIG.NUM_CAMERAS):
                if i < len(self.available_cameras):
                    img = self.take_picture(i)
                else:
                    img = self._make_placeholder(i)
                images.append(img)
        
        return images

# ==================== Detection Manager ====================
class DetectionManager:
    def __init__(self):
        self.yolo_model = None
        self.barcode_detector = None  # Wird später initialisiert
        self.all_barcodes: List[Dict[str, Any]] = []
        
    def load_yolo_model(self, model_path: str = CONFIG.YOLO_MODEL_PATH):
        """Lädt das YOLO-Modell (einmalig)"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            logger.info(f"YOLO-Modell geladen von {model_path}")
        except ImportError as e:
            logger.error(f"YOLO nicht verfügbar: {e}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des YOLO-Modells: {e}")
    
    def run_yolo_detection(self, images: List[np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
        """Führt YOLO-Objekterkennung durch"""
        all_dimensions: List[str] = []
        all_frames: List[np.ndarray] = []
        
        # Direkt importieren und verwenden
        try:
            # Importiere das Modul neu
            import workers.BoundingBox_Yolo03 as yolo_module
            logger.info("BoundingBox_Yolo03 erfolgreich importiert")
        except ImportError as e:
            logger.error(f"YOLO-Modul nicht gefunden: {e}")
            # Fallback: Leere Ergebnisse
            for _ in range(len(images)):
                all_dimensions.append("0 x 0")
                all_frames.append(None)
            return all_dimensions, all_frames
        
        for idx, frame in enumerate(images):
            logger.info(f"Verarbeite Bild {idx} für YOLO-Erkennung")
            
            if frame is None:
                logger.warning(f"Bild {idx} ist None")
                all_dimensions.append("0 x 0")
                all_frames.append(None)
                continue

            try:
                # Stelle sicher, dass das Bild das richtige Format hat
                if frame is not None and isinstance(frame, np.ndarray):
                    logger.debug(f"Bild {idx} - Größe: {frame.shape}, Typ: {frame.dtype}")
                    
                    # WICHTIG: Konvertiere BGR zu RGB für YOLO
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    # Führe YOLO-Erkennung durch
                    boxes_info = yolo_module.get_boxes_and_dimensions(frame_rgb)
                    logger.debug(f"Bild {idx}: {len(boxes_info)} Boxen erkannt")

                    if boxes_info:
                        dim_str = f"{boxes_info[0]['width']} x {boxes_info[0]['height']}"
                        logger.info(f"Bild {idx}: YOLO erkannt - {dim_str}")
                    else:
                        dim_str = "0 x 0"
                        logger.warning(f"Bild {idx}: Keine Boxen erkannt")

                    all_dimensions.append(dim_str)

                    # Annotiertes Bild erstellen
                    try:
                        # Verwende das Original-BGR-Bild für die Annotation
                        frame_with_boxes = yolo_module.draw_boxes(frame.copy(), boxes_info)
                        all_frames.append(frame_with_boxes)
                        logger.debug(f"Bild {idx}: Annotiertes Frame erstellt")
                    except Exception as draw_e:
                        logger.error(f"Fehler beim Zeichnen der Boxen Bild {idx}: {draw_e}")
                        # Fallback: Originalbild
                        all_frames.append(frame)
                        
                else:
                    logger.error(f"Bild {idx} hat ungültiges Format: {type(frame)}")
                    all_dimensions.append("0 x 0")
                    all_frames.append(None)
                
            except Exception as e:
                logger.error(f"Fehler bei YOLO-Erkennung Bild {idx}: {e}", exc_info=True)
                all_dimensions.append("0 x 0")
                all_frames.append(None)
                
        logger.info(f"YOLO-Erkennung abgeschlossen. Dimensionen: {all_dimensions}")
        return all_dimensions, all_frames
    

    def run_barcode_detection(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Erkennt alle Barcodes in den Bildern"""
        try:
            # Importiere die BarcodeDetector Klasse
            from workers.BarCode_v02 import BarcodeDetector
            
            # Initialisiere Detector
            detector = BarcodeDetector()
            self.all_barcodes = []
            
            image_names = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]
            
            for idx, img in enumerate(images):
                if img is None:
                    logger.warning(f"Bild {idx} ist None - überspringe")
                    continue
                
                img_name = image_names[idx] if idx < len(image_names) else f"Bild_{idx}"
                logger.info(f"Analysiere Bild {idx} ({img_name}) auf Barcodes...")
                
                try:
                    # Erkenne Barcodes in diesem Bild
                    barcodes_in_image = detector.detect_barcodes_in_image(img, idx, img_name)
                    
                    if barcodes_in_image:
                        logger.info(f"Bild {idx}: {len(barcodes_in_image)} Barcode(s) erkannt")
                        self.all_barcodes.extend(barcodes_in_image)
                    else:
                        logger.info(f"Bild {idx}: Keine Barcodes erkannt")
                        
                except Exception as e:
                    logger.error(f"Fehler bei Barcode-Erkennung Bild {idx}: {e}")
            
            logger.info(f"Insgesamt {len(self.all_barcodes)} Barcodes in {len(images)} Bildern erkannt")
            
            # Konvertiere zu einfachem Format für die GUI
            simple_barcodes = []
            for barcode in self.all_barcodes:
                simple_barcodes.append({
                    "found": True,
                    "value": barcode.get("value"),
                    "type": barcode.get("type"),
                    "image_index": barcode.get("image_index", 0),
                    "cropped_image": barcode.get("cropped_image")
                })
            
            return simple_barcodes
            
        except Exception as e:
            logger.error(f"Fehler in run_barcode_detection: {e}")
            return []



# ==================== Parallel Worker ====================
class ParallelWorker(QThread):   
    output_received = pyqtSignal(str, object)  # (task_name, result)
    progress_updated = pyqtSignal(int)  # Fortschritt in %
    finished = pyqtSignal()

    def __init__(self, images: List[np.ndarray]):
        super().__init__()
        self.images = images
        self.detection_manager = DetectionManager()
        self.progress = 0

    def _update_progress(self, increment: int):
        """Aktualisiert den Fortschritt"""
        self.progress += increment
        self.progress_updated.emit(self.progress)

    def run(self):
        """Führt parallele Verarbeitung mit ThreadPoolExecutor durch"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Alle Tasks parallel starten
            futures = {
                executor.submit(self._run_yolo_task): "yolo",
                executor.submit(self._run_barcode_task): "barcode",
                executor.submit(self._run_weight_task): "weight"
            }
            
            completed = 0
            total = len(futures)
            
            # Auf alle Futures warten und Ergebnisse verarbeiten
            for future in concurrent.futures.as_completed(futures):
                task_type = futures[future]
                try:
                    result = future.result()
                    self._process_result(task_type, result)
                except Exception as e:
                    logger.error(f"Fehler in {task_type} Task: {e}")
                
                completed += 1
                progress = int((completed / total) * 100)
                self.progress_updated.emit(progress)
        
        self.finished.emit()
    
    def _run_yolo_task(self):
        """Führt YOLO-Erkennung durch"""
        try:
            dimensions, frames = self.detection_manager.run_yolo_detection(self.images)
            return {"dimensions": dimensions, "frames": frames}
        except Exception as e:
            logger.error(f"YOLO Task Fehler: {e}")
            return {"dimensions": [], "frames": []}
    
    def _run_barcode_task(self):
        """Führt Barcode-Erkennung durch"""
        try:
            barcodes = self.detection_manager.run_barcode_detection(self.images)
            return {"barcodes": barcodes}
        except Exception as e:
            logger.error(f"Barcode Task Fehler: {e}")
            return {"barcodes": []}
    
    def _run_weight_task(self):
        """Führt Gewichtsmessung durch"""
        try:
            import workers.Gewichts_Messung
            weight = workers.Gewichts_Messung.get_weight()
            return {"weight": weight}
        except ImportError as e:
            logger.error(f"Gewichtsmodul nicht verfügbar: {e}")
            return {"weight": "Undefiniert"}
        except Exception as e:
            logger.error(f"Gewicht Task Fehler: {e}")
            return {"weight": "Undefiniert"}
    
    def _process_result(self, task_type: str, result: dict):
        """Verarbeitet Ergebnisse der Tasks"""
        if task_type == "yolo":
            dimensions = result.get("dimensions", [])
            frames = result.get("frames", [])
            self.output_received.emit("Abmessung", dimensions)
            self.output_received.emit("yolo_frames", frames)
        elif task_type == "barcode":
            barcodes = result.get("barcodes", [])
            for barcode in barcodes:
                self.output_received.emit("barcode", barcode)
        elif task_type == "weight":
            weight = result.get("weight", "Undefiniert")
            self.output_received.emit("weight", weight)



# ==================== Main Application ====================
class FullscreenApp(QMainWindow):
    """Hauptanwendung für den 3D-Scanner"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D-Scanner")
        self.showFullScreen()

        # Initialisierung
        self.camera = CameraManager(debug_single_camera=True)
        self.translator = TranslationManager()
        self.language = CONFIG.DEFAULT_LANGUAGE
        self.Explorer_Structure = CONFIG.GUI_RESOURCES_PATH

        # Datenvariablen
        self.abmessung: Optional[str] = None
        self.gewicht: Optional[str] = None
        self.barcode: Optional[str] = None
        self.barcode_type: Optional[str] = None

        self.images: List[Optional[np.ndarray]] = [None] * CONFIG.NUM_CAMERAS
        self.image_labels: List[Optional[QLabel]] = [None] * CONFIG.NUM_CAMERAS
        self.final_images: List[Optional[np.ndarray]] = [None] * CONFIG.NUM_CAMERAS
        self.final_image_labels: List[Optional[QLabel]] = [None] * CONFIG.NUM_CAMERAS

        self.keep: List[bool] = [True] * CONFIG.NUM_CAMERAS
        self.scan_start = False
        self.bilder_namen = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]

        # GUI Setup
        self._setup_ui()
        self.load_pages()
        self.update_buttons()

    def _setup_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        container = QWidget()
        container.setStyleSheet("background-color: #292929;")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 10)
        main_layout.setSpacing(10)
        self.setCentralWidget(container)

        # Stacked widget für die Seiten
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # Navigation-Buttons
        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(10)
        
        self.back_btn = QPushButton("←")
        self.next_btn = QPushButton("→")
        self.back_btn.setFixedSize(100, 50)
        self.next_btn.setFixedSize(100, 50)
        
        font = self.back_btn.font()
        font.setPointSize(26)
        self.back_btn.setFont(font)
        self.next_btn.setFont(font)
        
        self.back_btn.clicked.connect(self.go_back)
        self.next_btn.clicked.connect(self.go_next)

        # Navigation-Buttons Styling
        nav_style = """
            QPushButton {
                font-size: 26px;
                font-weight: bold;
                background: #3498db;
                color: #ecf0f1;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """
        self.back_btn.setStyleSheet(nav_style)
        self.next_btn.setStyleSheet(nav_style)

        bar_layout.addWidget(self.back_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(self.next_btn)
        main_layout.addLayout(bar_layout)
        
        # Tastaturkürzel
        QShortcut(QKeySequence("Left"), self, activated=self.go_back)
        QShortcut(QKeySequence("Right"), self, activated=self.go_next)
        QShortcut(QKeySequence("Escape"), self, activated=self.toggle_fullscreen)

    def toggle_fullscreen(self):
        """Wechselt zwischen Vollbild und Fenstermodus"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def set_language(self, language: str):
        """Setzt die Sprache der Anwendung"""
        self.language = language
        self.load_pages()
        self.update_buttons()
        logger.info(f"Sprache geändert zu: {language}")

    def create_flag_button(self, flag_file: str, language_code: str) -> QToolButton:
        """Erstellt einen Sprachumschalt-Button"""
        btn = QToolButton()
        btn.setIcon(QIcon(os.path.join(self.Explorer_Structure, flag_file)))
        btn.setIconSize(QSize(32, 32))
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QToolButton {
                background: #323f4d;
                border: 2px solid #5d6d7e;
                border-radius: 6px;
                padding: 5px;
            }
            QToolButton:hover {
                background: #3d566e;
                border: 2px solid #3498db;
            }
            QToolButton:pressed {
                background: #21618c;
            }
        """)
        btn.clicked.connect(lambda _, lang=language_code: self.set_language(lang))
        return btn


    def create_start_page(self) -> QWidget:
        """Erstellt die Startseite"""
        page = QWidget()
        page.setStyleSheet("background-color: #333333;")
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # Header mit Logo und Sprachbuttons
        header_widget = self.create_start_header()
        main_layout.addWidget(header_widget)

        # Hauptinhalt mit zwei Spalten
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(40)
        content_layout.setContentsMargins(0, 20, 0, 0)

        # Linke Spalte - Hauptinformationen
        left_column = self.create_start_left_column()
        content_layout.addWidget(left_column, stretch=3)

        # Rechte Spalte - Systemstatus
        right_column = self.create_start_right_column()
        content_layout.addWidget(right_column, stretch=2)

        main_layout.addWidget(content_widget, stretch=1)
        
        return page

    def create_start_header(self) -> QWidget:
        """Erstellt den Header der Startseite"""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo links
        logo_label = QLabel()
        logo_path = os.path.join(self.Explorer_Structure, "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaled(150, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("3D-SCANNER")
            logo_label.setStyleSheet("""
                color: #ecf0f1; 
                font-size: 24px; 
                font-weight: bold;
                font-family: Arial;
                padding: 10px;
                background: #323f4d;
                border-radius: 6px;
            """)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(logo_label)
        
        header_layout.addStretch()
        
        # Sprachbuttons rechts
        lang_widget = QWidget()
        lang_layout = QHBoxLayout(lang_widget)
        lang_layout.setSpacing(8)
        lang_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it") 
        btn_en = self.create_flag_button("en.png", "en")

        for btn in [btn_de, btn_it, btn_en]:
            lang_layout.addWidget(btn)

        header_layout.addWidget(lang_widget)
        return header_widget

    def create_start_left_column(self) -> QWidget:
        """Erstellt die linke Spalte der Startseite"""
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setSpacing(25)
        
        # Titel
        title_label = QLabel(self.translator.get_text(self.language, "start", "title"))
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("""
            font-size: 36px; 
            font-weight: bold; 
            color: #3498db;
            margin-bottom: 10px;
            font-family: Arial;
            padding-bottom: 15px;
            border-bottom: 2px solid #5d6d7e;
        """)
        left_layout.addWidget(title_label)

        # Untertitel
        subtitle_label = QLabel(self.translator.get_text(self.language, "start", "subtitle"))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("""
            color: #bdc3c7; 
            font-size: 18px; 
            padding: 15px 0; 
            line-height: 1.4;
        """)
        left_layout.addWidget(subtitle_label)

        # Anweisungen
        instructions_widget = QWidget()
        instructions_layout = QVBoxLayout(instructions_widget)
        instructions_layout.setSpacing(12)
        
        texts = [
            self.translator.get_text(self.language, "start", "instruction1"),
            self.translator.get_text(self.language, "start", "instruction2"),
            self.translator.get_text(self.language, "start", "instruction3"),
            self.translator.get_text(self.language, "start", "instruction4")
        ]
        
        for i, text in enumerate(texts):
            instruction_frame = QFrame()
            instruction_frame.setStyleSheet("""
                QFrame {
                    background: #323f4d;
                    border: 1px solid #5d6d7e;
                    border-radius: 6px;
                    padding: 15px;
                }
            """)
            frame_layout = QHBoxLayout(instruction_frame)
            
            # Text
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setWordWrap(True)
            label.setStyleSheet("color: #ecf0f1; font-size: 16px; line-height: 1.4;")
            frame_layout.addWidget(label, stretch=1)
            
            instructions_layout.addWidget(instruction_frame)

        left_layout.addWidget(instructions_widget)
        left_layout.addStretch()

        # Aktion-Buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setSpacing(20)
        
        scan_btn = QPushButton(self.translator.get_text(self.language, "start", "scan_btn"))
        save_btn = QPushButton(self.translator.get_text(self.language, "start", "save_btn"))
        
        # Scan-Button - Primäre Aktion
        scan_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: 600;
                padding: 16px 35px;
                border: none;
                border-radius: 6px;
                background: #3498db;
                color: #ecf0f1;
                min-width: 180px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        
        # Save-Button - Sekundäre Aktion
        save_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: 600;
                padding: 16px 35px;
                border: 2px solid #5d6d7e;
                border-radius: 6px;
                background: #323f4d;
                color: #ecf0f1;
                min-width: 180px;
            }
            QPushButton:hover {
                background: #3d566e;
                color: #ecf0f1;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        
        for btn in [scan_btn, save_btn]:
            btn.setFixedHeight(55)
            button_layout.addWidget(btn)

        scan_btn.clicked.connect(self.go_next)
        left_layout.addWidget(button_widget)

        return left_column

    def create_start_right_column(self) -> QFrame:
        """Erstellt die rechte Spalte (Systemstatus) der Startseite"""
        right_column = QFrame()
        right_column.setStyleSheet("""
            QFrame {
                background: #323f4d; 
                border: 1px solid #5d6d7e;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        right_layout = QVBoxLayout(right_column)
        right_layout.setSpacing(20)

        # Status-Überschrift
        status_title = QLabel(self.translator.get_text(self.language, "start", "status_title"))
        status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_title.setStyleSheet("""
            font-size: 22px; 
            font-weight: 600; 
            color: #3498db;
            padding: 10px 0;
            border-bottom: 1px solid #5d6d7e;
        """)
        right_layout.addWidget(status_title)

        # Status-Buttons
        status_buttons = [
            (self.translator.get_text(self.language, "start", "check_camera"), self.check_camera),
            (self.translator.get_text(self.language, "start", "check_light"), self.check_light),
            (self.translator.get_text(self.language, "start", "check_measure"), self.check_measure),
            (self.translator.get_text(self.language, "start", "calibrate_scale"), self.calibrate_scale),
            (self.translator.get_text(self.language, "start", "check_storage"), self.check_storage)
        ]

        for name, callback in status_buttons:
            status_button = self.create_status_button(name, callback)
            right_layout.addWidget(status_button)

        right_layout.addStretch()

        # Refresh Button
        refresh_btn = QPushButton(self.translator.get_text(self.language, "start", "refresh_btn"))
        refresh_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 12px; 
                background: #333333;
                color: #ecf0f1; 
                border: 1px solid #5d6d7e; 
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #3498db;
                color: #ecf0f1;
            }
        """)
        refresh_btn.setFixedHeight(45)
        right_layout.addWidget(refresh_btn)

        return right_column

    def create_status_button(self, name: str, callback) -> QPushButton:
        """Erstellt einen Status-Button"""
        button = QPushButton(name)
        button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: 500;
                padding: 12px 15px;
                border: 1px solid #5d6d7e;
                border-radius: 6px;
                background: #333333;
                color: #ecf0f1;
                text-align: left;
            }
            QPushButton:hover {
                background: #3498db;
                border-color: #3498db;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        button.setFixedHeight(45)
        if callback:
            button.clicked.connect(callback)
        return button

    def sap_integration_placeholder(self):
        """Platzhalter für SAP-Integration"""
        QMessageBox.information(self, 
                                self.translator.get_text(self.language, "start", "sap_integration_title"), 
                                self.translator.get_text(self.language, "start", "sap_integration_message"))

        logger.info("SAP-Integration Button gedrückt")
        
    def convert_to_pixmap(self, frame: np.ndarray, width: int = 300, height: int = 300) -> QPixmap:
        """Konvertiert OpenCV-Bild zu QPixmap"""
        if frame is None or (isinstance(frame, np.ndarray) and np.all(frame == 0)):
            gray_pixmap = QPixmap(width, height)
            gray_pixmap.fill(Qt.GlobalColor.lightGray)
            return gray_pixmap
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def retry_image(self, idx: int):
        """Wiederholt die Aufnahme eines einzelnen Bildes"""
        logger.info(f"Wiederhole Bild {idx+1}")
        self.scan_start = True
        new_img = self.camera.take_picture(idx)
        if new_img is not None:
            self.images[idx] = new_img
            pixmap = self.convert_to_pixmap(new_img)
            self.image_labels[idx].setPixmap(pixmap)

    def discard_image(self, idx: int):
        """Verwirft ein Bild"""
        logger.info(f"Verworfen Bild {idx+1}")
        self.scan_start = True
        self.keep[idx] = False
        label = self.image_labels[idx]
        gray_pixmap = QPixmap(label.pixmap().size())
        gray_pixmap.fill(Qt.GlobalColor.lightGray)
        label.setPixmap(gray_pixmap)

    def make_card(self, text: str) -> QLabel:
        """Erstellt eine Textkarte"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                color: #dedede;
                font-size: 17px;
                font-weight: 400;
                padding: 18px 30px;
                margin: 10px 0;
                background: transparent;
                border-bottom: 1px solid #323f4d;
                line-height: 1.5;
            }
        """)
        label.setWordWrap(True)
        return label

    def make_card_with_input(self, label_text: str = "", preset_text: str = "", placeholder: str = "") -> QFrame:
        """Erstellt eine Eingabekarte"""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        frame.setMinimumHeight(120)
        frame.setStyleSheet("""
            QFrame {
                background-color: #dedede;
                border-radius: 12px;
                border: 1px solid #bbb;
                padding: 12px;
            } QLabel {
                font-size: 18px;
                color: #333333;
            } QLineEdit {
                font-size: 20px;
                color: #333333;
                background: transparent;
                border: none;
                border-bottom: 2px solid #333333;
            }""")

        layout = QVBoxLayout(frame)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Überschrift
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Eingabefeld
        field = QLineEdit()
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if preset_text:
            field.setText(preset_text)
        if placeholder:
            field.setPlaceholderText(placeholder)

        layout.addWidget(field)
        return frame

    def _make_widget(self, item) -> QWidget:
        """Erstellt ein Widget basierend auf der Beschreibung"""
        if not isinstance(item, tuple):
            return self.make_card(str(item))
        
        widget_type = item[0]
        
        if widget_type == "custom":
            # Custom-Widget direkt zurückgeben
            return item[1]
        
        widget_creators = {
            "button": self._create_button_widget,
            "image": self._create_image_widget, 
            "ram_image": self._create_ram_image_widget,
            "ram_image_final": self._create_ram_image_final_widget,
            "title": self._create_title_widget,
            "input": self._create_input_widget,
            "text": self._create_text_widget
        }
        creator = widget_creators.get(widget_type)
        if creator:
            return creator(*item[1:])
        return self.make_card(str(item))

    def _create_text_widget(self, text: str) -> QLabel:
        """Erstellt einen Text-Widget"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                color: #BDC3C7;
                font-size: 18px;
                font-style: italic;
                padding: 30px;
            }
        """)
        label.setWordWrap(True)
        return label

    def _create_button_widget(self, text: str, callback=None) -> QPushButton:
        """Erstellt einen Button"""
        btn = QPushButton(text)
        btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: 600;
                padding: 14px 30px;
                border: none;
                border-radius: 6px;
                background: #495057;
                color: #ffffff;
                min-width: 140px;
            } QPushButton:hover {
                background: #6c757d;
            } QPushButton:pressed {
                background: #343a40;
            }
        """)
        
        if callback and callable(callback):
            btn.clicked.connect(callback)
        else:
            btn.clicked.connect(lambda: print(f"Button '{text}' gedrückt"))
        
        return btn

    def _create_image_widget(self, base_name: str) -> QLabel:
        """Erstellt ein Bild-Widget"""
        label = QLabel()
        path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            test_path = os.path.join(self.Explorer_Structure, base_name + ext)
            if os.path.exists(test_path):
                path = test_path
                break
        
        if path:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(250, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(pixmap)
            else:
                label.setText(f"Bild konnte nicht geladen werden:\n{path}")
        else:
            label.setText(f"Kein Bild gefunden für '{base_name}'")

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_ram_image_widget(self, idx: int) -> QLabel:
        """Erstellt ein RAM-Bild-Widget"""
        label = QLabel()
        self.image_labels[idx] = label
        if self.images[idx] is not None:
            pixmap = self.convert_to_pixmap(self.images[idx])
            label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_ram_image_final_widget(self, idx: int) -> QLabel:
        """Erstellt ein finales RAM-Bild-Widget"""
        label = QLabel()
        self.final_image_labels[idx] = label
        if self.final_images[idx] is not None:
            label.setPixmap(self.convert_to_pixmap(self.final_images[idx]))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_title_widget(self, text: str) -> QLabel:
        """Erstellt einen Titel"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dedede;")
        return label

    def _create_input_widget(self, label_text: str, placeholder: str = "", preset_text: str = "") -> QFrame:
        """Erstellt ein Eingabewidget"""
        return self.make_card_with_input(label_text, preset_text, placeholder)

    def add_page(self, title: str, widgets: List[Any]):
        """Fügt eine Seite zum Stack hinzu"""
        page = QWidget()
        page.setStyleSheet("background-color: #333333;")
        page_layout = QVBoxLayout(page)
        page_layout.setSpacing(16)

        # Taskbar für die Seite
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(20)

        # Linke Seite: Seitentitel
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #3498db;")
        title_layout.addWidget(title_label, stretch=1)

        # Sprachbuttons
        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it")
        btn_en = self.create_flag_button("en.png", "en")

        for btn in [btn_de, btn_it, btn_en]:
            title_layout.addWidget(btn)

        page_layout.addWidget(title_bar)

        # Scrollbereich
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        # Widgets hinzufügen
        for item in widgets:
            if isinstance(item, list):
                row_layout = QHBoxLayout()
                row_layout.setSpacing(12)
                for sub in item:
                    widget = self._make_widget(sub)
                    row_layout.addWidget(widget)
                layout.addLayout(row_layout)
            else:
                widget = self._make_widget(item)
                layout.addWidget(widget)

        layout.addStretch()
        scroll.setWidget(content)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)

    def load_pages(self):
        """Lädt alle Seiten neu, behält aber die aktuelle Seite bei"""
        current_page = self.stack.currentIndex() if hasattr(self, 'stack') else 0
        
        # Setze Standardwerte wenn nicht vorhanden
        if not hasattr(self, 'abmessung') or self.abmessung is None:
            self.abmessung = "Undefiniert"
        if not hasattr(self, 'gewicht') or self.gewicht is None:
            self.gewicht = "Undefiniert"
        if not hasattr(self, 'all_barcodes'):
            self.all_barcodes = []
        
        # Alte Seiten entfernen
        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        # Startseite
        start_page = self.create_start_page()
        self.stack.addWidget(start_page)

        # Gemeinsame Seitenstruktur für alle Sprachen
        page_configs = {
            "photo": {
                "title_key": "photo",
                "content": [
                    [("ram_image", 0), ("ram_image", 1)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=0: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=1: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=0: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=1: self.discard_image(idx))],
                    [("ram_image", 2), ("ram_image", 3)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=2: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=3: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=2: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=3: self.discard_image(idx))]
                ]
            },
            "overview": {
                "title_key": "overview", 
                "content": [
                    [("ram_image_final", 0), ("ram_image_final", 1)],
                    [("ram_image_final", 2), ("ram_image_final", 3)],
                    f"{self.translator.get_text(self.language, 'overview', 'dimensions')} {self.abmessung}{self.translator.get_text(self.language, 'overview', 'mm')}",
                    f"{self.translator.get_text(self.language, 'overview', 'weight')} {self.gewicht}{self.translator.get_text(self.language, 'overview', 'kg')}"
                ]
            },
            "storage": {
                "title_key": "storage",
                "content": self.get_storage_page_content()
            }
        }
        
        # Füge alle Seiten hinzu
        for page_key in ["photo", "overview", "storage"]:
            config = page_configs[page_key]
            self.add_page(
                self.translator.get_text(self.language, config["title_key"], "title"),
                config["content"]
            )
        
        # Zurück zur ursprünglichen Seite springen
        max_pages = self.stack.count()
        if current_page >= max_pages:
            current_page = max_pages - 1
        
        self.stack.setCurrentIndex(current_page)
        self.update_buttons()

    def rebeginn_application(self):
        """Startet die Anwendung von der Startseite neu"""
        if QMessageBox.question(self, self.translator.get_text(self.language, "messagebox", "data_loss_confirm"), 
                                          self.translator.get_text(self.language, "messagebox", "data_loss_message"),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Cancel:
            return
        
        self.abmessung = None
        self.gewicht = None
        self.barcode = None
        self.barcode_type = None
        self.images = [None] * CONFIG.NUM_CAMERAS
        self.final_images = [None] * CONFIG.NUM_CAMERAS
        self.keep = [True] * CONFIG.NUM_CAMERAS
        self.scan_start = False
        self.all_barcodes = []
        self.load_pages()
        self.stack.setCurrentIndex(0)
        self.update_buttons()
        logger.info("Anwendung wurde neu gestartet")


    def add_new_barcode_field(self):
        """Fügt ein neues leeres Barcode-Feld hinzu"""
        logger.info("Füge neues Barcode-Feld hinzu")
        
        if not hasattr(self, 'all_barcodes'):
            self.all_barcodes = []
        
        # Frage den Benutzer nach dem Typ (vereinfachte Version)
        dialog = QDialog(self)
        dialog.setWindowTitle("Barcode-Typ auswählen")
        dialog.setFixedSize(400, 200)
        dialog.setStyleSheet("""
            QDialog {
                background: #2C3E50;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 16px;
            }
            QPushButton {
                font-size: 14px;
                padding: 10px 20px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                margin: 5px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Welchen Typ von Barcode möchten Sie hinzufügen?")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        btn_layout = QHBoxLayout()
        
        # Knopf für Artikelnummer (Interne Materialnummer)
        btn_article = QPushButton("Interne Materialnummer")
        btn_article.setMinimumHeight(50)
        
        # Knopf für EAN-Code
        btn_ean = QPushButton("EAN-Code")
        btn_ean.setMinimumHeight(50)
        
        btn_layout.addWidget(btn_article)
        btn_layout.addWidget(btn_ean)
        
        layout.addLayout(btn_layout)
        
        # Neue Barcode-ID bestimmen
        new_index = len(self.all_barcodes)
        
        def add_barcode(is_article: bool):
            if is_article:
                new_barcode = {
                    "value": "", 
                    "type": "CODE128", 
                    "image_index": -1, 
                    "cropped_image": None,
                    "is_article_number": True,
                    "source": "manual"
                }
            else:
                new_barcode = {
                    "value": "", 
                    "type": "EAN13", 
                    "image_index": -1, 
                    "cropped_image": None,
                    "is_article_number": False,
                    "source": "manual"
                }
            
            self.all_barcodes.append(new_barcode)
            dialog.accept()
            
            # Seite neu laden, um das neue Feld anzuzeigen
            self.load_pages()
            
            # Direkt zur Storage Page springen (Index 3)
            if self.stack.count() > 3:
                self.stack.setCurrentIndex(3)
        
        btn_article.clicked.connect(lambda: add_barcode(True))
        btn_ean.clicked.connect(lambda: add_barcode(False))
        
        dialog.exec()

    def find_barcode_widgets(self, widget: QWidget) -> List[Tuple[QWidget, int]]:
        """Findet alle Barcode-Widgets und ihre Indizes"""
        barcode_widgets = []
        
        # Prüfe ob dies ein Barcode-Widget ist
        if isinstance(widget, QFrame) and widget.objectName().startswith("barcode_widget_"):
            try:
                # Extrahiere den Index aus dem objectName
                index_str = widget.objectName().replace("barcode_widget_", "")
                index = int(index_str)
                barcode_widgets.append((widget, index))
            except:
                pass
        
        # Rekursiv Kinder durchsuchen
        for child in widget.children():
            if isinstance(child, QWidget):
                barcode_widgets.extend(self.find_barcode_widgets(child))
        
        return barcode_widgets

    def get_storage_page_content(self) -> List[Any]:
        """Erzeugt den dynamischen Inhalt für die Storage Pages"""
        content = []
        
        # Übersetzungen laden
        no_barcodes_text = self.translator.get_text(self.language, "storage", "no_barcodes")
        add_barcode_btn_text = self.translator.get_text(self.language, "storage", "add_barcode_btn")
        
        # Stelle sicher, dass all_barcodes existiert
        if not hasattr(self, 'all_barcodes'):
            self.all_barcodes = []
        
        # Füge Barcode-Einträge hinzu
        if self.all_barcodes:
            self.barcode_input_widgets = []
            
            # Trenne EAN13 und Artikelnummern für bessere Darstellung
            ean13_barcodes = [b for b in self.all_barcodes if not b.get('is_article_number', False)]
            article_numbers = [b for b in self.all_barcodes if b.get('is_article_number', False)]
            
            # Zeige EAN13 Barcodes zuerst
            if ean13_barcodes:
                content.append(("text", "EAN13 Barcodes:"))
                for i, barcode in enumerate(ean13_barcodes):
                    barcode_card = ("custom", self.create_editable_barcode_widget(barcode, i))
                    content.append([barcode_card])
            
            # Zeige Artikelnummern
            if article_numbers:
                content.append(("text", "Artikelnummer:"))
                start_idx = len(ean13_barcodes)
                for j, article in enumerate(article_numbers):
                    idx = start_idx + j
                    article_card = ("custom", self.create_editable_barcode_widget(article, idx))
                    content.append([article_card])
        else:
            # Keine Barcodes gefunden - Standardformular erstellen UND in all_barcodes speichern
            content.append([("text", no_barcodes_text)])
            
            # Standard-EAN13 Feld erstellen und in all_barcodes speichern
            empty_ean = {
                "value": "", 
                "type": "EAN13", 
                "image_index": -1, 
                "cropped_image": None,
                "is_article_number": False,
                "source": "manual"
            }
            self.all_barcodes.append(empty_ean)
            ean_card = ("custom", self.create_editable_barcode_widget(empty_ean, 0))
            content.append([ean_card])
            
            # Standard-Artikelnummer Feld erstellen und in all_barcodes speichern
            empty_article = {
                "value": "", 
                "type": "CODE128",  # Standard für Artikelnummern
                "image_index": -1, 
                "cropped_image": None,
                "is_article_number": True,
                "source": "manual"
            }
            self.all_barcodes.append(empty_article)
            article_card = ("custom", self.create_editable_barcode_widget(empty_article, 1))
            content.append([article_card])
        
        # Button "Weiteren Barcode hinzufügen"
        content.append([
            ("button", add_barcode_btn_text, self.add_new_barcode_field)
        ])
        
        # Buttons am Ende (SAP-Eintrag, Lokal speichern, Neu beginnen)
        content.append([
            ("button", self.translator.get_text(self.language, "storage", "sap_btn"), self.sap_integration_placeholder),
            ("button", self.translator.get_text(self.language, "storage", "save_btn"), self.save_all_data_csv),
            ("button", self.translator.get_text(self.language, "storage", "restart_btn"), self.rebeginn_application)
        ])
        
        return content

    def create_editable_barcode_widget(self, barcode: Dict, index: int) -> QFrame:
        """Erstellt ein bearbeitbares Barcode-Widget mit Eingabefeldern"""
        frame = QFrame()
        frame.setObjectName(f"barcode_widget_{index}")
        
        # Bestimme ob Artikelnummer oder EAN13
        is_article_number = barcode.get('is_article_number', False)
        source = barcode.get('source', 'manual')
        
        # Größere Bildabmessungen
        IMAGE_WIDTH = 500  # Statt 250
        IMAGE_HEIGHT = 350  # Statt 150
        
        # Unterschiedliches Styling für Artikelnummer vs EAN13
        if is_article_number:
            frame_style = """
                QFrame {
                    background: #2C3E50;
                    border: 2px solid #E67E22;
                    border-radius: 12px;
                    padding: 20px;
                }
            """
            label_type = self.translator.get_text(self.language, "storage", "article_number_label")
            type_hint = self.translator.get_text(self.language, "storage", "for_other")
        else:
            frame_style = """
                QFrame {
                    background: #34495E;
                    border: 2px solid #3498db;
                    border-radius: 12px;
                    padding: 20px;
                }
            """
            label_type = self.translator.get_text(self.language, "storage", "barcode_label")
            type_hint = self.translator.get_text(self.language, "storage", "for_ean13")
        
        frame.setStyleSheet(frame_style + """
            QLabel {
                color: #ECF0F1;
            }
            QLineEdit, QComboBox {
                background: #2C3E50;
                border: 1px solid #5d6d7e;
                border-radius: 6px;
                padding: 8px;
                color: #ECF0F1;
                font-size: 14px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #3498db;
            }
        """)
        
        layout = QHBoxLayout(frame)
        layout.setSpacing(20)
        
        # Linke Seite: Barcode-Bild mit Farbcodierung
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Übersetzte Text für die GUI
        type_label_text = self.translator.get_text(self.language, "storage", "type_label")
        source_label_text = self.translator.get_text(self.language, "storage", "source_label")
        
        if "cropped_image" in barcode and barcode["cropped_image"] is not None:
            cropped_img = barcode["cropped_image"]
            
            # Farbige Umrandung basierend auf Typ
            border_color = (255, 165, 0) if is_article_number else (0, 255, 0)  # Orange für Artikel, Grün für EAN
            
            if len(cropped_img.shape) == 3:
                bordered_img = cv2.copyMakeBorder(cropped_img, 8, 8, 8, 8, 
                                                cv2.BORDER_CONSTANT, value=border_color)
                if cropped_img.shape[2] == 3:
                    bordered_img = cv2.cvtColor(bordered_img, cv2.COLOR_BGR2RGB)
            else:
                bordered_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
                bordered_img = cv2.copyMakeBorder(bordered_img, 8, 8, 8, 8,
                                                cv2.BORDER_CONSTANT, value=border_color)
            
            # VERGRÖSSERT: Neue Bildgröße
            pixmap = self.convert_to_pixmap(bordered_img, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Klick-Event für Vergrößerung hinzufügen
            image_label.mousePressEvent = lambda event, img=cropped_img, barcode_type=("Artikelnummer" if is_article_number else "EAN"): self.show_enlarged_image(img, barcode_type)
            image_label.setCursor(Qt.CursorShape.PointingHandCursor)
            image_label.setToolTip("Klicken zum Vergrößern")
            
            image_layout.addWidget(image_label)
            
            # Bildquelle
            image_names = ["ISO Bild", "Top Bild", "Right Bild", "Behind Bild"]
            img_idx = barcode.get('image_index', 0)
            if img_idx >= 0 and img_idx < len(image_names):
                source_text = f"{source_label_text} {image_names[img_idx]}"
            else:
                source_text = f"{source_label_text} {self.translator.get_text(self.language, 'storage', 'manual')}"
            
            source_label = QLabel(source_text)
            source_label.setStyleSheet("font-size: 12px; color: #BDC3C7; margin-top: 8px;")
            source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_layout.addWidget(source_label)
        else:
            placeholder = QLabel("Kein Bild verfügbar")
            placeholder.setStyleSheet("""
                color: #BDC3C7;
                font-style: italic;
                font-size: 14px;
            """)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setFixedSize(IMAGE_WIDTH, IMAGE_HEIGHT)  # Auch Platzhalter vergrößern
            image_layout.addWidget(placeholder)
            
            source_label = QLabel(f"{source_label_text} {self.translator.get_text(self.language, 'storage', 'manual')}")
            source_label.setStyleSheet("font-size: 12px; color: #BDC3C7; margin-top: 8px;")
            source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_layout.addWidget(source_label)
        layout.addWidget(image_container)
        
        # Rechte Seite: Bearbeitbare Informationen
        info_container = QWidget()
        info_layout = QVBoxLayout(info_container)
        info_layout.setSpacing(15)
        
        # Typ-Anzeige (Artikelnummer oder Barcode) mit Hinweis
        type_header = QLabel(f"{label_type} {type_hint}")
        type_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #F39C12;" if is_article_number else "font-size: 16px; font-weight: bold; color: #3498db;")
        info_layout.addWidget(type_header)
        
        # Eingabefeld für Wert
        barcode_input = QLineEdit()
        barcode_input.setText(barcode.get('value', ''))
        
        if is_article_number:
            barcode_input.setPlaceholderText("Artikelnummer hier eingeben...")
        else:
            barcode_input.setPlaceholderText("EAN13 Barcode hier eingeben...")
        
        barcode_input.textChanged.connect(lambda text: self.update_barcode_value(index, text))
        info_layout.addWidget(barcode_input)
        
        # Barcode-Typ Auswahl
        type_sublabel = QLabel(type_label_text)
        type_sublabel.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        info_layout.addWidget(type_sublabel)
        
        type_combo = QComboBox()
        
        type_combo.wheelEvent = lambda event: None  # Ignoriere alle Wheel-Events

        # Barcode-Typen mit Trennung
        type_combo.addItem("EAN13 - Produkt-Barcode")
        type_combo.addItem("EAN8")
        type_combo.addItem("UPC-A")
        type_combo.addItem("UPC-E")
        type_combo.addItem("CODE128 - Artikelnummer")
        type_combo.addItem("CODE39 - Artikelnummer")
        type_combo.addItem("ITF - Artikelnummer")
        type_combo.addItem("QR - Artikelnummer")
        type_combo.addItem("Andere - Artikelnummer")
        
        # Aktuellen Typ setzen
        current_type = barcode.get('type', 'CODE128' if is_article_number else 'EAN13')
        
        # Mapping für die Anzeige
        type_mapping = {
            'EAN13': "EAN13 - Produkt-Barcode",
            'EAN8': "EAN8",
            'UPC-A': "UPC-A",
            'UPC-E': "UPC-E",
            'CODE128': "CODE128 - Artikelnummer",
            'CODE39': "CODE39 - Artikelnummer",
            'ITF': "ITF - Artikelnummer",
            'QR': "QR - Artikelnummer"
        }
        
        display_type = type_mapping.get(current_type, "Andere - Artikelnummer")
        type_combo.setCurrentText(display_type)
        
        # Bei Typänderung: is_article_number aktualisieren
        type_combo.currentTextChanged.connect(lambda text: self.update_barcode_type_and_status(index, text))
        info_layout.addWidget(type_combo)
        
        # Status-Anzeige
        status_label = QLabel()
        if barcode.get('value'):
            if is_article_number:
                status_text = f"Artikelnummer {self.translator.get_text(self.language, 'storage', source)}"
                status_color = "#E67E22"  # Orange
            else:
                status_text = f"EAN13 Barcode {self.translator.get_text(self.language, 'storage', source)}"
                status_color = "#2ecc71"  # Grün
        else:
            if is_article_number:
                status_text = "Artikelnummer bitte manuell eingeben"
                status_color = "#e74c3c"  # Rot
            else:
                status_text = "EAN13 Barcode bitte manuell eingeben"
                status_color = "#e74c3c"  # Rot
        
        status_label.setText(status_text)
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; margin-top: 10px;")
        info_layout.addWidget(status_label)
        
        info_layout.addStretch()
        layout.addWidget(info_container, stretch=1)
        
        # Referenzen speichern
        frame.barcode_input = barcode_input
        frame.type_combo = type_combo
        frame.status_label = status_label
        frame.is_article_number = is_article_number
        
        return frame


    def update_barcode_value(self, index: int, value: str):
        """Aktualisiert den Barcode-Wert"""
        if index < len(self.all_barcodes):
            self.all_barcodes[index]['value'] = value
            
            # Status aktualisieren
            if hasattr(self, 'barcode_input_widgets') and index < len(self.barcode_input_widgets):
                frame = self.barcode_input_widgets[index]
                if value.strip():
                    frame.status_label.setText("Barcode erkannt/bearbeitet")
                    frame.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; margin-top: 10px;")
                else:
                    frame.status_label.setText("Kein Barcode erkannt - Bitte manuell eingeben")
                    frame.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; margin-top: 10px;")

    def update_barcode_type_and_status(self, index: int, display_type: str):
        """Aktualisiert Barcode-Typ und is_article_number Flag"""
        if index >= len(self.all_barcodes):
            return
        
        # Extrahiere den reinen Typ aus der Anzeige
        if " - " in display_type:
            barcode_type = display_type.split(" - ")[0]
        else:
            barcode_type = display_type
        
        # Bestimme ob Artikelnummer (alles außer EAN13)
        is_article_number = (barcode_type != "EAN13")
        
        # Update in der Datenstruktur
        self.all_barcodes[index]['type'] = barcode_type
        self.all_barcodes[index]['is_article_number'] = is_article_number
        
        logger.info(f"Barcode {index}: Typ auf {barcode_type} gesetzt, Artikelnummer={is_article_number}")
        
        # GUI aktualisieren (Seite neu laden für Farbänderung)
        self.load_pages()
        if self.stack.count() > 3:
            self.stack.setCurrentIndex(3)

    def save_all_data_csv(self):
        """Speichert alle Daten in CSV mit Trennung von EAN und Artikelnummern"""
        try:
            # Ordner für Bilder erstellen
            scan_folder = f"Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(scan_folder, exist_ok=True)
            
            # ISO-Bild speichern
            iso_bild_pfad = ""
            if self.images and len(self.images) > 0 and self.images[0] is not None:
                iso_bild_datei = "iso_bild.jpg"
                iso_bild_pfad = os.path.join(scan_folder, iso_bild_datei)
                if len(self.images[0].shape) == 3 and self.images[0].shape[2] == 3:
                    bgr_img = cv2.cvtColor(self.images[0], cv2.COLOR_RGB2BGR)
                else:
                    bgr_img = self.images[0]
                cv2.imwrite(iso_bild_pfad, bgr_img)
            
            # CSV-Datei erstellen
            csv_datei = os.path.join(scan_folder, "scanner_daten.csv")
            
            with open(csv_datei, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                
                # NEUE Kopfzeile mit separaten Spalten für EAN und Artikelnummer
                writer.writerow([
                    "Interne_Materialnummer",
                    "Gewicht [kg]",
                    "Abmessungen L [mm]",
                    "Abmessungen B [mm]", 
                    "Abmessungen H [mm]",
                    "EAN-Code",
                    "Iso_Bild"
                ])
                
                # Abmessungen parsen
                laenge = breite = hoehe = 0
                if self.abmessung and self.abmessung != "Undefiniert":
                    try:
                        teile = self.abmessung.split(" x ")
                        if len(teile) >= 2:
                            laenge = float(teile[0])
                            breite = float(teile[1])
                            hoehe = float(teile[2]) if len(teile) > 2 else 0
                    except:
                        pass
                
                # Gewicht parsen
                gewicht = 0
                if self.gewicht and self.gewicht != "Undefiniert":
                    try:
                        gewicht_str = str(self.gewicht).replace("kg", "").strip()
                        gewicht = float(gewicht_str)
                    except:
                        pass
                
                # Trenne EAN13 und Artikelnummern
                ean_codes = []
                article_numbers = []
                
                if hasattr(self, 'all_barcodes'):
                    for barcode in self.all_barcodes:
                        value = barcode.get('value', '').strip()
                        if not value:
                            continue
                        
                        if barcode.get('is_article_number', False):
                            article_numbers.append(value)
                        else:
                            ean_codes.append(value)
                
                # Für jede Kombination eine Zeile erstellen
                # Wenn keine Barcodes, eine leere Zeile
                if not ean_codes and not article_numbers:
                    writer.writerow([
                        "",  # Interne_Materialnummer
                        f"{gewicht:.3f}".replace(".", ","),
                        f"{laenge:.0f}".replace(".", ","),
                        f"{breite:.0f}".replace(".", ","),
                        f"{hoehe:.0f}".replace(".", ","),
                        "",  # EAN-Code
                        iso_bild_pfad if iso_bild_pfad else ""
                    ])
                else:
                    # Kombiniere alle Möglichkeiten
                    for ean in (ean_codes if ean_codes else [""]):
                        for article in (article_numbers if article_numbers else [""]):
                            writer.writerow([
                                article,
                                f"{gewicht:.3f}".replace(".", ","),
                                f"{laenge:.0f}".replace(".", ","),
                                f"{breite:.0f}".replace(".", ","),
                                f"{hoehe:.0f}".replace(".", ","),
                                ean,
                                iso_bild_pfad if iso_bild_pfad else ""
                            ])
                
            # Erfolgsmeldung
            QMessageBox.information(self, "CSV erstellt", f"Daten gespeichert in: {scan_folder}\n\n")
            
            logger.info(f"Daten gespeichert: {len(ean_codes)} EANs, {len(article_numbers)} Artikelnummern")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Speicherfehler",
                f"Fehler beim Speichern:\n{str(e)}"
            )
            logger.error(f"Fehler in save_all_data_csv: {e}")


    def go_back(self):
        """Geht zur vorherigen Seite"""
        idx = self.stack.currentIndex()
        logger.info(f"go_back: Aktuelle Seite {idx}, scan_start={self.scan_start}")
        
        # Spezialfall: Von Foto-Auswahl (Index 1) zurück zur Startseite (Index 0)
        if idx == 1:
            if QMessageBox.question(self, self.translator.get_text(self.language, "messagebox", "data_loss_confirm"), 
                                          self.translator.get_text(self.language, "messagebox", "data_loss_message"),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Cancel:
                return
            self.scan_start = False


        # Spezialfall: Von Kamera-Übersicht (Index 2) zurück zur Foto-Auswahl (Index 1)
        if idx == 2:
            # Setze scan_start zurück, damit wir neue Bilder aufnehmen können
            self.scan_start = True
            logger.info("Zurück zur Foto-Auswahl: scan_start=True gesetzt")
        
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self.update_buttons()
            self.centralWidget().updateGeometry()

    def go_next(self):
        """Geht zur nächsten Seite"""
        idx = self.stack.currentIndex()
        logger.info(f"go_next: Aktuelle Seite {idx}, scan_start={self.scan_start}")
        
        if idx >= self.stack.count() - 1:
            logger.info("Bereits auf letzter Seite")
            return
        
        # Von Startseite (Index 0) zu Foto-Auswahl (Index 1)
        elif idx == 0:
            if not self.scan_start:
                self.scan_start = True
                if not hasattr(self, "images"):
                    self.images = [None] * CONFIG.NUM_CAMERAS

                logger.info("Starte Bildaufnahme von allen Kameras")
                all_images = self.camera.take_all_pictures()
                for i, img in enumerate(all_images):
                    self.images[i] = img
                    if self.image_labels[i] is not None:
                        self.image_labels[i].setPixmap(self.convert_to_pixmap(img))

                self.stack.setCurrentIndex(idx + 1)
                self.update_buttons()
            else:
                # Falls scan_start schon True ist (z.B. nach Zurück-Navigation)
                self.stack.setCurrentIndex(idx + 1)
                self.update_buttons()
            return

        # Von Foto-Auswahl (Index 1) zu Kamera-Übersicht (Index 2)
        elif idx == 1:
            if self.scan_start:
                self.show_loading_dialog()
            else:
                QMessageBox.warning(
                    self,
                    self.translator.get_text(self.language, "messagebox", "no_images_title"),
                    self.translator.get_text(self.language, "messagebox", "no_images_message")
                )
        elif idx == 2:
            self.stack.setCurrentIndex(idx + 1)
            self.update_buttons()
            
    def show_loading_dialog(self):
        """Zeigt den Lade-Dialog mit Fortschrittsbalken"""
        self.loading_dialog = QDialog(self)
        self.loading_dialog.setWindowTitle("Ladevorgang der Daten")
        self.loading_dialog.setModal(True)
        self.loading_dialog.setFixedSize(350, 450)

        layout = QVBoxLayout(self.loading_dialog)
        
        # Lade-GIF
        movie = QMovie(os.path.join(self.Explorer_Structure, "loading.gif"))
        gif_label = QLabel()
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gif_label.setMovie(movie)
        movie.start()
        layout.addWidget(gif_label)

        # Status-Label
        status_label = QLabel("Daten werden verarbeitet...")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label.setStyleSheet("font-size: 16px; margin: 20px;")
        layout.addWidget(status_label)

        # Fortschrittsbalken
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #5d6d7e;
                border-radius: 5px;
                text-align: center;
                padding: 1px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Abbrechen-Button
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setFixedSize(120, 40)
        cancel_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
            }
        """)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Worker starten
        self.start_worker()

        def finish_loading():
            if self.loading_dialog.isVisible():
                self.loading_dialog.accept()
                self.stack.setCurrentIndex(2)  # Übersichtsseite
                self.update_buttons()
                self.scan_start = False
                QMessageBox.information(
                    self,
                    self.translator.get_text(self.language, "messagebox", "scan_completed_title"),
                    self.translator.get_text(self.language, "messagebox", "scan_completed_message")
                )
        
        def cancel_loading():
            try:
                if hasattr(self, 'worker') and self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)  # 1 Sekunde warten
            except Exception as e:
                logger.error(f"Fehler beim Abbrechen: {e}")
            
            self.loading_dialog.reject()
            self.stack.setCurrentIndex(1)
            self.update_buttons()
            QMessageBox.warning(
                self,
                self.translator.get_text(self.language, "messagebox", "scan_aborted_title"),
                self.translator.get_text(self.language, "messagebox", "scan_aborted_message")
            )
        
        # Verbindungen herstellen
        self.worker.finished.connect(finish_loading)
        cancel_btn.clicked.connect(cancel_loading)
        
        self.loading_dialog.exec()

    def update_buttons(self):
        """Aktualisiert die Sichtbarkeit der Navigationsbuttons"""
        current_index = self.stack.currentIndex()
        total_pages = self.stack.count()
        
        logger.debug(f"update_buttons: Seite {current_index}/{total_pages-1}, scan_start={self.scan_start}")

        if current_index == 0:
            self.back_btn.hide()
            self.next_btn.hide()
            if not self.camera.available_cameras:
                self.next_btn.setEnabled(False)
                self.next_btn.setToolTip("Keine Kamera verfügbar")
            else:
                self.next_btn.setEnabled(True)
                self.next_btn.setToolTip("")
        
        elif current_index == total_pages - 1:
            self.back_btn.show()
            self.next_btn.hide()
        else:
            self.back_btn.show()
            self.next_btn.show()
            
            # Auf Foto-Auswahl (Index 1): Weiter nur wenn Bilder vorhanden
            if current_index == 1:
                has_images = any(img is not None for img in self.images)
                self.next_btn.setEnabled(has_images and self.scan_start)
                if not has_images:
                    self.next_btn.setToolTip("Keine Bilder aufgenommen")
                elif not self.scan_start:
                    self.next_btn.setToolTip("Bitte Bilder aufnehmen")
                else:
                    self.next_btn.setToolTip("")
            else:
                self.next_btn.setEnabled(True)
                self.next_btn.setToolTip("")
    
    def start_worker(self):
        """Startet den Worker-Thread für parallele Verarbeitung"""
        self.worker = ParallelWorker(self.images)
        self.worker.output_received.connect(self.handle_output)
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.finished.connect(lambda: logger.info("Alle Tasks fertig"))
        self.worker.start()

    def update_progress_bar(self, value: int):
        """Aktualisiert den Fortschrittsbalken"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)

    def handle_output(self, script_name: str, data: Any):
        """Verarbeitet die Ergebnisse der Worker-Threads mit Barcode-Speicherung"""
        logger.info(f"Ergebnis von {script_name} erhalten: Typ={type(data)}")
        
        if script_name == "Abmessung":
            if isinstance(data, list):
                self.abmessungen = data
                logger.info(f"Abmessungen erhalten: {data}")
                
                # Versuche, eine aussagekräftige Abmessung zu extrahieren
                try:
                    # Finde die erste nicht-leere Abmessung
                    valid_dimensions = [d for d in data if d != "0 x 0"]
                    if valid_dimensions:
                        # Nimm die erste gültige Abmessung
                        first_dim = valid_dimensions[0]
                        parts = first_dim.split(" x ")
                        if len(parts) == 2:
                            try:
                                width = int(float(parts[0]))
                                height = int(float(parts[1]))
                                # Für einen Würfel: Länge = Breite = Höhe (vereinfacht)
                                length = width
                                self.abmessung_gesamt = f"{length} x {width} x {height}"
                                self.abmessung = self.abmessung_gesamt
                                logger.info(f"Gesamt-Abmessung berechnet: {self.abmessung}")
                            except:
                                self.abmessung = first_dim
                                logger.info(f"Abmessung übernommen: {first_dim}")
                        else:
                            self.abmessung = first_dim
                    else:
                        self.abmessung = "Undefiniert"
                        logger.warning("Keine gültigen Abmessungen gefunden")
                except Exception as e:
                    logger.error(f"Fehler beim Berechnen der Abmessung: {e}")
                    self.abmessung = "Undefiniert"
            else:
                logger.error(f"Unerwartetes Format für Abmessungen: {type(data)}")
                self.abmessung = "Undefiniert"

        elif script_name == "yolo_frames":
            if isinstance(data, list):
                self.annotierte_frames = data
                logger.info(f"Erhalte {len(data)} annotierte Frames")
                
                for i in range(min(len(data), CONFIG.NUM_CAMERAS)):
                    if data[i] is not None and self.keep[i]:
                        # BEHALTE RGB
                        self.final_images[i] = data[i]
                        logger.debug(f"Final image {i} gesetzt (Größe: {self.final_images[i].shape}, Typ: {self.final_images[i].dtype})")
                    else:
                        self.final_images[i] = None
            else:
                logger.error(f"Unerwartetes Format für yolo_frames: {type(data)}")

        elif script_name == "barcode":
            logger.info(f"Barcode-Daten empfangen: {data}")
            
            # Initialisiere all_barcodes wenn nötig
            if not hasattr(self, 'all_barcodes'):
                self.all_barcodes = []
            else:
                # Lösche alte Barcodes, bevor neue hinzugefügt werden
                self.all_barcodes.clear()
            
            # Überprüfe den Typ von data
            if isinstance(data, list):
                # Falls data bereits eine Liste von Barcode-Dicts ist
                for barcode in data:
                    if isinstance(barcode, dict) and barcode.get("found", False):
                        barcode_info = {
                            "value": barcode.get("value"),
                            "type": barcode.get("type"),
                            "image_index": barcode.get("image_index", 0),
                            "cropped_image": barcode.get("cropped_image")
                        }
                        self.all_barcodes.append(barcode_info)
                
                logger.info(f"{len(self.all_barcodes)} Barcodes gesammelt")
                
                # Debug-Ausgabe der Barcode-Daten
                for i, barcode in enumerate(self.all_barcodes):
                    logger.info(f"Barcode {i}: Wert={barcode.get('value')}, Typ={barcode.get('type')}")
                    
            elif isinstance(data, dict):
                # Falls data ein einzelnes Barcode-Dict ist (für Kompatibilität)
                if data.get("found", False):
                    barcode_info = {
                        "value": data.get("value"),
                        "type": data.get("type"),
                        "image_index": data.get("image_index", 0),
                        "cropped_image": data.get("cropped_image")
                    }
                    self.all_barcodes.append(barcode_info)
                    logger.info(f"Barcode gespeichert: {data.get('value')}")
                else:
                    logger.info("Barcode wurde nicht gefunden (found=False)")
            else:
                logger.error(f"Unerwartetes Format für barcode: {type(data)} - {data}")
                
        elif script_name == "weight":
            self.gewicht = data
            logger.info(f"Gewicht: {data}")
        
        # Prüfe ob alle Daten vorhanden sind und aktualisiere GUI
        self._check_and_update_gui()

    def _check_and_update_gui(self):
        """Prüft ob alle Daten vorhanden sind und aktualisiert die GUI"""
        # Stelle sicher, dass all_barcodes existiert
        if not hasattr(self, 'all_barcodes'):
            self.all_barcodes = []
        
        # Stelle sicher, dass abmessung und gewicht existieren
        if not hasattr(self, 'abmessung') or self.abmessung is None:
            self.abmessung = "Undefiniert"
        if not hasattr(self, 'gewicht') or self.gewicht is None:
            self.gewicht = "Undefiniert"
        
        # Nach der Barcode-Erkennung Zugeschnittene Bilder erstellen
        if self.all_barcodes and hasattr(self, 'images'):
            for barcode in self.all_barcodes:
                if barcode.get("cropped_image") is None:
                    img_idx = barcode.get("image_index", 0)
                    if img_idx < len(self.images) and self.images[img_idx] is not None:
                        img = self.images[img_idx]
                        if img is not None:
                            # Erstelle einen Ausschnitt um den Barcode herum
                            h, w = img.shape[:2]
                            crop_h = min(300, h)
                            crop_w = min(500, w)
                            x = max(0, w // 2 - crop_w // 2)
                            y = max(0, h // 2 - crop_h // 2)
                            
                            # BGR zu RGB konvertieren für Qt
                            roi = img[y:y+crop_h, x:x+crop_w]
                            if len(roi.shape) == 3 and roi.shape[2] == 3:
                                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            else:
                                roi_rgb = roi
                            
                            barcode["cropped_image"] = roi_rgb
        
        # Prüfe ob alle notwendigen Daten vorhanden sind
        has_abmessung = self.abmessung not in [None, "Undefiniert"]
        has_gewicht = self.gewicht not in [None, "Undefiniert"]
        has_barcodes = len(self.all_barcodes) > 0
        has_yolo_frames = hasattr(self, "annotierte_frames") and self.annotierte_frames
        
        # GUI aktualisieren wenn:
        # 1. Alle Daten vorhanden sind (Abmessung, Gewicht), ODER
        # 2. YOLO-Frames vorhanden sind (für Bildanzeige), ODER
        # 3. Barcodes erkannt wurden
        update_needed = False
        
        if has_abmessung and has_gewicht:
            logger.info("Alle Hauptdaten vorhanden - aktualisiere GUI")
            update_needed = True
        elif has_yolo_frames:
            logger.info("YOLO-Frames vorhanden - aktualisiere Bilder in GUI")
            update_needed = True
        elif has_barcodes:
            logger.info("Barcodes vorhanden - aktualisiere GUI")
            update_needed = True
        
        if update_needed:
            # Stelle sicher, dass alle final_images gesetzt sind
            for i in range(CONFIG.NUM_CAMERAS):
                if self.final_images[i] is None and i < len(self.images):
                    self.final_images[i] = self.images[i]
            
            # Lade die Seiten neu
            self.load_pages()
            QApplication.processEvents()  # Erzwinge GUI-Update

    # Platzhalter-Funktionen für die Systemprüfung
    def check_camera(self):
        QMessageBox.information(self, "Kamera-Prüfung", "Kamera-System wird geprüft...")
        logger.info("Kamera-Prüfung gestartet")

    def check_light(self):
        QMessageBox.information(self, "Beleuchtungs-Prüfung", "Beleuchtung wird geprüft...")
        logger.info("Beleuchtungs-Prüfung gestartet")

    def check_measure(self):
        QMessageBox.information(self, "Mess-System-Prüfung", "Mess-System wird geprüft...")
        logger.info("Mess-System-Prüfung gestartet")

    def calibrate_scale(self):
        QMessageBox.information(self, "Waagen-Kalibrierung", "Waage wird kalibriert...")
        logger.info("Waagen-Kalibrierung gestartet")

    def check_storage(self):
        QMessageBox.information(self, "Speicher-Prüfung", "Speicher wird geprüft...")
        logger.info("Speicher-Prüfung gestartet")

    def keyPressEvent(self, event):
        """Behandelt Tastatureingaben"""
        if event.key() == Qt.Key.Key_Left:
            self.go_back()
        elif event.key() == Qt.Key.Key_Right:
            self.go_next()
        elif event.key() == Qt.Key.Key_Escape:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    logger.info("3D-Scanner wird gestartet...")
    app = QApplication(sys.argv)
    w = FullscreenApp()
    w.show()
    sys.exit(app.exec())


