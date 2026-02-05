#Interface_v08.py
"""=======TODO-Liste v0.8=======
Objekt-Detection muss verbessert werden/Mit Bemassung der Distanz von der LIDAR-Kamera      !!!!!!!!!
SAP-Integration                 (Platzhalter-Button/optional)
Lokal speichern Integration     (Formatierung?)

================================"""

import os # Für Dateipfade und Betriebssysteminteraktionen
import csv # Für CSV-Verarbeitung (SAP-Integration)
import sys
import time # Für Verzögerungen
import cv2 # Für Kamerazugriff und Bildverarbeitung
import json
import serial # Für Beleuchtung
import shutil  # Für Speicherplatzprüfung
import logging  # Für Logging
import platform # Für Betriebssystemerkennung
import numpy as np
import depthai as dai
from datetime import datetime 
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, QProgressBar, QComboBox, QInputDialog
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer


# OpenCV-Warnungen reduzieren
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'V4L2'

# In setup_logging fügen Sie hinzu:
logging.getLogger('cv2').setLevel(logging.ERROR)


# ==================== Konfiguration ====================
@dataclass
class AppConfig:
    """Konfigurationsklasse für die Anwendung"""
    NUM_CAMERAS: int = 4
    IMAGE_WIDTH: int = 640
    IMAGE_HEIGHT: int = 480
    # Bei USB-Verbindung meist /dev/ttyACM0 oder /dev/ttyUSB0
    USB0: str = "/dev/ttyUSB0" 
    BAURATE: int = 9600
    
    DEBUG_SINGLE_CAMERA: bool = False
    DEFAULT_LANGUAGE: str = "de"
    GUI_RESOURCES_PATH: str = "GUI_Anzeige"
    LOG_LEVEL: str = "INFO"
    #YOLO_MODEL_PATH: str = "models/YOLOV8s_Barcode_Detection.pt"
    #SCANS_FOLDER: str = "C:\\Users\\username\\Desktop\\Scans" # Windows
    SCANS_FOLDER: str = "/home/leitner/Desktop/Scans"  # Linux

    
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
                "quit_btn": ("Programm beenden", "Quit Program", "Esci dal Programma"),
                "check_camera": ("Kamera prüfen", "Check Camera", "Controlla Fotocamera"),
                "check_light": ("Beleuchtung prüfen", "Check Lighting", "Controlla Illuminazione"),
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
                
                "scale_calibration_title": ("Waagen-Kalibrierung", "Scale Calibration", "Calibrazione Bilancia"),
                "scale_calibration_input": ("Referenzgewicht eingeben (kg):", "Enter reference weight (kg):", "Inserisci peso di riferimento (kg):"),
                "scale_calibration_default": ("1.000", "1.000", "1.000"),
                "scale_calibration_starting": ("Kalibrierung wird gestartet...", "Starting calibration...", "Avvio calibrazione..."),
                "scale_calibration_live": ("Rohdaten-Live-Anzeige", "Live Raw Data Display", "Visualizzazione Dati in Tempo Reale"),
                "scale_calibration_raw": ("Rohwert:", "Raw Value:", "Valore Grezzo:"),
                "scale_calibration_progress": ("Kalibrierungsfortschritt:", "Calibration Progress:", "Progresso Calibrazione:"),
                "scale_calibration_complete": ("Kalibrierung erfolgreich!", "Calibration successful!", "Calibrazione completata!"),
                "scale_calibration_error": ("Kalibrierungsfehler:", "Calibration error:", "Errore di calibrazione:"),

                "storage_info_title": ("Speicherplatz-Information", "Storage Information", "Informazioni Spazio"),
                "storage_total": ("Gesamter Speicher:", "Total Storage:", "Spazio Totale:"),
                "storage_used": ("Belegt:", "Used:", "Utilizzato:"),
                "storage_free": ("Frei:", "Free:", "Libero:"),
                "storage_config_folder": ("📁 Konfigurierter Scans-Ordner:", "📁 Configured Scans Folder:", "📁 Cartella Scans Configurata:"),
                "storage_error_title": ("Fehler", "Error", "Errore"),
                "storage_error_message": ("Speicherprüfung fehlgeschlagen:", "Storage check failed:", "Verifica spazio fallita:")
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
        self.oak_available = False
        self.available_cameras = self._find_cameras()
        
        # Serielle Verbindung für Blitz-Steuerung initialisieren
        self.serial_port = None
        logger.info(f"Verfügbare Kameras gefunden: {self.available_cameras}, OAK-D2: {self.oak_available}")
    
    def _send_command(self, command: str):
        """Sendet einen Befehl zur seriellen Schnittstelle"""
        try:
            if self.serial_port is None:
                self.serial_port = serial.Serial(CONFIG.USB0, CONFIG.BAURATE, timeout=1)
                time.sleep(2)  # Warten bis Serial Port bereit
            
            full_command = command + "\n"
            self.serial_port.write(full_command.encode('utf-8'))
            time.sleep(0.1)  # Kurze Pause zur Verarbeitung
            logger.debug(f"Serieller Befehl gesendet: {command}")
            
        except Exception as e:
            logger.warning(f"Fehler bei serieller Kommunikation: {e}")
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
                self.serial_port = None
    
    def _control_light(self, state: bool):
        """Steuert die Beleuchtung für OAK-D2 Aufnahmen"""
        try:
            if state:
                logger.info("Licht für Aufnahme einschalten...")
                self._send_command("Change")
                time.sleep(0.1)
                self._send_command("a")  # Alles an
                time.sleep(0.5)  # Kurze Pause für Licht-Stabilisierung
            else:
                logger.info("Licht nach Aufnahme ausschalten...")
                self._send_command("Change")
                time.sleep(0.1)
                self._send_command("0")  # Alles aus
            
        except Exception as e:
            logger.warning(f"Fehler bei Licht-Steuerung: {e}")

    def _get_camera_backend(self) -> int:
        """Bestimmt den passenden Camera Backend für das Betriebssystem"""
        if platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_V4L2  # Besser für Linux

    def _find_cameras(self) -> List[int]:
        """Findet verfügbare Kameras"""
        available: List[int] = []
        backend = self._get_camera_backend()
        
        for i in range(CONFIG.NUM_CAMERAS-1):  # Letzte Kamera ist OAK-D2
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                logger.warning(f"Fehler beim Zugriff auf Kamera {i}: {e}")
        
        # OAK-D2 prüfen
        self._check_oak_availability()
        
        return available
    
    def _check_oak_availability(self):
        """Prüft ob OAK-D2 verfügbar ist"""
        try:
            devices = dai.Device.getAllAvailableDevices()
            if len(devices) > 0:
                self.oak_available = True
                logger.info(f"OAK-D2 ist verfügbar: {len(devices)} Gerät(e)")
            else:
                logger.warning("Keine OAK-D2 Kamera gefunden")
        except Exception as e:
            logger.error(f"Fehler bei OAK-D2 Prüfung: {e}")
            self.oak_available = False

    def _take_oak_picture(self) -> Optional[np.ndarray]:
        """Nimmt ein Bild mit der OAK-D2 RGB-Kamera auf - MIT LICHT"""
        try:
            # Licht einschalten vor der Aufnahme
            self._control_light(True)
            time.sleep(0.3)  # Kurze Pause für Licht-Stabilisierung
        
            pipeline = dai.Pipeline()
        
            # RGB-Kamera der OAK-D2
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setPreviewSize(CONFIG.IMAGE_WIDTH, CONFIG.IMAGE_HEIGHT)
        
            # ENTFERNT: setExposureTime und setIspScale - die sind nicht in der API
            # Die Standardeinstellungen sollten mit Blitz funktionieren
        
            # Ausgabestream
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)
        
            # Verbindung herstellen und Bild aufnehmen
            with dai.Device(pipeline) as device:
                q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                logger.info("OAK-D2 Bild mit Licht erfolgreich aufgenommen")
            
                # Licht ausschalten NACH der Aufnahme
                self._control_light(False)
                return frame
            
        except Exception as e:
            logger.error(f"Fehler bei OAK-D2 Bildaufnahme: {e}")
            # Sicherheitshalber Licht ausschalten auch bei Fehler
            try:
                self._control_light(False)
            except:
                pass
            return None
        
    
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
        
        # Spezialfall: OAK-D2 (letzte Kamera) - hat eigene Lichtsteuerung
        if camera_id == CONFIG.NUM_CAMERAS - 1:  # Letzte Kamera ist OAK-D2
            if self.oak_available:
                oak_img = self._take_oak_picture()  # Licht wird hier gesteuert
                if oak_img is not None:
                    return oak_img
                else:
                    logger.warning(f"OAK-D2 Bildaufnahme fehlgeschlagen")
                    return self._make_placeholder(camera_id)
            else:
                logger.warning(f"OAK-D2 nicht verfügbar")
                return self._make_placeholder(camera_id)
        
        # Normale USB-Kameras - mit Licht
        if camera_id not in self.available_cameras:
            logger.warning(f"Kamera {camera_id} nicht verfügbar")
            return self._make_placeholder(camera_id)
        
        # Licht für USB-Kamera einschalten
        self._control_light(True)
        time.sleep(0.3)  # Kurze Pause für Licht-Stabilisierung
        
        backend = self._get_camera_backend()
        
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if not cap.isOpened():
                logger.error(f"Kamera {camera_id} konnte nicht geöffnet werden")
                cap.release()
                self._control_light(False)  # Licht ausschalten bei Fehler
                return self._make_placeholder(camera_id)
            
            # Kurze Verzögerung für Kamera-Initialisierung
            time.sleep(0.2)
            
            # Bild mit erhöhter Belichtung für Blitz
            cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)  # Kurze Belichtung
            ret, frame = cap.read()
            cap.release()
            
            # Licht nach Aufnahme ausschalten
            self._control_light(False)
            
            if ret:
                logger.info(f"Bild erfolgreich von Kamera {camera_id} aufgenommen")
                return frame
            else:
                logger.error(f"Bildaufnahme von Kamera {camera_id} fehlgeschlagen")
                return self._make_placeholder(camera_id)
                
        except Exception as e:
            logger.error(f"Fehler bei Bildaufnahme von Kamera {camera_id}: {e}")
            # Sicherheitshalber Licht ausschalten
            try:
                self._control_light(False)
            except:
                pass
            return self._make_placeholder(camera_id)

    def take_all_pictures(self) -> List[np.ndarray]:
        """Nimmt Bilder von allen Kameras auf"""
        images: List[np.ndarray] = []
        
        # Für alle Kameras wird Licht automatisch in take_picture() gesteuert
        if self.debug_single_camera:
            # Debug: Eine Kamera für alle Bilder
            logger.debug("Debug-Modus: Verwende eine Kamera für alle Bilder")
            for i in range(CONFIG.NUM_CAMERAS):
                if i < len(self.available_cameras):
                    img = self.take_picture(0)  # Erste Kamera für alles
                else:
                    img = self._make_placeholder(i)
                images.append(img)
        else:
            # Normal: Jede Kamera macht ein Bild
            for i in range(CONFIG.NUM_CAMERAS):
                img = self.take_picture(i)  # Licht wird für jede Kamera separat gesteuert
                images.append(img)
        
        return images
    
    def close(self):
        """Schließt alle Ressourcen"""
        if self.serial_port:
            try:
                # Sicherstellen dass Licht aus ist
                self._send_command("Change")
                self._send_command("0")
                self.serial_port.close()
                logger.info("Serielle Verbindung geschlossen")
            except Exception as e:
                logger.warning(f"Fehler beim Schließen der seriellen Verbindung: {e}")
                
                

# ==================== Detection Manager ====================
class DetectionManager:
    def __init__(self):
        self.barcode_detector = None  # Wird später initialisiert
        self.all_barcodes: List[Dict[str, Any]] = []

    def run_barcode_detection(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Erkennt alle Barcodes in den Bildern"""
        try:
            # Importiere die BarcodeDetector Klasse
            from workers.BarCode_v03 import BarcodeDetector
            
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

class ParallelWorker(QThread):   
    output_received = pyqtSignal(str, object)  # (task_name, result)
    progress_updated = pyqtSignal(int)  # Fortschritt in %
    finished = pyqtSignal()

    def __init__(self, images: List[np.ndarray], keep: List[bool]):
        super().__init__()
        self.images = images
        self.keep = keep
        self.detection_manager = DetectionManager()
        self.progress = 0

        # Filtere nur die nicht verworfenen Bilder
        self.images_to_process = [img for i, img in enumerate(images) if keep[i]]
        self.original_indices = [i for i in range(len(images)) if keep[i]]

    def _update_progress(self, increment: int):
        """Aktualisiert den Fortschritt"""
        self.progress += increment
        self.progress_updated.emit(self.progress)



    def run(self):  # Einfügen
        """Führt parallele Verarbeitung mit ThreadPoolExecutor durch"""
        valid_images = [keep for keep in self.keep if keep is not False]
    
        if not valid_images:
            logger.warning("Keine gültigen Bilder zum Verarbeiten - alle sind Verworfen")
            self.finished.emit()
            return
        
        import concurrent.futures
        
        # 3 Workers 
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Entfernt: YOLO-Task
            futures = {
                executor.submit(self._run_barcode_task): "barcode",
                executor.submit(self._run_weight_task): "weight",
                executor.submit(self._run_volume_task): "volume"
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
    
    def _run_volume_task(self):
        """Führt Volumenmessung mit OAK-D2 durch"""
        try:
            import workers.Tiefenkamera_Messung_03 as volume_module
            volume_result = volume_module.get_volume()
            
            # Formatieren für Anzeige
            if volume_result.get("success"):
                dimensions = f"{volume_result['length']:.1f} x {volume_result['width']:.1f} x {volume_result['height']:.1f}"
                volume_cm3 = volume_result['volume'] / 1000  # mm³ zu cm³
                
                return {
                    "volume": volume_cm3,
                    "abmessung": dimensions,
                    "success": True,
                    "depth_frame": volume_result.get("depth_frame")
                }
            else:
                return {
                    "volume": 0.0,
                    "abmessung": "0 x 0 x 0",
                    "success": False,
                    "error": volume_result.get("error")
                }
                
        except ImportError as e:
            logger.error(f"Volumenmodul nicht verfügbar: {e}")
            return {
                "volume": 0.0,
                "abmessung": "0 x 0 x 0",
                "success": False,
                "error": "Modul nicht gefunden"
            }
        except Exception as e:
            logger.error(f"Volumen Task Fehler: {e}")
            return {
                "volume": 0.0,
                "abmessung": "0 x 0 x 0",
                "success": False,
                "error": str(e)
            }
   
    def _run_barcode_task(self):
        """Führt Barcode-Erkennung durch - NUR auf nicht verworfenen Bildern"""
        try:
            # Verwende images_to_process anstatt images
            barcodes = self.detection_manager.run_barcode_detection(self.images_to_process)
            
            # Passe die image_indices an die originalen Indizes an
            for barcode in barcodes:
                original_idx = barcode.get('image_index', 0)
                if original_idx < len(self.original_indices):
                    barcode['image_index'] = self.original_indices[original_idx]
                    
            return {"barcodes": barcodes}
        except Exception as e:
            logger.error(f"Barcode Task Fehler: {e}")
            return {"barcodes": []}
    
    def _run_weight_task(self):
        """Führt Gewichtsmessung durch"""
        try:
            import workers.Gewichts_Messung
            weight = workers.Gewichts_Messung.get_weight()
            if weight is not None:
                weight = round(weight, 3)
            return {"weight": weight}

        except ImportError as e:
            logger.error(f"Gewichtsmodul nicht verfügbar: {e}")
            return {"weight": "Undefiniert"}
        except Exception as e:
            logger.error(f"Gewicht Task Fehler: {e}")
            return {"weight": "Undefiniert"}
    
    def _process_result(self, task_type: str, result: dict):
        """Verarbeitet Ergebnisse der Tasks"""
        if task_type == "barcode":
            barcodes = result.get("barcodes", [])
            for barcode in barcodes:
                self.output_received.emit("barcode", barcode)
        elif task_type == "weight":
            weight = result.get("weight", "Undefiniert")
            self.output_received.emit("weight", weight)
        elif task_type == "volume":
            self.output_received.emit("volume", result)





# ==================== Main Application ====================
class FullscreenApp(QMainWindow):
    """Hauptanwendung für den 3D-Scanner"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D-Scanner")
        self.showFullScreen()

        # Initialisierung
        self.camera = CameraManager(debug_single_camera=False)
        self.translator = TranslationManager()
        self.language = CONFIG.DEFAULT_LANGUAGE
        self.Explorer_Structure = CONFIG.GUI_RESOURCES_PATH
        # Bei USB-Verbindung
        self.port = CONFIG.USB0
        self.baudrate = CONFIG.BAURATE
        
        import workers.Gewichts_Messung
        workers.Gewichts_Messung.init_adc()
        workers.Gewichts_Messung.tara()

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
        logo_path = os.path.join(self.Explorer_Structure, "Logo.png")
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
            (self.translator.get_text(self.language, "start", "calibrate_scale"), self.calibrate_scale),
            (self.translator.get_text(self.language, "start", "check_storage"), self.check_storage)
        ]

        for name, callback in status_buttons:
            status_button = self.create_status_button(name, callback)
            right_layout.addWidget(status_button)

        right_layout.addStretch()

        # Quit Button
        quit_btn = QPushButton(self.translator.get_text(self.language, "start", "quit_btn"))
        quit_btn.setStyleSheet("""
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
        quit_btn.setFixedHeight(45)
        quit_btn.clicked.connect(self.close_application)
        right_layout.addWidget(quit_btn)

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

    def close_application(self):
        """Schließt die Anwendung mit Bestätigungsdialog"""
        logger.info("Programm wird beendet")
        
        # Worker-Ressourcen sauber freigeben        
        if hasattr(self, 'worker') and hasattr(self.worker, 'isRunning'):
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)
        
        # Anwendung schließen
        QApplication.quit()


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
            self.keep[idx] = True

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

                    f"{self.translator.get_text(self.language, 'overview', 'dimensions')} {self.abmessung} {self.translator.get_text(self.language, 'overview', 'mm')}",
                    
                    f"{self.translator.get_text(self.language, 'overview', 'weight')} {self.gewicht}{self.translator.get_text(self.language, 'overview', 'kg')}",

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
        

    def clear_image_memory(self):
        """Löscht explizit alle Bildreferenzen und gibt RAM frei"""
        for i in range(CONFIG.NUM_CAMERAS):
            # Setze die Referenzen auf None, ohne die Liste zu verändern
            self.images[i] = None
            self.final_images[i] = None
        
        # Barcode-Bilder löschen
        for barcode in self.all_barcodes:
            if "cropped_image" in barcode and barcode["cropped_image"] is not None:
                barcode["cropped_image"] = None
        
        # Garbage Collector manuell aufrufen
        import gc
        gc.collect()
        
        logger.debug("Bildspeicher explizit freigegeben")


    def rebeginn_application(self):
        """Startet die Anwendung von der Startseite neu"""
        if QMessageBox.question(self, self.translator.get_text(self.language, "messagebox", "data_loss_confirm"), 
                                          self.translator.get_text(self.language, "messagebox", "data_loss_message"),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Cancel:
            return
        
        self.clear_image_memory()


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
        """Speichert Daten in Tages-CSV (append) und Bilder in Foto-Ordner"""
        try:
            # 1. Basisordner aus Konfiguration verwenden
            base_scans_folder = CONFIG.SCANS_FOLDER
            if not base_scans_folder:  # Fallback falls leer
                base_scans_folder = "Scans"
            
            if not os.path.exists(base_scans_folder):
                os.makedirs(base_scans_folder)
                logger.info(f"Basisordner erstellt: {base_scans_folder}")
            
            # 2. Heutiges Datum als Ordner
            date_str = datetime.now().strftime('%Y-%m-%d')
            date_folder = os.path.join(base_scans_folder, date_str)
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)
            
            # 3. "Fotos"-Ordner im Tagesordner erstellen
            fotos_folder = os.path.join(date_folder, "Fotos")
            if not os.path.exists(fotos_folder):
                os.makedirs(fotos_folder)
            
            # 4. Scan-spezifischen Unterordner für Bilder
            scan_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            scan_bilder_folder = os.path.join(fotos_folder, f"Scan_{scan_timestamp}")
            os.makedirs(scan_bilder_folder, exist_ok=True)
            
            # 5. CSV-Datei im Tagesordner (gleiche Datei für alle Scans des Tages)
            csv_datei = os.path.join(date_folder, f"scans_{date_str}.csv")
            
            # 6. Prüfen ob CSV bereits existiert (dann append Mode)
            file_exists = os.path.exists(csv_datei)
            
            with open(csv_datei, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                
                # 7. Nur Kopfzeile schreiben wenn Datei neu ist
                if not file_exists:
                    writer.writerow([
                        "Scan_ID",
                        "Materialnummer",
                        "Gewicht_kg",
                        "Laenge_mm",
                        "Breite_mm",
                        "Hoehe_mm",
                        "EAN",
                        "ISO_Bild_Pfad",
                    ])
                    logger.info(f"Neue CSV-Datei erstellt: {csv_datei}")
                
                # 8. Alle Bilder speichern und Pfade merken
                image_names = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]
                bild_pfade = {}
                
                for idx, img in enumerate(self.images):
                    if img is not None and idx < len(image_names):
                        img_name = image_names[idx]
                        bild_datei = f"{img_name}.jpg"
                        bild_pfad = os.path.join(scan_bilder_folder, bild_datei)
                        
                        # Bild speichern
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        else:
                            bgr_img = img
                        
                        cv2.imwrite(bild_pfad, bgr_img)
                        bild_pfade[img_name] = bild_pfad
                        logger.info(f"{img_name} gespeichert: {bild_pfad}")
                
                # 9. Relative Pfade für CSV (relativ zur CSV-Datei)
                def get_relative_path(full_path):
                    if not full_path:
                        return ""
                    # Berechne relativen Pfad von CSV zu Bild
                    return os.path.relpath(full_path, date_folder)
                
                # 10. Abmessungen von OAK-D2
                laenge = breite = hoehe = 0
                if hasattr(self, 'abmessung') and self.abmessung:
                    try:
                        # Format: "Länge x Breite x Höhe" (z.B. "250 x 150 x 100")
                        teile = self.abmessung.split(" x ")
                        if len(teile) >= 3:
                            laenge = float(teile[0])
                            breite = float(teile[1])
                            hoehe = float(teile[2])
                        else:
                            logger.warning(f"Ungültiges Format für abmessung: {self.abmessung}")
                    except Exception as e:
                        logger.error(f"Fehler beim Parsen der 3D-Abmessungen: {e}")
        
                # 11. Gewicht parsen (bleibt gleich)
                gewicht = 0
                if self.gewicht and self.gewicht != "Undefiniert":
                    try:
                        gewicht_str = str(self.gewicht).replace("kg", "").strip()
                        gewicht = float(gewicht_str)
                    except:
                        pass
				# 11. Gewicht parsen
                gewicht = 0
                if self.gewicht and self.gewicht != "Undefiniert":
                    try:
                        gewicht_str = str(self.gewicht).replace("kg", "").strip()
                        gewicht = float(gewicht_str)
                    except:
                        pass
                
                # 12. Barcodes trennen
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
       
                # 13. CSV-Zeilen schreiben - Für SAP
                if not ean_codes and not article_numbers:
                    writer.writerow([
                        scan_timestamp,  # Scan_ID
                        "",  # Interne Materialnummer
                        f"{gewicht:.3f}".replace(".", ","),  # Gewicht in kg
                        f"{laenge:.0f}",  # Länge in mm
                        f"{breite:.0f}",  # Breite in mm
                        f"{hoehe:.0f}",  # Höhe in mm
                        "",  # EAN
                        get_relative_path(bild_pfade.get("iso_Bild", "")) # ISO Bild
                    ])
                    logger.info(f"Leerer Scan zu CSV hinzugefügt: {scan_timestamp}")
                else:
                    for article in (article_numbers if article_numbers else [""]):
                        for ean in (ean_codes if ean_codes else [""]):                            
                            writer.writerow([
                                scan_timestamp,  # Scan_ID
                                article,  # Interne Materialnummer
                                f"{gewicht:.3f}".replace(".", ","),  # Gewicht in kg
                                f"{laenge:.0f}",  # Länge in mm
                                f"{breite:.0f}",  # Breite in mm
                                f"{hoehe:.0f}",  # Höhe in mm
                                ean,  # EAN
                                get_relative_path(bild_pfade.get("iso_Bild", ""))
                           ])
                    logger.info(f"{len(article_numbers)}x{len(ean_codes)} Datensätze zu CSV hinzugefügt")
        
                # Erfolgsmeldung aktualisieren
                lines_added = max(1, len(article_numbers) * max(1, len(ean_codes)))
        
                success_msg = f"""Scan erfolgreich gespeichert!
Statistik:
• {lines_added} neue Zeile(n) in CSV
• {len(ean_codes)} EAN-Code(s)
• {len(article_numbers)} Artikelnummer(n)
• {len(bild_pfade)} Bild(er) gespeichert
• Abmessungen: {laenge} x {breite} x {hoehe} mm

CSV-Status: {os.path.getsize(csv_datei):,} Bytes
({'Datei neu erstellt' if not file_exists else 'An bestehende Datei angehängt'})
"""

            
            QMessageBox.information(self, "Scan gespeichert", success_msg)
            logger.info(f"Scan {scan_timestamp} zu {csv_datei} hinzugefügt")
            
            # 15. Optional: CSV-Datei öffnen (nur bei erstem Scan des Tages)
            try:
                if not file_exists and platform.system() == "Windows":
                    os.startfile(csv_datei)  # CSV im Excel öffnen
            except:
                pass
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Speicherfehler",
                f"Fehler beim Speichern:\n{str(e)}"
            )
            logger.error(f"Fehler in save_all_data_csv: {e}", exc_info=True)


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
            self.keep = [True] * CONFIG.NUM_CAMERAS



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
        self.worker = ParallelWorker(self.images, self.keep)
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

        if script_name == "barcode":
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
            if data < 0.0:
                self.gewicht = 0.0
            else:
                self.gewicht = data
            
            logger.info(f"Gewicht: {data}")
        
        elif script_name == "volume":
            if isinstance(data, dict):
                logger.info(f"Volumendaten erhalten: {data}")
                
                # Speichere die 3D-Abmessungen
                if data.get("success"):
                    print("Hallaidisakjdhjkasgdsflkhadkjfkj")

                    print(data.get("abmessung", "0 x 0 x 0"))
                    self.abmessung = data.get("abmessung", "0 x 0 x 0")
                    
                    # Optional: Tiefenbild anzeigen/speichern
                    #depth_frame = data.get("depth_frame")
                    #if depth_frame is not None:
                        #self.depth_image = depth_frame
                        
                    logger.info(f"3D-Abmessungen: {self.abmessung}")
                else:
                    logger.warning(f"Volumenmessung fehlgeschlagen: {data.get('error')}")
            else:
                logger.error(f"Unerwartetes Format für Volumenmessung: {type(data)}")

        # Prüfe ob alle Daten vorhanden sind und aktualisiere GUI
        self._check_and_update_gui()


    def _check_and_update_gui(self):
        """Prüft ob alle Daten vorhanden sind und aktualisiert die GUI"""
        # Stelle sicher, dass all_barcodes, abmessung und gewicht existiert
        if not hasattr(self, 'all_barcodes'):
            self.all_barcodes = []
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
        
        # GUI aktualisieren wenn:
        # 1. Alle Daten vorhanden sind (Abmessung, Gewicht), ODER
        # 3. Barcodes erkannt wurden
        update_needed = False
        
        if has_abmessung and has_gewicht:
            logger.info("Alle Hauptdaten vorhanden - aktualisiere GUI")
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


    def check_camera(self):
        """Verbesserte Kamera-Prüfung für Linux"""
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Kamera-Prüfung")
        dialog.setFixedSize(400, 200)
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        
        progress = QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(progress)
        
        status = QLabel("Teste Kameras...")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status)
        
        # TEST: Direkt DepthAI prüfen
        try:
            devices = dai.Device.getAllAvailableDevices()
            logger.info(f"DepthAI Geräte gefunden: {devices}")
        
            for device_info in devices:
                logger.info(f"Device: {device_info}")
                try:
                    with dai.Device(device_info) as device:
                        logger.info(f"Device {device_info.name} erfolgreich geöffnet")
                        logger.info(f"  Kameras: {device.getConnectedCameras()}")
                        logger.info(f"  USB Geschwindigkeit: {device.getUsbSpeed()}")
                except Exception as e:
                   logger.error(f"Fehler beim öffnen von {device_info}: {e}")
        except Exception as e:
           logger.error(f"DepthAI Test fehlgeschlagen: {e}")
        
        def check_and_close():
            # Test für USB-Kameras
            usb_count = 0
            for i in range(CONFIG.NUM_CAMERAS-1): # Letzte Kamera ist OAK-D2
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cap.release()
                        usb_count += 1
                except:
                    pass
            
            # OAK-D Erkennung mit DepthAI
            oak = False
            try:
                verfügbar = len(dai.Device.getAllAvailableDevices()) > 0
                if verfügbar:
                    oak = True
                    print("OAK-D2 gefunden!")
                else:
                    print("Keine OAK-D2 Kamera gefunden!")
            except Exception as e:
                print(f"OAK-D Check Fehler: {e}")
                oak = False

            dialog.close()
            
            result = f"USB-Kameras: {usb_count}/3\nOAK-D2: {'1/1' if oak else '0/1'}\n\n"
            
            if usb_count >= 3 and oak:
                result += "Alle Kameras OK"
                icon = QMessageBox.Icon.Information
            elif usb_count >= 3:
                result += "OAK-D2 fehlt"
                icon = QMessageBox.Icon.Warning
            else:
                result += "Nicht genügend Kameras"
                icon = QMessageBox.Icon.Critical
            
            QMessageBox(dialog).information(self, "Ergebnis", result)
        
        QTimer.singleShot(1000, check_and_close)
        dialog.exec()
            

    def check_light(self):
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Warten, bis der Serial Port nach dem öffnen bereit ist

            def send_command(command):
                full_command = command + "\n"
                ser.write(full_command.encode('utf-8'))
                time.sleep(0.1)  # Kurze Pause zur Verarbeitung

            # Schritt 1: In den Change-Modus wechseln
            send_command("Change")

            ''' Mögliche Commands:
        
            case '1': Blitz(1); break;
            case '2': Blitz(2); break;
            case '3': Blitz(3); break;
            case '4': Strip_On(); break;
            case 'a': All_ON(); break;
            case '0': All_OFF(); break;
            '''
        
            send_command("a")
            time.sleep(1)  # Kurze Pause zur Verarbeitung

            send_command("Change")
            send_command("0")

            ser.close()
        except Exception as e:
            logger.Waning(f"Fehler: {e}")
        

    def calibrate_scale(self):
        """ Führt eine Referenzkalibrierung für 3 Wägezellen durch. """
        from workers.Gewichts_Messung import calibrate_cell

        try:
            # 1) Referenzgewicht abfragen
            reference_weight, ok = QInputDialog.getDouble(
                self,
                "Referenzkalibrierung",
                "Referenzgewicht in kg eingeben:",
                decimals=3,
                min=0.1
            )

            if not ok:
                return

            # 2) Dialog für Kalibrierung
            dialog = QDialog(self)
            dialog.setWindowTitle("Referenzkalibrierung - Rohdaten")
            dialog.setModal(True)

            layout = QVBoxLayout(dialog)

            info_label = QLabel(
                "Lege das Referenzgewicht auf die jeweilige Wägezelle\n"
                "und drücke den passenden Button."
            )
            layout.addWidget(info_label)

            # Rohdatenanzeige
            self.raw_value_label = QLabel("Faktor: ---")
            self.raw_value_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            layout.addWidget(self.raw_value_label)

            # 3) Buttons für jede Zelle
            btn_cell_1 = QPushButton("Zelle 1 kalibrieren")
            btn_cell_2 = QPushButton("Zelle 2 kalibrieren")
            btn_cell_3 = QPushButton("Zelle 3 kalibrieren")

            layout.addWidget(btn_cell_1)
            layout.addWidget(btn_cell_2)
            layout.addWidget(btn_cell_3)

            # Abbrechen
            btn_close = QPushButton("Schließen")
            layout.addWidget(btn_close)

        

            # 4) Button-Logik
            faktor = btn_cell_1.clicked.connect(lambda: calibrate_cell(0, reference_weight))
            btn_cell_2.clicked.connect(lambda: calibrate_cell(1, reference_weight))
            btn_cell_3.clicked.connect(lambda: calibrate_cell(2, reference_weight))

            btn_close.clicked.connect(dialog.close)

            dialog.exec()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Kalibrierungsfehler",
                f"Fehler bei der Referenzkalibrierung:\n{str(e)}"
            )


    def check_storage(self):
        """Prüft und zeigt den verfügbaren Speicherplatz - kompakte Version"""
        try:
            # Laufwerk bestimmen
            storage_path = "C:\\" if platform.system() == "Windows" else "/"
            
            # Speicherinformationen
            total, used, free = shutil.disk_usage(storage_path)
            percent_used = (used / total) * 100
            
            # Übersetzungen abrufen
            texts = self.translator.translations["messagebox"]
            lang_idx = self.translator.language_map.get(self.language, 0)
            
            # Nachricht zusammenbauen
            message = f"""
    {texts['storage_total'][lang_idx]}  {total/1024**3:.2f} GB
    {texts['storage_used'][lang_idx]}   {used/1024**3:.2f} GB ({percent_used:.1f}%)
    {texts['storage_free'][lang_idx]}   {free/1024**3:.2f} GB

    {texts['storage_config_folder'][lang_idx]} {CONFIG.SCANS_FOLDER}"""
            
            # Anzeigen
            QMessageBox.information(self, texts['storage_info_title'][lang_idx], message)
            logger.info(f"Speicherprüfung: {free/1024**3:.2f} GB frei")
            
        except Exception as e:
            texts = self.translator.translations["messagebox"]
            lang_idx = self.translator.language_map.get(self.language, 0)
            QMessageBox.warning(self, texts['storage_error_title'][lang_idx], 
                            f"{texts['storage_error_message'][lang_idx]}\n{str(e)}")
            


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


