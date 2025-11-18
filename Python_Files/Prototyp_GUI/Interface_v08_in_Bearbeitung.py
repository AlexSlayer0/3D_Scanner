"""=======TODO-Liste v0.6=======
Objekt-Detection muss verbessert werden
Lade-Dialog schöner gestalten
Fehlerbehandlung bei Kamerazugriff
SAP-Integration                 (Platzhalter-Button/optional)
Lokal speichern Integration     (Platzhalter-Button)
GUI-Design verbessern           (optional)
================================"""

import sys
import cv2
import os
import numpy as np
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, QGridLayout
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal

class TranslationManager:
    def __init__(self):
        self.translations = {
            "de": {
                "start": {
                    "title": "3D-Scanner Interface",
                    "subtitle": "Interface um den 3D-Scanner zu bedienen",
                    "instruction1": "Bitte lege den Artikel der gescannt werden soll in die Box ein",
                    "instruction2": "Stellen Sie sicher, dass der Artikel vollständig im Sichtfeld aller Kameras liegt",
                    "instruction3": "Maximale Größe: 50×50×50 cm",
                    "instruction4": "Maximales Gewicht: 20 kg",
                    "scan_btn": "Scan Starten",
                    "save_btn": "Lokal speichern",
                    "status_title": "System Status",
                    "camera_status": "Kamera System",
                    "light_status": "Beleuchtung", 
                    "measure_status": "Mess-System",
                    "scale_status": "Waage",
                    "storage_status": "Speicher",
                    "ready": "Bereit",
                    "active": "Aktiv",
                    "calibrated": "Kalibriert",
                    "connected": "Verbunden",
                    "available": "Verfügbar",
                    "refresh_btn": "Status aktualisieren"
                },
                "photo": {
                    "title": "Foto-Auswahl",
                    "retry_btn": "Wiederholen",
                    "discard_btn": "Verwerfen"
                },
                "overview": {
                    "title": "Kamera-Übersicht",
                    "dimensions": "Abmessungen:",
                    "weight": "Gewicht:",
                    "mm": "mm",
                    "kg": "kg"
                },
                "storage": {
                    "title": "Speicher Option",
                    "barcode_label": "Ausgewerteter Barcode:",
                    "barcode_type": "Barcode-Typ:",
                    "sap_btn": "SAP-Eintrag",
                    "save_btn": "Lokal speichern"
                }
            },
            "en": {
                "start": {
                    "title": "3D Scanner Interface",
                    "subtitle": "Interface to operate the 3D scanner",
                    "instruction1": "Please place the item to be scanned in the box",
                    "instruction2": "Make sure the item is completely in the field of view of all cameras", 
                    "instruction3": "Maximum size: 50×50×50 cm",
                    "instruction4": "Maximum weight: 20 kg",
                    "scan_btn": "Start Scan",
                    "save_btn": "Save Locally",
                    "status_title": "System Status",
                    "camera_status": "Camera System",
                    "light_status": "Lighting",
                    "measure_status": "Measurement System",
                    "scale_status": "Scale", 
                    "storage_status": "Storage",
                    "ready": "Ready",
                    "active": "Active",
                    "calibrated": "Calibrated",
                    "connected": "Connected",
                    "available": "Available",
                    "refresh_btn": "Refresh Status"
                },
                "photo": {
                    "title": "Photo Selection",
                    "retry_btn": "Retake",
                    "discard_btn": "Discard"
                },
                "overview": {
                    "title": "Camera Overview", 
                    "dimensions": "Dimensions:",
                    "weight": "Weight:",
                    "mm": "mm",
                    "kg": "kg"
                },
                "storage": {
                    "title": "Storage Options",
                    "barcode_label": "Barcode:",
                    "barcode_type": "Barcode Type:",
                    "sap_btn": "SAP Entry",
                    "save_btn": "Save Locally"
                }
            },
            "it": {
                "start": {
                    "title": "Interfaccia Scanner 3D",
                    "subtitle": "Interfaccia per gestire lo scanner 3D",
                    "instruction1": "Si prega di posizionare l'articolo nella scatola",
                    "instruction2": "Assicurarsi che l'articolo sia completamente nel campo visivo di tutte le telecamere",
                    "instruction3": "Dimensione massima: 50×50×50 cm",
                    "instruction4": "Peso massimo: 20 kg",
                    "scan_btn": "Avvia Scan", 
                    "save_btn": "Salva localmente",
                    "status_title": "Stato del Sistema",
                    "camera_status": "Sistema Fotocamera",
                    "light_status": "Illuminazione",
                    "measure_status": "Sistema di Misura",
                    "scale_status": "Bilancia",
                    "storage_status": "Memoria",
                    "ready": "Pronto",
                    "active": "Attivo", 
                    "calibrated": "Calibrato",
                    "connected": "Connesso",
                    "available": "Disponibile",
                    "refresh_btn": "Aggiorna Stato"
                },
                "photo": {
                    "title": "Selezione Foto",
                    "retry_btn": "Ripeti",
                    "discard_btn": "Scarta"
                },
                "overview": {
                    "title": "Panoramica Fotocamera",
                    "dimensions": "Dimensioni:",
                    "weight": "Peso:",
                    "mm": "mm", 
                    "kg": "kg"
                },
                "storage": {
                    "title": "Opzioni di Memorizzazione",
                    "barcode_label": "Barcode:",
                    "barcode_type": "Tipo di barcode:",
                    "sap_btn": "SAP Entry",
                    "save_btn": "Salva localmente"
                }
            }
        }
    
    def get_text(self, language, page, key):
        return self.translations.get(language, {}).get(page, {}).get(key, f"[{key}]")

class CameraManager:
    def __init__(self, debug_single_camera=False):
        self.debug_single_camera = debug_single_camera
        self.available_cameras = self._find_cameras()
    
    def _find_cameras(self):
        available = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def _enable_flash(self):
        # BLITZ-IMPLEMENTIERUNG HIER EINFÜGEN
        # GPIO, serielle Schnittstelle oder kamerainterner Blitz
        pass
    
    def _disable_flash(self):
        pass
    
    def _make_placeholder(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "BILD NICHT AUFGENOMMEN", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 255), 2)
        return img
    
    def take_picture(self, camera_id):
        if camera_id not in self.available_cameras:
            return self._make_placeholder()
        
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return self._make_placeholder()
            
            # Blitz hier aktivieren wenn gewünscht
            # self._enable_flash()
            
            ret, frame = cap.read()
            cap.release()
            
            # Blitz hier deaktivieren
            # self._disable_flash()
            
            return frame if ret else self._make_placeholder()
            
        except Exception:
            return self._make_placeholder()
    
    def take_all_pictures(self):
        images = []
        
        if self.debug_single_camera:
            # Debug: Eine Kamera für alle Bilder
            for i in range(4):
                img = self.take_picture(0)
                images.append(img)
        else:
            # Normal: Jede Kamera macht ein Bild
            for i in range(4):
                if i < len(self.available_cameras):
                    img = self.take_picture(i)
                else:
                    img = self._make_placeholder()
                images.append(img)
        
        return images

class ParallelWorker(QThread):
    output_received = pyqtSignal(str, object)  # (task_name, result)
    finished = pyqtSignal()

    def __init__(self, images):
        super().__init__()
        self.images = images  # Liste mit 4 Bildern

    def run(self):
        # --- Worker-Funktionen ---
        def run_yolo():
            try:
                import BoundingBox_Yolo03 as yolo_module
                
                all_dimensions = []   # Liste für alle Bilder
                all_frames = []       # Annotierte Frames

                for idx, frame in enumerate(self.images):
                    if frame is None:
                        all_dimensions.append("0 x 0")
                        all_frames.append(None)
                        continue

                    boxes_info = yolo_module.get_boxes_and_dimensions(frame)

                    if boxes_info:
                        dim_str = f"{boxes_info[0]['width']} x {boxes_info[0]['height']}"
                    else:
                        dim_str = "0 x 0"

                    all_dimensions.append(dim_str)

                    # Annotiertes Bild
                    frame_with_boxes = yolo_module.draw_boxes(frame, boxes_info)
                    all_frames.append(frame_with_boxes)

                # Emit: Dimensionen + annotierte Frames
                self.output_received.emit("Abmessung", all_dimensions)
                self.output_received.emit("yolo_frames", all_frames)

            except Exception as e:
                self.output_received.emit("Abmessung", f"Fehler: {e}")

        def run_barcode():
            try:
                from BarCode_v02 import process_roi
                from ultralytics import YOLO

                model = YOLO("YOLOV8s_Barcode_Detection.pt")  # Einmal laden

                for idx, img in enumerate(self.images):
                    if img is None:
                        self.output_received.emit('barcode', {"index": idx, "found": False})
                        continue

                    found = False
                    decoded_value = None
                    decoded_type = None

                    results = model.predict(img)

                    for r in results:
                        for box in r.boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box)
                            roi = img[y1:y2, x1:x2]
                            result = process_roi(roi, f"image_{idx}")
                            if result["found"]:
                                found = True
                                decoded_value = result["value"]
                                decoded_type = result["type"]
                                break

                        if found:
                            break

                    self.output_received.emit('barcode', {
                        "index": idx,
                        "found": found,
                        "value": decoded_value,
                        "type": decoded_type
                    })

            except Exception as e:
                self.output_received.emit('barcode', {"index": idx, "found": False, "error": str(e)})

        def run_weight():
            import Gewichts_Messung
            w = Gewichts_Messung.get_weight()
            self.output_received.emit('weight', w)

        # --- Threads starten ---
        threads = []
        for func in [run_yolo, run_barcode, run_weight]:
            t = threading.Thread(target=func)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        self.finished.emit()

class FullscreenApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D-Scanner")
        self.showFullScreen()

        #Erst-Anpassung------------------------------------------------------
        self.camera = CameraManager(debug_single_camera=True)  # True = 1 Kamera, False = 4 Kameras
        self.translator = TranslationManager()

        self.language = "de"  # oder "it" / "en" standartmäßig
        self.Explorer_Structure = r"GUI_Anzeige"

        self.abmessung = None
        self.gewicht = None
        self.barcode = None

        self.images = [None]*4              # Platzhalter für die 4 Bilder
        self.image_labels = [None]*4        # Labels für die Bilder
        self.final_images = [None]*4        # Für Übersicht
        self.final_image_labels = [None]*4  #Labels für die fertigen Bilder

        self.keep = [True]*4        # True = Bild behalten, False = Bild verworfen 
        self.scan_start = False
        self.bilder_namen = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]   

        # Hauptcontainer
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        self.setCentralWidget(container)

        # Stacked widget für die Seiten
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # Button-Leiste (fix am unteren Rand)
        bar_layout = QHBoxLayout()
        self.back_btn = QPushButton("←")
        self.next_btn = QPushButton("→")
        self.back_btn.setFixedSize(100, 60)  # Breite x Höhe
        self.next_btn.setFixedSize(100, 60)
        font = self.back_btn.font()
        font.setPointSize(26)  # Schriftgröße
        self.back_btn.setFont(font)
        self.next_btn.setFont(font)
        self.back_btn.clicked.connect(self.go_back)
        self.next_btn.clicked.connect(self.go_next)

        bar_layout.addWidget(self.back_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(self.next_btn)
        main_layout.addLayout(bar_layout)
        QShortcut(QKeySequence("Left"), self, activated=self.go_back)
        QShortcut(QKeySequence("Right"), self, activated=self.go_next)

        self.load_pages()
        self.update_buttons()

    def set_language(self, language):
        self.language = language
        self.load_pages()
        self.update_buttons()

    def create_flag_button(self, flag_file, language_code):
        btn = QToolButton()
        btn.setIcon(QIcon(os.path.join(self.Explorer_Structure, flag_file)))
        btn.setIconSize(QSize(32, 32))
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QToolButton {
                border-radius: 20px;  /* rund */
            }""")
        btn.clicked.connect(lambda _, lang=language_code: self.set_language(lang))
        return btn
                
    def convert_to_pixmap(self, frame, width=300, height=300):
        # Prüfe ob es ein Platzhalter-Bild ist (schwarzes Bild)
        if frame is None or (isinstance(frame, np.ndarray) and np.all(frame == 0)):
            # Erstelle einen grauen Platzhalter
            gray_pixmap = QPixmap(width, height)
            gray_pixmap.fill(Qt.GlobalColor.lightGray)
            return gray_pixmap
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def retry_image(self, idx):
        print(f"🔄 Wiederhole Bild {idx+1}")
        self.scan_start = True
        new_img = self.camera.take_picture(idx)
        if new_img is not None:
            self.images[idx] = new_img
            pixmap = self.convert_to_pixmap(new_img)
            self.image_labels[idx].setPixmap(pixmap)

    def discard_image(self, idx):
        print(f"❌ Verworfen Bild {idx+1}")
        self.scan_start = True
        self.keep[idx] = False
        label = self.image_labels[idx]
        gray_pixmap = QPixmap(label.pixmap().size())
        gray_pixmap.fill(Qt.GlobalColor.lightGray)
        label.setPixmap(gray_pixmap)

    def create_start_page(self):
        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Linke Spalte (2/3)
        left_column = self.create_start_left_column()
        main_layout.addWidget(left_column, stretch=2)

        # Rechte Spalte (1/3) - Status
        right_column = self.create_start_right_column()
        main_layout.addWidget(right_column, stretch=1)

        return page

    def create_start_left_column(self):
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setSpacing(20)
        
        # Titel
        title_label = QLabel(self.translator.get_text(self.language, "start", "title"))
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #ffffff; margin-bottom: 20px;")
        left_layout.addWidget(title_label)

        # Texte
        texts = [
            self.translator.get_text(self.language, "start", "subtitle"),
            self.translator.get_text(self.language, "start", "instruction1"),
            self.translator.get_text(self.language, "start", "instruction2"),
            self.translator.get_text(self.language, "start", "instruction3"),
            self.translator.get_text(self.language, "start", "instruction4")
        ]
        
        for text in texts:
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setWordWrap(True)
            label.setStyleSheet("color: #bdc3c7; font-size: 16px; padding: 12px 0; line-height: 1.5;")
            left_layout.addWidget(label)

        # Buttons
        button_layout = QHBoxLayout()
        scan_btn = QPushButton(self.translator.get_text(self.language, "start", "scan_btn"))
        save_btn = QPushButton(self.translator.get_text(self.language, "start", "save_btn"))
        
        for btn in [scan_btn, save_btn]:
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
                }
                QPushButton:hover {
                    background: #6c757d;
                }
                QPushButton:pressed {
                    background: #343a40;
                }
            """)
            btn.setFixedHeight(50)
            button_layout.addWidget(btn)

        scan_btn.clicked.connect(self.go_next)
        left_layout.addLayout(button_layout)
        left_layout.addStretch()

        return left_column

    def create_start_right_column(self):
        right_column = QWidget()
        right_column.setFixedWidth(300)
        right_layout = QVBoxLayout(right_column)
        right_layout.setSpacing(15)

        # Status-Überschrift
        status_title = QLabel(self.translator.get_text(self.language, "start", "status_title"))
        status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_title.setStyleSheet("font-size: 20px; font-weight: 600; color: #3498db; padding: 10px 0; border-bottom: 2px solid #34495e;")
        right_layout.addWidget(status_title)

        # Status-Items
        status_items = [
            ("✅", self.translator.get_text(self.language, "start", "camera_status"), self.translator.get_text(self.language, "start", "ready")),
            ("⚡", self.translator.get_text(self.language, "start", "light_status"), self.translator.get_text(self.language, "start", "active")),
            ("📏", self.translator.get_text(self.language, "start", "measure_status"), self.translator.get_text(self.language, "start", "calibrated")),
            ("⚖️", self.translator.get_text(self.language, "start", "scale_status"), self.translator.get_text(self.language, "start", "connected")),
            ("💾", self.translator.get_text(self.language, "start", "storage_status"), self.translator.get_text(self.language, "start", "available"))
        ]

        for icon, name, status in status_items:
            status_widget = self.create_status_item(icon, name, status)
            right_layout.addWidget(status_widget)

        refresh_btn = QPushButton("⟳ " + self.translator.get_text(self.language, "start", "refresh_btn"))
        refresh_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px; 
                padding: 8px; 
                background: #34495e; 
                color: #7f8c8d; 
                border: 1px solid #2c3e50; 
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #3d566e;
            }
        """)
        refresh_btn.setFixedHeight(35)
        right_layout.addWidget(refresh_btn)
        right_layout.addStretch()

        return right_column

    def create_status_item(self, icon, name, status):
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(10, 5, 10, 5)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")
        
        name_label = QLabel(name)
        name_label.setStyleSheet("color: #ecf0f1; font-size: 14px; font-weight: 500;")
        
        status_label = QLabel(status)
        status_label.setStyleSheet("color: #2ecc71; font-size: 13px; font-weight: 400;")
        
        status_layout.addWidget(icon_label)
        status_layout.addWidget(name_label)
        status_layout.addStretch()
        status_layout.addWidget(status_label)
        
        return status_widget

    def make_card(self, text):
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
                border-bottom: 1px solid #34495e;
                line-height: 1.5;
            }
        """)
        label.setWordWrap(True)
        return label

    def make_card_with_input(self, label_text="", preset_text="", placeholder=""):
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
                color: #2c3e50;
            } QLineEdit {
                font-size: 20px;
                color: #2c3e50;
                background: transparent;
                border: none;
                border-bottom: 2px solid #2c3e50;
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

    def _make_widget(self, item):
        if not isinstance(item, tuple):
            return self.make_card(str(item))
        
        widget_type = item[0]
        widget_creators = {
            "button": self._create_button_widget,
            "image": self._create_image_widget, 
            "ram_image": self._create_ram_image_widget,
            "ram_image_final": self._create_ram_image_final_widget,
            "title": self._create_title_widget,
            "input": self._create_input_widget
        }
        creator = widget_creators.get(widget_type)
        if creator:
            return creator(*item[1:])
        return self.make_card(str(item))

    def _create_button_widget(self, text, callback=None):
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

    def _create_image_widget(self, base_name):
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

    def _create_ram_image_widget(self, idx):
        label = QLabel()
        self.image_labels[idx] = label
        if self.images[idx] is not None:
            pixmap = self.convert_to_pixmap(self.images[idx])
            label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_ram_image_final_widget(self, idx):
        label = QLabel()
        self.final_image_labels[idx] = label
        if self.final_images[idx] is not None:
            label.setPixmap(self.convert_to_pixmap(self.final_images[idx]))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_title_widget(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dedede;")
        return label

    def _create_input_widget(self, label_text, placeholder="", preset_text=""):
        return self.make_card_with_input(label_text, preset_text, placeholder)

    def add_page(self, title, labels):
        page = QWidget()
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
        title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #2c3ea0;")
        title_layout.addWidget(title_label, stretch=1)

        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it")
        btn_en = self.create_flag_button("en.png", "en")

        for btn in [btn_de, btn_it, btn_en]:
            title_layout.addWidget(btn)

        page_layout.addWidget(title_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        for item in labels:
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
        if self.abmessung == None:
            self.abmessung = "Undefiniert"
        if self.gewicht == None:
            self.gewicht = "Undefiniert"
        if self.barcode == None:
            self.barcode = "Undefiniert"
            self.barcode_type = "Undefiniert"
        
        # Alte Seiten entfernen
        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        # Startseite mit speziellem Layout
        start_page = self.create_start_page()
        self.stack.addWidget(start_page)

        # Weitere Seiten mit generischem System
        if self.language == "de":
            self.add_page(self.translator.get_text(self.language, "photo", "title"), 
                    [[("ram_image",0), ("ram_image",1)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=0: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=1: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=0: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=1: self.discard_image(idx))],

                    [("ram_image",2),("ram_image",3)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=2: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=3: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=2: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=3: self.discard_image(idx))]])
                
            self.add_page(self.translator.get_text(self.language, "overview", "title"), 
                    [[("ram_image_final", 0), ("ram_image_final", 1)], [("ram_image_final", 2), ("ram_image_final", 3)], 
                    f"{self.translator.get_text(self.language, 'overview', 'dimensions')} {self.abmessung}{self.translator.get_text(self.language, 'overview', 'mm')}", 
                    f"{self.translator.get_text(self.language, 'overview', 'weight')} {self.gewicht}{self.translator.get_text(self.language, 'overview', 'kg')}"])
            self.add_page(self.translator.get_text(self.language, "storage", "title"), 
                    [("image", "barcode"), 
                    ("input", self.translator.get_text(self.language, "storage", "barcode_label"), f"{self.barcode}"),
                    ("input", self.translator.get_text(self.language, "storage", "barcode_type"), f"{self.barcode_type}"), 
                    [("button", self.translator.get_text(self.language, "storage", "sap_btn")), 
                     ("button", self.translator.get_text(self.language, "storage", "save_btn"))]])

        elif self.language == "it":
            self.add_page(self.translator.get_text(self.language, "photo", "title"), 
                    [[("ram_image", 0), ("ram_image", 1)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=0: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=1: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=0: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=1: self.discard_image(idx))],

                    [("ram_image", 2), ("ram_image", 3)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=2: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=3: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=2: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=3: self.discard_image(idx))]])
            
            self.add_page(self.translator.get_text(self.language, "overview", "title"),
                    [[("ram_image_final", 0), ("ram_image_final", 1)], [("ram_image_final", 2), ("ram_image_final", 3)], 
                    f"{self.translator.get_text(self.language, 'overview', 'dimensions')} {self.abmessung}{self.translator.get_text(self.language, 'overview', 'mm')}", 
                    f"{self.translator.get_text(self.language, 'overview', 'weight')} {self.gewicht}{self.translator.get_text(self.language, 'overview', 'kg')}"])
            self.add_page(self.translator.get_text(self.language, "storage", "title"),
                    [("image", "barcode"), 
                    ("input", self.translator.get_text(self.language, "storage", "barcode_label"), f"{self.barcode}"),
                    ("input", self.translator.get_text(self.language, "storage", "barcode_type"), f"{self.barcode_type}"),
                    [("button", self.translator.get_text(self.language, "storage", "sap_btn")), 
                     ("button", self.translator.get_text(self.language, "storage", "save_btn"))]])

        else:  # englisch
            self.add_page(self.translator.get_text(self.language, "photo", "title"), 
                    [[("ram_image", 0), ("ram_image", 1)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=0: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=1: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=0: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=1: self.discard_image(idx))],

                    [("ram_image", 2), ("ram_image", 3)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=2: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=3: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=2: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=3: self.discard_image(idx))]])
            
            self.add_page(self.translator.get_text(self.language, "overview", "title"),
                    [[("ram_image_final", 0), ("ram_image_final", 1)], [("ram_image_final", 2), ("ram_image_final", 3)],
                    f"{self.translator.get_text(self.language, 'overview', 'dimensions')} {self.abmessung}{self.translator.get_text(self.language, 'overview', 'mm')}", 
                    f"{self.translator.get_text(self.language, 'overview', 'weight')} {self.gewicht}{self.translator.get_text(self.language, 'overview', 'kg')}"])
            self.add_page(self.translator.get_text(self.language, "storage", "title"),
                    [("image", "barcode"), 
                    ("input", self.translator.get_text(self.language, "storage", "barcode_label"), f"{self.barcode}"),
                    ("input", self.translator.get_text(self.language, "storage", "barcode_type"), f"{self.barcode_type}"), 
                    [("button", self.translator.get_text(self.language, "storage", "sap_btn")), 
                     ("button", self.translator.get_text(self.language, "storage", "save_btn"))]])

    def go_back(self):
        idx = self.stack.currentIndex()
        
        if idx == 1:
            self.scan_start = False
            reply = QMessageBox.question(
                self,
                "Datenverlust bestätigen",
                "Möchten Sie wirklich zurück zur Startseite? Alle erfassten Daten gehen verloren.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

            if reply == QMessageBox.StandardButton.Cancel:
                return
    
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self.update_buttons()

    def go_next(self):
        idx = self.stack.currentIndex()

        if idx >= self.stack.count() - 1:
            return
        
        elif idx == 0 and self.scan_start == False:
            self.scan_start = True
            if not hasattr(self, "images"):
                self.images = [None] * 4

            all_images = self.camera.take_all_pictures()
            for i, img in enumerate(all_images):
                self.images[i] = img
                if self.image_labels[i] is not None:
                    self.image_labels[i].setPixmap(self.convert_to_pixmap(img))

            self.stack.setCurrentIndex(idx + 1)
            self.update_buttons()
            return

        elif idx == 1 and self.scan_start == True:
            self.loading_dialog = QDialog(self)
            self.loading_dialog.setWindowTitle("Ladevorgang der Daten")
            self.loading_dialog.setModal(True)
            self.loading_dialog.setFixedSize(350, 400)

            layout = QVBoxLayout(self.loading_dialog)
            movie = QMovie(os.path.join(self.Explorer_Structure, "loading.gif"))
            gif_label = QLabel()
            gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gif_label.setMovie(movie)
            movie.start()
            layout.addWidget(gif_label)

            status_label = QLabel("Daten werden verarbeitet...")
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet("font-size: 16px; margin: 20px;")
            layout.addWidget(status_label)

            cancel_btn = QPushButton("Abbrechen")
            cancel_btn.setFixedSize(120, 40)
            cancel_btn.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    padding: 8px;
                }
            """)
            layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)

            self.start_worker()

            def finish_loading():
                if self.loading_dialog.isVisible():
                    self.loading_dialog.accept()
                    self.stack.setCurrentIndex(idx + 1)
                    self.update_buttons()
                    self.scan_start = False
                    QMessageBox.information(
                        self,
                        "Scan abgeschlossen",
                        "Der Scan war erfolgreich!\nDie Daten stehen nun zur Verfügung."
                    )

            self.worker.finished.connect(finish_loading)

            def cancel_loading():
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait()
                self.loading_dialog.reject()
                self.stack.setCurrentIndex(1)
                self.update_buttons()
                QMessageBox.warning(
                    self,
                    "Scan abgebrochen",
                    "Der Scan wurde abgebrochen."
                )

            cancel_btn.clicked.connect(cancel_loading)
            self.loading_dialog.exec()

        else:
            self.stack.setCurrentIndex(idx + 1)
            self.update_buttons()
            
    def update_buttons(self):
        current_index = self.stack.currentIndex()
        total_pages = self.stack.count()

        if current_index == 0:
            self.back_btn.hide()
            self.next_btn.hide()
        elif current_index == total_pages - 1:
            self.next_btn.hide()
        else:
            self.back_btn.show()
            self.next_btn.show()
    
    def start_worker(self):
        self.worker = ParallelWorker(self.images)
        self.worker.output_received.connect(self.handle_output)
        self.worker.finished.connect(lambda: print("Alle Tasks fertig"))
        self.worker.start()

    def handle_output(self, script_name, data):
        print(f"\n========== Debug [{script_name}] ==========")

        if script_name == "Abmessung":
            if isinstance(data, list) and len(data) >= 3:
                self.abmessungen = data

                try:
                    length = int(data[0].split(" x ")[0])
                    width  = int(data[1].split(" x ")[0])
                    height = int(data[2].split(" x ")[1])

                    self.abmessung_gesamt = f"{length} x {width} x {height}"
                except Exception as e:
                    print(f"Fehler beim Berechnen der Gesamt-Abmessung: {e}")
                    self.abmessung_gesamt = "Undefiniert"

                print(f"🔹 Gesamt-Abmessung: {self.abmessung_gesamt}")
                self.abmessung = self.abmessung_gesamt

                for idx, dim in enumerate(data):
                    print(f"Bild {idx}: Original = {dim}")

            else:
                self.abmessungen = []
                self.abmessung_gesamt = "Undefiniert"
                print(f"Fehler bei Abmessung: {data}")

        elif script_name == "yolo_frames":
            self.annotierte_frames = data
            for idx, frame in enumerate(data):
                if frame is not None:
                    print(f"Bild {idx}: Frame mit Bounding Boxen erhalten")
                else:
                    print(f"Bild {idx}: Kein Frame vorhanden")

        elif script_name == "barcode":
            idx = data.get("index", -1)
            found = data.get("found", False)
            value = data.get("value", None)
            b_type = data.get("type", None)
            error = data.get("error", None)

            if found:
                print(f"✅ Barcode erkannt in Bild {idx}: Wert='{value}', Typ='{b_type}'")
            elif error:
                print(f"❌ Barcode Fehler in Bild {idx}: {error}")
            else:
                print(f"❌ Kein Barcode in Bild {idx}")

            self.barcode_type = b_type if found else "Undefiniert"
            self.barcode = value if found else "Undefiniert"

        elif script_name == "weight":
            self.gewicht = data
            print(f"Gewicht: {data}")

        else:
            print(f"Unbekanntes Script '{script_name}': {data}")

        print("========== Ende Debug ==========\n")

        abmessung_ready = hasattr(self, "abmessung_gesamt") and self.abmessung_gesamt != "Undefiniert"
        barcode_ready = hasattr(self, "barcode") and any(self.barcode)
        gewicht_ready = hasattr(self, "gewicht") and self.gewicht not in ["Undefiniert", None]

        if abmessung_ready and barcode_ready and gewicht_ready:
            for i in range(4):
                if self.keep[i] and self.images[i] is not None:
                    self.final_images[i] = self.images[i].copy()
                else:
                    self.final_images[i] = None

            self.load_pages()
            self.stack.setCurrentIndex(2)
            self.update_buttons()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.go_back()
        elif event.key() == Qt.Key.Key_Right:
            self.go_next()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FullscreenApp()
    w.show()
    sys.exit(app.exec())