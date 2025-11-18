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
import threading
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, QProgressBar  # QProgressBar hinzugefügt
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal


class CameraManager:
    """Kapselt die Kamerafunktionalität"""
    @staticmethod
    def capture_single(idx):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"❌ Kamera {idx} nicht verfügbar!")
            return None
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        print(f"⚠️ Kein Bild von der Kamera {idx} erhalten!")
        return None

    @staticmethod
    def capture_images(cam_indices=(0, 1, 2)):
        images = []
        for idx in cam_indices:
            img = CameraManager.capture_single(idx)
            images.append(img)
        return images


class ImageProcessor:
    """Hilfsklasse für Bildverarbeitung"""
    @staticmethod
    def convert_to_pixmap(frame, width=300, height=300):
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class ParallelWorker(QThread):
    """Worker für parallele Verarbeitung - interne Funktionen unverändert"""
    output_received = pyqtSignal(str, object)
    finished = pyqtSignal()

    def __init__(self, images):
        super().__init__()
        self.images = images

    def run(self):
        def run_yolo():
            try:
                import BoundingBox_Yolo03 as yolo_module
                
                all_dimensions = []
                all_frames = []

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
                    frame_with_boxes = yolo_module.draw_boxes(frame, boxes_info)
                    all_frames.append(frame_with_boxes)

                self.output_received.emit("Abmessung", all_dimensions)
                self.output_received.emit("yolo_frames", all_frames)

            except Exception as e:
                self.output_received.emit("Abmessung", f"Fehler: {e}")

        def run_barcode():
            try:
                from BarCode_v02 import process_roi
                from ultralytics import YOLO

                model = YOLO("YOLOV8s_Barcode_Detection.pt")

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
            try:
                import Gewichts_Messung
                w = Gewichts_Messung.get_weight()
                self.output_received.emit('weight', w)
            except Exception as e:
                self.output_received.emit('weight', f"Fehler: {e}")

        threads = []
        for func in [run_yolo, run_barcode, run_weight]:
            t = threading.Thread(target=func)
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()

        self.finished.emit()


class LoadingDialog(QDialog):
    """Lade-Dialog mit verbessertem Design"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ladevorgang der Daten")
        self.setModal(True)
        self.setFixedSize(400, 200)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Lade-Animation
        movie = QMovie(os.path.join("GUI_Anzeige", "loading.gif"))
        gif_label = QLabel()
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gif_label.setMovie(movie)
        movie.start()
        layout.addWidget(gif_label)

        # Status-Text
        self.status_label = QLabel("Daten werden verarbeitet...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Fortschrittsbalken
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Unbestimmter Modus
        layout.addWidget(self.progress_bar)

        # Abbrechen-Button
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)


class BasePage(QWidget):
    """Basisklasse für alle Seiten mit Sprachumschaltung"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        """Setup der Seitenstruktur mit Titelleiste"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Titelleiste mit Sprachumschaltung
        title_bar = self.create_title_bar()
        layout.addWidget(title_bar)

        # Inhaltsbereich
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)

        # Seiten-spezifischen Inhalt hinzufügen
        self.setup_content()

    def create_title_bar(self):
        """Erstellt die Titelleiste mit Sprachumschaltung"""
        title_bar = QWidget()
        title_bar.setFixedHeight(70)
        title_bar.setStyleSheet("background-color: #2c3e50; padding: 10px;")
        
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 10, 20, 10)

        # Seitentitel
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()

        # Sprachumschalt-Buttons
        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it") 
        btn_en = self.create_flag_button("en.png", "en")

        for btn in [btn_de, btn_it, btn_en]:
            title_layout.addWidget(btn)

        return title_bar

    def create_flag_button(self, flag_file, language_code):
        """Erstellt einen Sprachumschalt-Button"""
        btn = QToolButton()
        flag_path = os.path.join("GUI_Anzeige", flag_file)
        if os.path.exists(flag_path):
            btn.setIcon(QIcon(flag_path))
        btn.setIconSize(QSize(32, 32))
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QToolButton {
                border-radius: 20px;
                background-color: #34495e;
            }
            QToolButton:hover {
                background-color: #4a6a8a;
            }
        """)
        btn.clicked.connect(lambda: self.parent_app.set_language(language_code))
        return btn

    def setup_content(self):
        """Muss von Unterklassen implementiert werden"""
        raise NotImplementedError("Subclasses must implement setup_content")


class StartPage(BasePage):
    """Startseite der Anwendung"""
    def setup_content(self):
        layout = self.content_layout
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Titel
        title_label = QLabel("3D-Scanner Interface")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dedede;")
        layout.addWidget(title_label)

        # Beschreibungstexte
        if self.parent_app.language == "de":
            desc1 = QLabel("Interface um den 3D-Scanner zu bedienen")
            desc2 = QLabel("Bitte lege den Artikel der gescannt werden soll in die Box ein")
        elif self.parent_app.language == "it":
            desc1 = QLabel("Interfaccia per gestire lo scanner 3D")
            desc2 = QLabel("Si prega di posizionare l'articolo nella scatola")
        else:  # englisch
            desc1 = QLabel("Interface to operate the 3D scanner")
            desc2 = QLabel("Please place the item in the box")

        for label in [desc1, desc2]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 18px; color: #cccccc;")
            layout.addWidget(label)

        # Buttons
        button_layout = QHBoxLayout()
        
        if self.parent_app.language == "de":
            scan_btn = QPushButton("Scan Starten")
            save_btn = QPushButton("Lokal speichern")
        elif self.parent_app.language == "it":
            scan_btn = QPushButton("Avvia Scan")
            save_btn = QPushButton("Local")
        else:  # englisch
            scan_btn = QPushButton("Start Scan")
            save_btn = QPushButton("Local save")
        
        for btn in [scan_btn, save_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px; 
                    padding: 15px;
                    background-color: #3498db;
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            btn.setFixedHeight(60)
            button_layout.addWidget(btn)

        scan_btn.clicked.connect(self.parent_app.go_next)
        # save_btn.clicked.connect()  # TODO: Implementieren
        
        layout.addLayout(button_layout)
        layout.addStretch()


class PhotoSelectionPage(BasePage):
    """Seite für Foto-Auswahl und -Bearbeitung"""
    def setup_content(self):
        layout = self.content_layout
        layout.setSpacing(15)
        layout.setContentsMargins(40, 40, 40, 40)

        # Erste Bildreihe (0, 1)
        row1_layout = QHBoxLayout()
        self.setup_image_row(row1_layout, [0, 1])
        layout.addLayout(row1_layout)

        # Zweite Bildreihe (2, 3)
        row2_layout = QHBoxLayout()
        self.setup_image_row(row2_layout, [2, 3])
        layout.addLayout(row2_layout)

    def setup_image_row(self, layout, indices):
        for idx in indices:
            container = QVBoxLayout()
            container.setSpacing(5)

            # Bild-Label
            label = QLabel()
            self.parent_app.image_labels[idx] = label
            if self.parent_app.images[idx] is not None:
                pixmap = ImageProcessor.convert_to_pixmap(self.parent_app.images[idx])
                label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 2px solid #cccccc; border-radius: 5px;")
            container.addWidget(label)

            # Buttons
            btn_layout = QHBoxLayout()
            
            if self.parent_app.language == "de":
                retry_btn = QPushButton("Wiederholen")
                discard_btn = QPushButton("Verwerfen")
            elif self.parent_app.language == "it":
                retry_btn = QPushButton("Ripeti")
                discard_btn = QPushButton("Scarta")
            else:  # englisch
                retry_btn = QPushButton("Retake")
                discard_btn = QPushButton("Discard")
            
            for btn in [retry_btn, discard_btn]:
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 14px; 
                        padding: 8px;
                        background-color: #95a5a6;
                        color: white;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #7f8c8d;
                    }
                """)
                btn.setFixedHeight(40)

            retry_btn.clicked.connect(lambda _, i=idx: self.parent_app.retry_image(i))
            discard_btn.clicked.connect(lambda _, i=idx: self.parent_app.discard_image(i))
            
            btn_layout.addWidget(retry_btn)
            btn_layout.addWidget(discard_btn)
            container.addLayout(btn_layout)

            layout.addLayout(container)


class CameraOverviewPage(BasePage):
    """Übersichtsseite mit allen Bildern und Messdaten"""
    def setup_content(self):
        layout = self.content_layout
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Bildreihen
        row1_layout = QHBoxLayout()
        self.setup_final_images(row1_layout, [0, 1])
        layout.addLayout(row1_layout)

        row2_layout = QHBoxLayout()
        self.setup_final_images(row2_layout, [2, 3])
        layout.addLayout(row2_layout)

        # Messdaten
        data_layout = QVBoxLayout()
        
        if self.parent_app.language == "de":
            dimensions_label = QLabel(f"Abmessungen: {self.parent_app.abmessung}mm")
            weight_label = QLabel(f"Gewicht: {self.parent_app.gewicht}kg")
        elif self.parent_app.language == "it":
            dimensions_label = QLabel(f"Dimensioni: {self.parent_app.abmessung}mm")
            weight_label = QLabel(f"Peso: {self.parent_app.gewicht}kg")
        else:  # englisch
            dimensions_label = QLabel(f"Dimensions: {self.parent_app.abmessung}mm")
            weight_label = QLabel(f"Weight: {self.parent_app.gewicht}kg")
        
        for label in [dimensions_label, weight_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 20px; color: #dedede;")
            data_layout.addWidget(label)

        layout.addLayout(data_layout)

    def setup_final_images(self, layout, indices):
        for idx in indices:
            label = QLabel()
            self.parent_app.final_image_labels[idx] = label
            if self.parent_app.final_images[idx] is not None:
                pixmap = ImageProcessor.convert_to_pixmap(self.parent_app.final_images[idx])
                label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 2px solid #cccccc; border-radius: 5px;")
            layout.addWidget(label)


class StorageOptionsPage(BasePage):
    """Seite für Speicheroptionen"""
    def setup_content(self):
        layout = self.content_layout
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Barcode-Bild
        barcode_label = QLabel()
        barcode_path = self.find_barcode_image()
        if barcode_path:
            pixmap = QPixmap(barcode_path)
            pixmap = pixmap.scaledToWidth(250, Qt.TransformationMode.SmoothTransformation)
            barcode_label.setPixmap(pixmap)
        else:
            barcode_label.setText("Barcode Bild nicht gefunden")
        barcode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(barcode_label)

        # Eingabefelder
        if self.parent_app.language == "de":
            barcode_input = self.create_input_field("Ausgewerteter Barcode", self.parent_app.barcode)
            barcode_type_input = self.create_input_field("Barcode-Typ:", self.parent_app.barcode_type)
        elif self.parent_app.language == "it":
            barcode_input = self.create_input_field("Barcode:", self.parent_app.barcode)
            barcode_type_input = self.create_input_field("Tipo di barcode:", self.parent_app.barcode_type)
        else:  # englisch
            barcode_input = self.create_input_field("Barcode:", self.parent_app.barcode)
            barcode_type_input = self.create_input_field("Barcode Type:", self.parent_app.barcode_type)
        
        layout.addWidget(barcode_input)
        layout.addWidget(barcode_type_input)

        # Buttons
        button_layout = QHBoxLayout()
        
        if self.parent_app.language == "de":
            sap_btn = QPushButton("SAP-Eintrag")
            save_btn = QPushButton("Lokal speichern")
        elif self.parent_app.language == "it":
            sap_btn = QPushButton("SAP Entry")
            save_btn = QPushButton("Salva localmente")
        else:  # englisch
            sap_btn = QPushButton("SAP Entry")
            save_btn = QPushButton("Save Locally")
        
        for btn in [sap_btn, save_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 20px; 
                    padding: 15px;
                    background-color: #27ae60;
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #219a52;
                }
            """)
            btn.setFixedHeight(60)
            button_layout.addWidget(btn)

        # TODO: Button-Funktionen implementieren
        layout.addLayout(button_layout)

    def find_barcode_image(self):
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            path = os.path.join("GUI_Anzeige", "barcode" + ext)
            if os.path.exists(path):
                return path
        return None

    def create_input_field(self, label_text, preset_text):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        frame.setMinimumHeight(80)
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
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        field = QLineEdit()
        field.setText(preset_text)
        field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(label)
        layout.addWidget(field)
        
        return frame


class FullscreenApp(QMainWindow):
    """Hauptanwendung mit optimierter Struktur"""
    def __init__(self):
        super().__init__()
        self.setup_app_data()
        self.setup_ui()
        self.setup_shortcuts()

    def setup_app_data(self):
        """Initialisiert Anwendungsdaten"""
        self.language = "de"
        self.Explorer_Structure = "GUI_Anzeige"  # Wichtig für Sprachumschaltung
        
        self.abmessung = "Undefiniert"
        self.gewicht = "Undefiniert"
        self.barcode = "Undefiniert"
        self.barcode_type = "Undefiniert"
        
        self.images = [None] * 4
        self.image_labels = [None] * 4
        self.final_images = [None] * 4
        self.final_image_labels = [None] * 4
        
        self.keep = [True] * 4
        self.scan_start = False
        self.bilder_namen = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]

    def setup_ui(self):
        """Setup der Benutzeroberfläche"""
        self.setWindowTitle("3D-Scanner")
        self.showFullScreen()

        # Hauptcontainer
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Keine Margins, da Seiten eigene haben
        main_layout.setSpacing(0)
        self.setCentralWidget(container)

        # Stacked Widget für Seiten
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Navigationsleiste
        self.setup_navigation_bar(main_layout)

        # Seiten laden
        self.load_pages()
        self.update_buttons()

    def setup_navigation_bar(self, parent_layout):
        """Setup der Navigationsleiste"""
        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(20, 10, 20, 20)
        
        self.back_btn = QPushButton("←")
        self.next_btn = QPushButton("→")
        
        for btn in [self.back_btn, self.next_btn]:
            btn.setFixedSize(100, 60)
            font = btn.font()
            font.setPointSize(26)
            btn.setFont(font)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #34495e;
                    color: white;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #4a6a8a;
                }
            """)

        self.back_btn.clicked.connect(self.go_back)
        self.next_btn.clicked.connect(self.go_next)

        bar_layout.addWidget(self.back_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(self.next_btn)
        parent_layout.addLayout(bar_layout)

    def setup_shortcuts(self):
        """Setup der Tastaturkürzel"""
        QShortcut(QKeySequence("Left"), self, activated=self.go_back)
        QShortcut(QKeySequence("Right"), self, activated=self.go_next)

    def set_language(self, language):
        """Wechselt die Sprache der Anwendung"""
        self.language = language
        current_index = self.stack.currentIndex()

        # Alte Seiten entfernen
        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        # Neue Seiten laden
        self.load_pages()

        # Zur vorherigen Position zurückkehren
        if current_index < self.stack.count():
            self.stack.setCurrentIndex(current_index)
        else:
            self.stack.setCurrentIndex(self.stack.count() - 1)

        self.update_buttons()

    def load_pages(self):
        """Lädt die Seiten basierend auf der aktuellen Sprache"""
        # Update Daten falls nötig
        self.abmessung = self.abmessung or "Undefiniert"
        self.gewicht = self.gewicht or "Undefiniert"
        self.barcode = self.barcode or "Undefiniert"
        self.barcode_type = self.barcode_type or "Undefiniert"

        # Seiten je nach Sprache hinzufügen
        if self.language == "de":
            self.stack.addWidget(StartPage("Startseite", self))
            self.stack.addWidget(PhotoSelectionPage("Foto-Auswahl", self))
            self.stack.addWidget(CameraOverviewPage("Kamera-Übersicht", self))
            self.stack.addWidget(StorageOptionsPage("Speicher Option", self))
        elif self.language == "it":
            self.stack.addWidget(StartPage("Pagina Iniziale", self))
            self.stack.addWidget(PhotoSelectionPage("Selezione Foto", self))
            self.stack.addWidget(CameraOverviewPage("Panoramica Fotocamera", self))
            self.stack.addWidget(StorageOptionsPage("Opzioni di Memorizzazione", self))
        else:  # englisch
            self.stack.addWidget(StartPage("Home", self))
            self.stack.addWidget(PhotoSelectionPage("Photo Selection", self))
            self.stack.addWidget(CameraOverviewPage("Camera Overview", self))
            self.stack.addWidget(StorageOptionsPage("Storage Options", self))

    def retry_image(self, idx):
        """Wiederholt die Aufnahme eines Bildes"""
        print(f"🔄 Wiederhole Bild {idx+1}")
        self.scan_start = True
        new_img = CameraManager.capture_single(idx)
        if new_img is not None:
            self.images[idx] = new_img
            pixmap = ImageProcessor.convert_to_pixmap(new_img)
            if self.image_labels[idx] is not None:
                self.image_labels[idx].setPixmap(pixmap)

    def discard_image(self, idx):
        """Verwirft ein Bild"""
        print(f"❌ Verworfen Bild {idx+1}")
        self.scan_start = True
        self.keep[idx] = False
        label = self.image_labels[idx]
        if label and label.pixmap():
            gray_pixmap = QPixmap(label.pixmap().size())
            gray_pixmap.fill(Qt.GlobalColor.lightGray)
            label.setPixmap(gray_pixmap)

    def go_back(self):
        """Geht zur vorherigen Seite"""
        idx = self.stack.currentIndex()
        
        if idx == 1:
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
        """Geht zur nächsten Seite"""
        idx = self.stack.currentIndex()

        if idx >= self.stack.count() - 1:
            return
        
        elif idx == 0 and not self.scan_start:
            self.start_scanning()
        elif idx == 1 and self.scan_start:
            self.start_processing()
        else:
            self.stack.setCurrentIndex(idx + 1)
            self.update_buttons()

    def start_scanning(self):
        """Startet den Scan-Vorgang"""
        self.scan_start = True
        self.images = [None] * 4

        for i in range(len(self.image_labels)):
            img = CameraManager.capture_single(i)
            if img is not None:
                self.images[i] = img
                if i < len(self.image_labels) and self.image_labels[i] is not None:
                    self.image_labels[i].setPixmap(ImageProcessor.convert_to_pixmap(img))

        self.stack.setCurrentIndex(1)
        self.update_buttons()

    def start_processing(self):
        """Startet die Datenverarbeitung"""
        self.loading_dialog = LoadingDialog(self)
        
        self.start_worker()

        def finish_loading():
            if self.loading_dialog.isVisible():
                self.loading_dialog.accept()
                self.stack.setCurrentIndex(2)
                self.update_buttons()
                self.scan_start = False
                QMessageBox.information(
                    self,
                    "Scan abgeschlossen",
                    "Der Scan war erfolgreich!\nDie Daten stehen nun zur Verfügung."
                )

        self.worker.finished.connect(finish_loading)

        def cancel_loading():
            if hasattr(self, 'worker') and self.worker.isRunning():
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

        self.loading_dialog.rejected.connect(cancel_loading)
        self.loading_dialog.exec()

    def update_buttons(self):
        """Aktualisiert die Sichtbarkeit der Navigationsbuttons"""
        current_index = self.stack.currentIndex()
        total_pages = self.stack.count()

        self.back_btn.setVisible(current_index > 0)
        self.next_btn.setVisible(0 < current_index < total_pages - 1)

    def start_worker(self):
        """Startet den ParallelWorker"""
        self.worker = ParallelWorker(self.images)
        self.worker.output_received.connect(self.handle_output)
        self.worker.finished.connect(lambda: print("Alle Tasks fertig"))
        self.worker.start()

    def handle_output(self, script_name, data):
        """Verarbeitet die Ergebnisse der parallelen Verarbeitung"""
        print(f"\n========== Debug [{script_name}] ==========")

        try:
            if script_name == "Abmessung":
                self.handle_dimension_data(data)
            elif script_name == "yolo_frames":
                self.handle_yolo_frames(data)
            elif script_name == "barcode":
                self.handle_barcode_data(data)
            elif script_name == "weight":
                self.handle_weight_data(data)
            else:
                print(f"Unbekanntes Script '{script_name}': {data}")

            self.check_all_tasks_complete()

        except Exception as e:
            print(f"Fehler in handle_output: {e}")
            traceback.print_exc()

        print("========== Ende Debug ==========\n")

    def handle_dimension_data(self, data):
        """Verarbeitet Abmessungsdaten"""
        if isinstance(data, list) and len(data) >= 3:
            self.abmessungen = data

            try:
                length = int(data[0].split(" x ")[0])
                width = int(data[1].split(" x ")[0])
                height = int(data[2].split(" x ")[1])
                self.abmessung_gesamt = f"{length} x {width} x {height} mm"
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

    def handle_yolo_frames(self, data):
        """Verarbeitet YOLO-Frame-Daten"""
        self.annotierte_frames = data
        for idx, frame in enumerate(data):
            if frame is not None:
                print(f"Bild {idx}: Frame mit Bounding Boxen erhalten")
            else:
                print(f"Bild {idx}: Kein Frame vorhanden")

    def handle_barcode_data(self, data):
        """Verarbeitet Barcode-Daten"""
        idx = data.get("index", -1)
        found = data.get("found", False)
        value = data.get("value", None)
        b_type = data.get("type", None)
        error = data.get("error", None)

        if found:
            print(f"✅ Barcode erkannt in Bild {idx}: Wert='{value}', Typ='{b_type}'")
            self.barcode_type = b_type
            self.barcode = value
        elif error:
            print(f"❌ Barcode Fehler in Bild {idx}: {error}")
            self.barcode_type = "Undefiniert"
            self.barcode = "Undefiniert"
        else:
            print(f"❌ Kein Barcode in Bild {idx}")
            self.barcode_type = "Undefiniert"
            self.barcode = "Undefiniert"

    def handle_weight_data(self, data):
        """Verarbeitet Gewichtsdaten"""
        self.gewicht = data
        print(f"Gewicht: {data}")

    def check_all_tasks_complete(self):
        """Prüft ob alle Tasks abgeschlossen sind und aktualisiert die GUI"""
        abmessung_ready = hasattr(self, "abmessung_gesamt") and self.abmessung_gesamt != "Undefiniert"
        barcode_ready = hasattr(self, "barcode") and self.barcode != "Undefiniert"
        gewicht_ready = hasattr(self, "gewicht") and self.gewicht not in ["Undefiniert", None]

        if abmessung_ready and barcode_ready and gewicht_ready:
            # Finale Bilder vorbereiten
            for i in range(4):
                if self.keep[i] and self.images[i] is not None:
                    self.final_images[i] = self.images[i].copy()
                else:
                    self.final_images[i] = None

            # GUI aktualisieren
            self.reload_pages_for_results()

    def reload_pages_for_results(self):
        """Lädt die Seiten neu um Ergebnisse anzuzeigen"""
        current_index = self.stack.currentIndex()
        
        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        self.load_pages()
        self.stack.setCurrentIndex(min(2, self.stack.count() - 1))
        self.update_buttons()

    def keyPressEvent(self, event):
        """Behandelt Tastatureingaben"""
        if event.key() == Qt.Key.Key_Left:
            self.go_back()
        elif event.key() == Qt.Key.Key_Right:
            self.go_next()


def main():
    """Hauptfunktion der Anwendung"""
    app = QApplication(sys.argv)
    
    # Exception Handling
    def excepthook(exc_type, exc_value, exc_tb):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"Unbehandelte Exception:\n{tb}")
        QMessageBox.critical(None, "Fehler", f"Ein Fehler ist aufgetreten:\n{str(exc_value)}")
    
    sys.excepthook = excepthook
    
    try:
        window = FullscreenApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fehler beim Starten der Anwendung: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()