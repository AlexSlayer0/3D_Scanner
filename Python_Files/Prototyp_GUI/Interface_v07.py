"""=======TODO-Liste v0.6=======
Objekt-Detection muss verbessert werden
Lade-Dialog schöner gestalten
Fehlerbehandlung bei Kamerazugriff
SAP-Integration         (Platzhalter-Button)
USB-Stick Integration   (Platzhalter-Button)
GUI-Design verbessern   (optional)
================================"""

import sys
import cv2
import os
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, 
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal


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

def capture_images(cam_indices=(0, 1, 2)):
    images = []
    for idx in cam_indices:
        img = capture_single(idx)
        images.append(img)  # egal ob Bild oder None
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
            w = Gewichts_Messung.get_weight()  # sollte z.B. 5 zurückgeben
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
        self.language = "de"  # oder "it" / "en" standartmäßig
        self.Explorer_Structure = r"C:\Users\grane\Felix_Schule\Diplomarbeit\Prototyp_GUI\GUI_Anzeige"

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
        current_index = self.stack.currentIndex()

        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        self.load_pages()

        if current_index < self.stack.count():
            self.stack.setCurrentIndex(current_index)
        else:
            self.stack.setCurrentIndex(self.stack.count() - 1)

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def retry_image(self, idx):
        print(f"🔄 Wiederhole Bild {idx+1}")
        self.scan_start = True
        new_img = capture_single(idx)
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

    def load_pages(self):
        if self.abmessung == None:
            self.abmessung = "Undefiniert"
        if self.gewicht == None:
            self.gewicht = "Undefiniert"
        if self.barcode == None:
            self.barcode = "Undefiniert"
            self.barcode_type = "Undefiniert"
            
        if self.language == "de":
            self.add_page("Startseite", 
                    [("title", "3D-Scanner Interface"), "Interface um den 3D-Scanner zu bedienen", "Bitte lege den Artikel der gescannt werden soll in die Box ein",
                    [("button", "Scan Starten", self.go_next),("button","Auf USB-Stick laden")]]) 
            self.add_page("Foto-Auswahl", 
                    [[("ram_image",0), ("ram_image",1)],
                    [("button", "Wiederholen", lambda _, idx=0: self.retry_image(idx)),
                    ("button", "Wiederholen", lambda _, idx=1: self.retry_image(idx))],
                    [("button", "Verwerfen", lambda _, idx=0: self.discard_image(idx)),
                    ("button", "Verwerfen", lambda _, idx=1: self.discard_image(idx))],

                    [("ram_image",2),("ram_image",3)],
                    [("button", "Wiederholen", lambda _, idx=2: self.retry_image(idx)),
                    ("button", "Wiederholen", lambda _, idx=3: self.retry_image(idx))],
                    [("button", "Verwerfen", lambda _, idx=2: self.discard_image(idx)),
                    ("button", "Verwerfen", lambda _, idx=3: self.discard_image(idx))]])
                
            self.add_page("Kamera-Übersicht", 
                    [[("ram_image_final", 0), ("ram_image_final", 1)], [("ram_image_final", 2), ("ram_image_final", 3)], 
                    f"Abmessungen: {self.abmessung}mm", f"Gewicht: {self.gewicht}kg"])
            self.add_page("Speicher Option", 
                    [("image", "barcode"), ("input", "Ausgewerteter Barcode", f"{self.barcode}"),("input", "Barcode-Typ:", f"{self.barcode_type}"), 
                    [("button", "SAP-Eintrag"), ("button","Lokal speichern")]])

        elif self.language == "it":
            self.add_page("Pagina Iniziale",
                    [("title", "3D-Scanner Interface"), "Interfaccia per gestire lo scanner 3D", "Si prega di posizionare l'articolo nella scatola",
                    [("button","Avvia Scan", self.go_next),("button","Carica su USB")]])
            self.add_page("Selezione Foto", 
                    [[("ram_image", 0), ("ram_image", 1)],
                    [("button", "Ripeti", lambda _, idx=0: self.retry_image(idx)),
                    ("button", "Ripeti", lambda _, idx=1: self.retry_image(idx))],
                    [("button", "Scarta", lambda _, idx=0: self.discard_image(idx)),
                    ("button", "Scarta", lambda _, idx=1: self.discard_image(idx))],

                    [("ram_image", 2), ("ram_image", 3)],
                    [("button", "Ripeti", lambda _, idx=2: self.retry_image(idx)),
                    ("button", "Ripeti", lambda _, idx=3: self.retry_image(idx))],
                    [("button", "Scarta", lambda _, idx=2: self.discard_image(idx)),
                    ("button", "Scarta", lambda _, idx=3: self.discard_image(idx))]])
            
            self.add_page("Panoramica Fotocamera",
                    [[("ram_image_final", 0), ("ram_image_final", 1)], [("ram_image_final", 2), ("ram_image_final", 3)], 
                    f"Dimensioni: {self.abmessung}mm", f"Peso: {self.gewicht}kg"])
            self.add_page("Opzioni di Memorizzazione",
                    [("image", "barcode"), ("input", "Barcode:", f"{self.barcode}"),("input", "Tipo di barcode:", f"{self.barcode_type}"),
                    [("button", "SAP Entry"), ("button","Salva localmente")]])

        else:  # englisch
            self.add_page("Home",
                    [("title", "3D Scanner Interface"), "Interface to operate the 3D scanner", "Please place the item in the box",
                    [("button", "Start Scan",self.go_next), ("button", "Load to USB")]])
            self.add_page("Photo Selection", 
                    [[("ram_image", 0), ("ram_image", 1)],
                    [("button", "Retake", lambda _, idx=0: self.retry_image(idx)),
                    ("button", "Retake", lambda _, idx=1: self.retry_image(idx))],
                    [("button", "Discard", lambda _, idx=0: self.discard_image(idx)),
                    ("button", "Discard", lambda _, idx=1: self.discard_image(idx))],

                    [("ram_image", 2), ("ram_image", 3)],
                    [("button", "Retake", lambda _, idx=2: self.retry_image(idx)),
                    ("button", "Retake", lambda _, idx=3: self.retry_image(idx))],
                    [("button", "Discard", lambda _, idx=2: self.discard_image(idx)),
                    ("button", "Discard", lambda _, idx=3: self.discard_image(idx))]])
            
            self.add_page("Camera Overview",
                    [[("ram_image_final", 0), ("ram_image_final", 1)], [("ram_image_final", 2), ("ram_image_final", 3)],
                    f"Dimensions: {self.abmessung}mm", f"Weight: {self.gewicht} kg"])
            self.add_page("Storage Options",
                    [("image", "barcode"), ("input", "Barcode:", f"{self.barcode}"),("input", "Barcode Type:", f"{self.barcode_type}"), 
                    [("button", "SAP Entry"), ("button", "Save Locally")]])


    def make_card(self, text):
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
                font-size: 20px;
                color: #2c3e50; /* explizite Schriftfarbe */
            }""")
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return frame

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

        # Eingabefeld -> dehnt sich automatisch auf volle Breite
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
        if isinstance(item, tuple):
            if item[0] == "button":
                text = item[1]
                btn = QPushButton(text)
                btn.setStyleSheet("font-size: 20px; padding: 10px;")
                
                # Prüfen, ob eine Funktion übergeben wurde
                if len(item) > 2 and callable(item[2]):
                    btn.clicked.connect(item[2])
                else:
                    btn.clicked.connect(lambda _, t=text: print(f"Button {t} gedrückt"))
                
                return btn

            elif item[0] == "image":
                label = QLabel()
                base_name = item[1]
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
            
            elif item[0] == "ram_image":
                idx = item[1]
                label = QLabel()
                self.image_labels[idx] = label
                if self.images[idx] is not None:
                    pixmap = self.convert_to_pixmap(self.images[idx])
                    label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                return label
            
            elif item[0] == "ram_image_final":
                idx = item[1]
                label = QLabel()
                self.final_image_labels[idx] = label  # Referenz speichern
                if self.final_images[idx] is not None:
                    label.setPixmap(self.convert_to_pixmap(self.final_images[idx]))
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                return label

            elif item[0] == "title":
                label = QLabel(item[1])
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dedede;")
                return label
            
            elif item[0] == "input":
                label_text = item[1] if len(item) > 1 else ""
                placeholder = item[2] if len(item) > 2 else ""
                preset_text = item[3] if len(item) > 3 else ""
                return self.make_card_with_input(label_text, preset_text, placeholder)

        return self.make_card(str(item))

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
        title_layout.addWidget(title_label, stretch=1)  # stretch=1 -> füllt linken Platz

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
            if isinstance(item, list):  # Reihe mit mehreren Elementen
                row_layout = QHBoxLayout()
                row_layout.setSpacing(12)
                for sub in item:
                    widget = self._make_widget(sub)
                    row_layout.addWidget(widget)
                layout.addLayout(row_layout)
            else:  # Einzelnes Element
                widget = self._make_widget(item)
                layout.addWidget(widget)

        layout.addStretch()
        scroll.setWidget(content)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)

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
                return  # Abbrechen, keine Änderung
    
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

            for i, label in enumerate(self.image_labels):
                img = capture_single(i)
                if img is not None:
                    self.images[i] = img
                    label.setPixmap(self.convert_to_pixmap(img))

            self.stack.setCurrentIndex(idx + 1)
            self.update_buttons()
            return

        elif idx == 1 and self.scan_start == True:
            self.loading_dialog = QDialog(self)
            self.loading_dialog.setWindowTitle("Ladevorgang der Daten")
            self.loading_dialog.setModal(True)
            self.loading_dialog.setFixedSize(300, 300)

            layout = QVBoxLayout(self.loading_dialog)
            movie = QMovie(os.path.join(self.Explorer_Structure, "loading.gif"))
            gif_label = QLabel()
            gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gif_label.setMovie(movie)
            movie.start()
            layout.addWidget(gif_label)

            cancel_btn = QPushButton("Abbrechen")
            layout.addWidget(cancel_btn)

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
                self.loading_dialog.reject()
                self._counter = 0
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
        self.worker = ParallelWorker(self.images)  # Übergabe der 4 Bilder
        self.worker.output_received.connect(self.handle_output)
        self.worker.finished.connect(lambda: print("Alle Tasks fertig"))
        self.worker.start()


    def handle_output(self, script_name, data):
        # --- Debug Header ---
        print(f"\n========== Debug [{script_name}] ==========")

        if script_name == "Abmessung":
            if isinstance(data, list) and len(data) >= 3:
                self.abmessungen = data  # Rohwerte pro Bild bleiben erhalten

                try:
                    # Berechnung Länge x Breite x Höhe:
                    # Bild 0: Länge, Bild 1: Breite, Bild 2: Höhe
                    length = int(data[0].split(" x ")[0])  # Länge aus Bild 0
                    width  = int(data[1].split(" x ")[0])  # Breite aus Bild 1
                    height = int(data[2].split(" x ")[1])  # Höhe aus Bild 2

                    self.abmessung_gesamt = f"{length} x {width} x {height} mm"
                except Exception as e:
                    print(f"Fehler beim Berechnen der Gesamt-Abmessung: {e}")
                    self.abmessung_gesamt = "Undefiniert"

                print(f"🔹 Gesamt-Abmessung: {self.abmessung_gesamt}")
                self.abmessung = self.abmessung_gesamt
                # Debug: Originalwerte der Bilder
                for idx, dim in enumerate(data):
                    print(f"Bild {idx}: Original = {dim}")

            else:
                self.abmessungen = []
                self.abmessung_gesamt = "Undefiniert"
                print(f"Fehler bei Abmessung: {data}")

        elif script_name == "yolo_frames":
            # Annotierte Frames (optional für GUI)
            self.annotierte_frames = data
            for idx, frame in enumerate(data):
                if frame is not None:
                    print(f"Bild {idx}: Frame mit Bounding Boxen erhalten")
                else:
                    print(f"Bild {idx}: Kein Frame vorhanden")

        elif script_name == "barcode":
            # data = {"index": idx, "found": bool, "value": str, "type": str}
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

            #Muss geändert werden wenn zwei verschiedene barcodes erkannt werden
            self.barcode_type = b_type if found else "Undefiniert"
            self.barcode = value if found else "Undefiniert"

        elif script_name == "weight": # Gewicht
            self.gewicht = data
            print(f"Gewicht: {data}")

        else:
            print(f"Unbekanntes Script '{script_name}': {data}")

        print("========== Ende Debug ==========\n")

        # --- Prüfen, ob alle Tasks fertig sind ---
        abmessung_ready = hasattr(self, "abmessung_gesamt") and self.abmessung_gesamt != "Undefiniert"
        barcode_ready = hasattr(self, "barcode") and any(self.barcode)
        gewicht_ready = hasattr(self, "gewicht") and self.gewicht not in ["Undefiniert", None]

        if abmessung_ready and barcode_ready and gewicht_ready:
            for i in range(4):
                if self.keep[i] and self.images[i] is not None:
                    self.final_images[i] = self.images[i].copy()
                else:
                    self.final_images[i] = None

            # GUI aktualisieren: alle alten Seiten entfernen
            while self.stack.count() > 0:
                widget = self.stack.widget(0)
                self.stack.removeWidget(widget)
                widget.deleteLater()

            # Neue Seiten laden
            self.load_pages()
            self.stack.setCurrentIndex(2)  # Index der Kamera-Übersicht
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
