import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, 
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie
from PyQt6.QtCore import Qt, QSize, QTimer

class FullscreenApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D-Scanner")
        self.showFullScreen()

        #Erst-Anpassung------------------------------------------------------
        self.language = "de"  # oder "it" / "en" standartmäßig
        self.Explorer_Structure = r"C:\Users\grane\Felix_Schule\Diplomarbeit\Interface_GUI\GUI_Anzeige"

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


        self.load_pages(self.language)
        self.update_buttons()


    def set_language(self, language):
        current_index = self.stack.currentIndex()

        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        self.load_pages(language)

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
                }
            """)
            btn.clicked.connect(lambda _, lang=language_code: self.set_language(lang))
            return btn
    
    def load_pages(self, language):
        tiefe_C20 = 20
        breite_C20 = 20
        höhe_C20 = 20
        gewicht_artikel = 5

        if language == "de":
            self.add_page("Startseite", 
                        [("title", "3D-Scanner Interface"), "Interface um den 3D-Scanner zu bedienen", "'Text'",
                         [("button", "Scan Starten", self.go_next),("button","Auf USB-Stick laden")]])
            self.add_page("Kamera-Übersicht", 
                        [[("image","iso_Bild"), ("image","top_Bild")], [("image","right_Bild"), ("image","behind_Bild")], 
                        f"Abmessungen: {tiefe_C20}mm x {breite_C20}mm x {höhe_C20}mm", f"Gewicht: {gewicht_artikel}kg"])
            self.add_page("Speicher Option", 
                        [("image", "barcode"), ("input", "Ausgewerteter Barcode", "dekodierter-Barcode"),("input", "Barcode-Typ:", "EAN-CODE-TYP"), 
                        [("button", "SAP-Eintrag"), ("button","Lokal speichern")]])
        
        elif language == "it":
            self.add_page("Pagina Iniziale",
                        [("title", "3D-Scanner Interface"), "Interfaccia per gestire lo scanner 3D", "'Testo'", 
                         [("button","Scan start", self.go_next),("button","Carica su USB")]])
            self.add_page("Panoramica Fotocamera",
                        [[("image","iso_Bild"), ("image","top_Bild")], [("image","right_Bild"), ("image","behind_Bild")], 
                        f"Dimensioni: {tiefe_C20} x {breite_C20} x {höhe_C20}", f"Peso: {gewicht_artikel}kg"])
            self.add_page("Opzioni di Memorizzazione",
                        [("image", "barcode"), ("input", "Barcode:", "decodificato-Barcode"),("input", "Tipo di barcode:", "EAN-CODE-TYP"),
                        [("button", "SAP Entry"), ("button","Salva localmente")]])

        else:  # englisch
            self.add_page("Home",
                        [("title", "3D Scanner Interface"), "Interface to operate the 3D scanner", "'Text'", 
                         [("button", "Start scan",self.go_next), ("button", "Load to USB")]])
            self.add_page("Camera Overview",
                        [[("image", "iso_Bild"), ("image", "top_Bild")], [("image", "right_Bild"), ("image", "behind_Bild")],
                        f"Dimensions: {tiefe_C20}mm x {breite_C20}mm x {höhe_C20}mm", f"Weight: {gewicht_artikel} kg"])
            self.add_page("Storage Options",
                        [("image", "barcode"), ("input", "Barcode:", "decodierter-Barcode"),("input", "Barcode-Typ:", "EAN-CODE-TYP"), 
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
            }
            QLabel {
                font-size: 20px;
                color: #2c3e50; /* explizite Schriftfarbe */
            }
        """)
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
            }
            QLabel {
                font-size: 18px;
                color: #2c3e50;
            }
            QLineEdit {
                font-size: 20px;
                color: #2c3e50;
                background: transparent;
                border: none;
                border-bottom: 2px solid #2c3e50;
            }
        """)

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

        # Sprachauswahl-Buttons
        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it")
        btn_en = self.create_flag_button("en.png", "en")

        # In die Taskbar einfügen
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
        
        if idx == 1:  # Seite 2 (Index 1)
            reply = QMessageBox.question(
                self,
                "Datenverlust bestätigen",
                "Möchten Sie wirklich zurück zur Startseite? Alle erfassten Daten gehen verloren.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )

            if reply != QMessageBox.StandardButton.Ok:
                return  # Abbrechen, keine Änderung
    
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self.update_buttons()

    def go_next(self):
        idx = self.stack.currentIndex()
        if idx < self.stack.count() - 1:
            if idx == 0: #Seite 1 -> 2
                self.loading_dialog = QDialog(self)
                self.loading_dialog.setWindowTitle("Ladevorgang")
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

                def finish_loading():
                    if self.loading_dialog.isVisible():
                        self.loading_dialog.accept()  # Lade-Dialog schließen
                        self.stack.setCurrentIndex(idx + 1)
                        self.update_buttons()

                        QMessageBox.information(
                            self,
                            "Scan abgeschlossen",
                            "Der Scan war erfolgreich!\nDie Daten stehen nun zur Verfügung."
                        )

                QTimer.singleShot(6000, finish_loading) 
                #Barcode-Decodierer Aufrufen und verarbeiten
                #Abmessungen der Bilder durchführen über die Kameras
                #Bilder mit Abmessungen gestallten umrandung ziehen

                def cancel_loading():
                    self.loading_dialog.reject()
                    self.stack.setCurrentIndex(0)  # Zurück zur Startseite
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
 