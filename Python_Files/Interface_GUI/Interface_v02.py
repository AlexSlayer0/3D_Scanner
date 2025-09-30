import sys
import os
import shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QHBoxLayout, QFrame, QFileDialog, QLineEdit, QMessageBox,
    QSizePolicy, QGridLayout, QScrollArea
)
from PyQt6.QtGui import QIcon, QPixmap, QFont
from PyQt6.QtCore import Qt, QSize

# ----------------- Konfiguration -----------------
BASE_PATH = os.path.abspath(os.path.join("Felix_Schule", "Diplomarbeit", "Interface_GUI"))
IN_PROGRESS = os.path.join(BASE_PATH, "In_Bearbeitung")
DONE = os.path.join(BASE_PATH, "Bearbeitet")

# Erwartete Bild-Dateinamen
IMAGE_FILES = ["Bild_Iso.png", "Bild_Top.png", "Bild_Right.png", "Bild_Behind.png"]
BARCODE_FILENAME = "barcode.png"

# Flaggen-Icon-Größe
ICON_W, ICON_H = 64, 64
# -------------------------------------------------

# Ordner erstellen
os.makedirs(IN_PROGRESS, exist_ok=True)
os.makedirs(DONE, exist_ok=True)


class SelectAllLineEdit(QLineEdit):
    def __init__(self, text="", grey_initial=True):
        super().__init__(text)
        self.grey_initial = grey_initial
        self._first_focus = True
        self.update_style(grey_initial)

    def update_style(self, grey):
        if grey:
            self.setStyleSheet("QLineEdit { background-color: #f0f0f0; color: #5a5a5a; border: 1px solid #d0d0d0; padding: 8px; }")
        else:
            self.setStyleSheet("QLineEdit { background-color: white; color: black; border: 1px solid #bdbdbd; padding: 8px; }")

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._first_focus and self.text():
            self.selectAll()
        self._first_focus = False
        self.update_style(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D-Scanner")
        self.setGeometry(50, 50, 1200, 800)

        # Flaggen-Icons
        self.flag_paths = {
            "de": os.path.join("flags", "de.png"),
            "en": os.path.join("flags", "gb.png"),
            "it": os.path.join("flags", "it.png"),
        }

        # Pfade
        self.base_path = BASE_PATH
        self.in_progress = IN_PROGRESS
        self.done = DONE

        # Seite: 0 = Start, 1 = Bildansicht, 2 = SAP
        self.page = 0
        self.lang = "de"
        
        # Erweiterte Sprachdaten
        self.lang_map = {
            "de": {
                "scan": "🔍 Scan starten",
                "load": "📁 Pfad/USB laden",
                "back": "Zurück",
                "next": "Weiter",
                "finish": "Abschließen / Speichern",
                "title": "3D-Scanner Interface",
                "path_label": "<b>In_Bearbeitung Pfad:</b> ",
                "barcode_label": "Decodierter Code:",
                "type_label": "Produkttyp:",
                "product_placeholder": "Produkttyp (z. B. EAN13)",
            },
            "en": {
                "scan": "🔍 Start Scan",
                "load": "📁 Load Path / USB",
                "back": "Back",
                "next": "Next",
                "finish": "Finish / Save",
                "title": "3D-Scanner Interface",
                "path_label": "<b>In Progress Path:</b> ",
                "barcode_label": "Decoded Code:",
                "type_label": "Product Type:",
                "product_placeholder": "Product type (e.g. EAN13)",
            }
        }

        # UI Initialisierung
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        main_widget.setLayout(self.main_layout)

        # Bildschirmgröße
        screen = QApplication.primaryScreen()
        if screen:
            s = screen.size()
            self.screen_w = s.width()
            self.screen_h = s.height()
        else:
            self.screen_w = 1366
            self.screen_h = 768

        # Top bar
        self._build_top_bar()

        # Content area
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(18, 18, 18, 18)
        self.content_layout.setSpacing(12)
        self.content_frame.setLayout(self.content_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.content_frame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.main_layout.addWidget(self.scroll)

        # Navigation footer
        self.nav_frame = QFrame()
        self.nav_layout = QHBoxLayout()
        nav_height = max(96, int(self.screen_h * 0.12))
        self.nav_frame.setFixedHeight(nav_height)
        self.nav_layout.setContentsMargins(18, 8, 18, 12)
        self.nav_frame.setLayout(self.nav_layout)
        self.main_layout.addWidget(self.nav_frame)

        # Navigationsbuttons
        self.btn_nav_back = QPushButton()
        self.btn_nav_back.setMinimumHeight(max(64, int(self.screen_h * 0.08)))
        self.btn_nav_back.clicked.connect(self.on_back)
        self.btn_nav_back.setStyleSheet("QPushButton { background-color: #f39c12; color: white; border-radius: 10px; font-size: 18px; } QPushButton:hover { background-color: #e07b09; }")

        self.btn_nav_next = QPushButton()
        self.btn_nav_next.setMinimumHeight(max(64, int(self.screen_h * 0.08)))
        self.btn_nav_next.clicked.connect(self.on_next)
        self.btn_nav_next.setStyleSheet("QPushButton { background-color: #27ae60; color: white; border-radius: 10px; font-size: 18px; } QPushButton:hover { background-color: #1e8449; }")

        # Styling
        self._apply_global_styles()
        self.show_start_page()

    def _build_top_bar(self):
        top_bar = QFrame()
        top_bar.setObjectName("topbar")
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(12, 6, 12, 6)
        top_layout.setSpacing(8)
        top_bar.setLayout(top_layout)

        title = QLabel("3D-Scanner")
        title.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top_layout.addWidget(title)

        # Flaggen buttons
        for code in ("de", "en", "it"):
            btn = QPushButton()
            path = self.flag_paths.get(code, "")
            if os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    scaled = pix.scaled(ICON_W, ICON_H, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    btn.setIcon(QIcon(scaled))
                    btn.setIconSize(QSize(scaled.width(), scaled.height()))
                else:
                    btn.setText(code.upper())
            else:
                btn.setText(code.upper())
            
            btn.setFixedSize(ICON_W + 14, ICON_H + 14)
            btn.setProperty("lang", code)
            btn.clicked.connect(lambda checked, c=code: self.on_change_lang(c))
            btn.setStyleSheet("QPushButton { border: none; background: transparent; } QPushButton:hover { background: rgba(0,0,0,0.04); border-radius: 6px; }")
            top_layout.addWidget(btn)

        self.main_layout.addWidget(top_bar)

    def on_change_lang(self, code):
        if code == "it":
            code = "en"
        self.lang = code
        self._apply_language_to_ui()

    def _apply_language_to_ui(self):
        labels = self.lang_map.get(self.lang, self.lang_map["en"])
        self.btn_nav_back.setText(labels["back"])
        self.btn_nav_next.setText(labels["next"] if self.page != 2 else labels["finish"])
        
        if self.page == 0:
            self.show_start_page()
        elif self.page == 1:
            self.show_image_page()
        elif self.page == 2:
            self.show_sap_page()

    def _apply_global_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #eef3f7; font-family: Arial; }
            QFrame#card { background: white; border-radius: 10px; padding: 14px; }
            QLabel.title { color: #2c3e50; }
            QLabel.small { color: #6b6f75; font-size: 12px; }
            QPushButton.primary { background-color: #2b78d4; color: white; border-radius: 10px; padding: 12px; font-size: 18px; }
        """)

    def clear_content(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_start_page(self):
        self.page = 0
        self.clear_content()
        self.nav_frame.hide()

        labels = self.lang_map.get(self.lang, self.lang_map["en"])

        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout()
        v.setContentsMargins(22, 22, 22, 22)
        v.setSpacing(22)
        card.setLayout(v)

        title = QLabel(labels["title"])
        title.setFont(QFont("Arial", 30, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)

        hbtn = QHBoxLayout()
        btn_scan = QPushButton(labels["scan"])
        btn_scan.setMinimumHeight(max(160, int(self.screen_h * 0.18)))
        btn_scan.setFont(QFont("Arial", 20))
        btn_scan.setStyleSheet("QPushButton { background-color: #2b78d4; color: white; border-radius: 14px; } QPushButton:hover { background-color: #2464b0; }")
        btn_scan.clicked.connect(self.on_scan_start)
        hbtn.addWidget(btn_scan, 3)

        btn_load = QPushButton(labels["load"])
        btn_load.setMinimumHeight(max(120, int(self.screen_h * 0.12)))
        btn_load.setFont(QFont("Arial", 18))
        btn_load.setStyleSheet("QPushButton { background-color: #6c7ae0; color: white; border-radius: 14px; } QPushButton:hover { background-color: #5866d0; }")
        btn_load.clicked.connect(self.on_load_path)
        hbtn.addWidget(btn_load, 1)
        v.addLayout(hbtn)

        path_label = QLabel(f"{labels['path_label']}{self.in_progress}")
        path_label.setWordWrap(True)
        path_label.setStyleSheet("color: #4b5563; font-size: 14px;")
        v.addWidget(path_label)

        self.content_layout.addWidget(card)

    def on_load_path(self):
        dlg = QFileDialog(self, "Ordner wählen")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        if dlg.exec():
            chosen = dlg.selectedFiles()[0]
            self.in_progress = chosen
            self.done = os.path.join(os.path.dirname(chosen), "Bearbeitet")
            os.makedirs(self.done, exist_ok=True)
            self.show_start_page()

    def on_scan_start(self):
        self.show_image_page()

    def show_image_page(self):
        self.page = 1
        self.clear_content()
        self.nav_frame.show()
        labels = self.lang_map.get(self.lang, self.lang_map["en"])
        self._setup_nav_buttons(show_back=True, show_next=True)

        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout()
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setSpacing(18)
        card.setLayout(grid)

        keys = IMAGE_FILES
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        preview_w = int((self.screen_w - 240) / 2)
        preview_h = int((self.screen_h - 320) / 2)

        for fname, pos in zip(keys, positions):
            full = os.path.join(self.in_progress, fname)
            panel = QFrame()
            panel.setStyleSheet("QFrame { background: #ffffff; border-radius: 10px; }")
            v = QVBoxLayout()
            v.setContentsMargins(12, 12, 12, 12)
            v.setSpacing(10)
            panel.setLayout(v)

            title = QLabel(fname.replace(".png", ""))
            title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            v.addWidget(title)

            label_img = QLabel()
            label_img.setMaximumSize(preview_w, preview_h)
            label_img.setMinimumSize(300, 200)
            label_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if os.path.exists(full):
                pix = QPixmap(full)
                if not pix.isNull():
                    scaled = pix.scaled(label_img.width(), label_img.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    label_img.setPixmap(scaled)
                else:
                    label_img.setText("Ungültiges Bild")
            else:
                label_img.setText("Nicht gefunden")
            
            v.addWidget(label_img)
            grid.addWidget(panel, pos[0], pos[1])

        self.content_layout.addWidget(card)

    def show_sap_page(self):
        self.page = 2
        self.clear_content()
        self.nav_frame.show()
        labels = self.lang_map.get(self.lang, self.lang_map["en"])
        self._setup_nav_buttons(show_back=True, show_next=True)

        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout()
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(12)
        card.setLayout(v)

        barcode_path = os.path.join(self.in_progress, BARCODE_FILENAME)
        label_bar = QLabel()
        label_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_bar.setMaximumHeight(int(self.screen_h * 0.32))
        
        if os.path.exists(barcode_path):
            pix = QPixmap(barcode_path)
            if not pix.isNull():
                scaled = pix.scaledToHeight(int(self.screen_h * 0.28), Qt.TransformationMode.SmoothTransformation)
                label_bar.setPixmap(scaled)
            else:
                label_bar.setText("Barcode ungültig")
        else:
            label_bar.setText("Barcode nicht gefunden")
        
        v.addWidget(label_bar)

        decoded = "1234567890123"  # Beispielwert
        self.edit_code = SelectAllLineEdit(decoded, grey_initial=True)
        self.edit_code.setPlaceholderText("Decodierter Code")
        v.addWidget(QLabel(labels["barcode_label"]))
        v.addWidget(self.edit_code)

        self.edit_type = SelectAllLineEdit("", grey_initial=True)
        self.edit_type.setPlaceholderText(labels["product_placeholder"])
        v.addWidget(QLabel(labels["type_label"]))
        v.addWidget(self.edit_type)

        self.edit_code.textChanged.connect(self._update_product_type_from_code)
        self.content_layout.addWidget(card)

    def _update_product_type_from_code(self, txt):
        if len(txt) == 13 and txt.isdigit():
            self.edit_type.setText("EAN13")
        else:
            self.edit_type.clear()

    def _setup_nav_buttons(self, show_back=False, show_next=False):
        while self.nav_layout.count():
            item = self.nav_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if show_back:
            self.nav_layout.addWidget(self.btn_nav_back)
        else:
            self.nav_layout.addStretch()

        if show_next:
            self.nav_layout.addWidget(self.btn_nav_next)

    def on_back(self):
        if self.page == 1:
            self.show_start_page()
        elif self.page == 2:
            self.show_image_page()

    def on_next(self):
        if self.page == 1:
            self.show_sap_page()
        elif self.page == 2:
            self._save_and_finish()

    def _save_and_finish(self):
        try:
            for fname in IMAGE_FILES + [BARCODE_FILENAME]:
                src = os.path.join(self.in_progress, fname)
                dst = os.path.join(self.done, fname)
                if os.path.exists(src):
                    shutil.move(src, dst)
            
            with open(os.path.join(self.done, "metadata.txt"), "w") as f:
                f.write(f"Code: {self.edit_code.text()}\nType: {self.edit_type.text()}\n")
            
            QMessageBox.information(self, "Erfolg", "Daten erfolgreich gespeichert")
            self.show_start_page()
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Speichern fehlgeschlagen: {str(e)}")

    def closeEvent(self, event):
        self.showMinimized()
        event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())