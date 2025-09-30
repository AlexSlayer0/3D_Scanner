import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QHBoxLayout, QTreeView, QDialog, QFrame
)
from PyQt6.QtGui import QIcon, QFileSystemModel, QFont
from PyQt6.QtCore import Qt


class ClickableLabel(QLabel):
    """Ein Label, das klickbar ist und Auswahlzustand zeigt"""
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(100)
        self.setStyleSheet("""
            QLabel {
                background-color: #bdc3c7;
                border: 2px dashed #7f8c8d;
                border-radius: 10px;
                font-size: 16px;
                color: #2c3e50;
            }
        """)
        self.selected = False

    def mousePressEvent(self, event):
        self.selected = not self.selected
        if self.selected:
            self.setStyleSheet("""
                QLabel {
                    background-color: #2ecc71;  /* grün bei Auswahl */
                    border: 2px solid #27ae60;
                    border-radius: 10px;
                    font-size: 16px;
                    color: white;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: #bdc3c7;
                    border: 2px dashed #7f8c8d;
                    border-radius: 10px;
                    font-size: 16px;
                    color: #2c3e50;
                }
            """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D-Scanner Interface")
        self.setGeometry(100, 100, 900, 600)

        # Hauptlayout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        main_widget.setLayout(self.layout)

        # Spracheinstellungen
        self.langs = {
            "de": {"scan": "🔍 Scan starten", "internet": "🌐 Internet verbinden",
                   "sap": "💼 SAP Zugriff", "ordner": "📂 Ordnerstruktur anzeigen"},
            "en": {"scan": "🔍 Start Scan", "internet": "🌐 Connect Internet",
                   "sap": "💼 SAP Access", "ordner": "📂 Show Folder Structure"},
            "it": {"scan": "🔍 Avvia scansione", "internet": "🌐 Connetti Internet",
                   "sap": "💼 Accesso SAP", "ordner": "📂 Mostra struttura cartelle"}
        }
        self.current_lang = "de"

        # --- Top-Bar ---
        top_bar = QFrame()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 5, 10, 5)
        top_layout.setSpacing(5)
        top_bar.setLayout(top_layout)

        # Titel
        title = QLabel("3D-Scanner Interface")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50;")
        top_layout.addWidget(title)
        top_layout.addStretch()

        # Sprach-Buttons
        btn_de = QPushButton()
        btn_de.setIcon(QIcon("flage_de.png"))
        btn_de.setFixedSize(40, 30)
        btn_de.clicked.connect(lambda: self.change_language("de"))

        btn_en = QPushButton()
        btn_en.setIcon(QIcon("flage_en.png"))
        btn_en.setFixedSize(40, 30)
        btn_en.clicked.connect(lambda: self.change_language("en"))

        btn_it = QPushButton()
        btn_it.setIcon(QIcon("flage_it.png"))
        btn_it.setFixedSize(40, 30)
        btn_it.clicked.connect(lambda: self.change_language("it"))

        top_layout.addWidget(btn_de)
        top_layout.addWidget(btn_en)
        top_layout.addWidget(btn_it)

        self.layout.addWidget(top_bar)

        # --- Content Bereich (wechselbar) ---
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout()
        self.content_frame.setLayout(self.content_layout)
        self.layout.addWidget(self.content_frame)

        # Buttons initial anzeigen
        self.show_main_buttons()

        # Sprache initial setzen
        self.change_language("de")

    def clear_content(self):
        """Löscht alles im Content-Bereich, inkl. Layouts"""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # rekursiv Layouts löschen
                self.clear_layout(item.layout())

    def clear_layout(self, layout):
        """Hilfsfunktion um verschachtelte Layouts zu löschen"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())


    def show_main_buttons(self):
        """Zeigt die Hauptbuttons"""
        self.clear_content()

        self.btn_scan = QPushButton()
        self.btn_scan.clicked.connect(self.show_scan_view)

        self.btn_internet = QPushButton()
        self.btn_sap = QPushButton()
        self.btn_ordner = QPushButton()
        self.btn_ordner.clicked.connect(self.show_folder_structure)

        for btn in [self.btn_scan, self.btn_internet, self.btn_sap, self.btn_ordner]:
            btn.setMinimumHeight(50)
            btn.setFont(QFont("Arial", 14))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border-radius: 10px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #1c5980;
                }
            """)
            self.content_layout.addWidget(btn)

        # Sprache wieder anwenden
        self.change_language(self.current_lang)

    def show_scan_view(self):
        """Zeigt die Scan-Ansicht mit 5 klickbaren Bildern"""
        self.clear_content()

        self.image_labels = []
        for i in range(5):
            label = ClickableLabel(f"Bild {i+1}")
            self.image_labels.append(label)
            self.content_layout.addWidget(label)

        # Buttons zurück und weiter
        btn_layout = QHBoxLayout()
        back_btn = QPushButton("⬅️ Zurück")
        back_btn.setMinimumHeight(40)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #d35400; }
        """)
        back_btn.clicked.connect(self.show_main_buttons)

        forward_btn = QPushButton("✅ Weiter")
        forward_btn.setMinimumHeight(40)
        forward_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #1e8449; }
        """)
        forward_btn.clicked.connect(self.process_selection)

        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(forward_btn)
        self.content_layout.addLayout(btn_layout)

    def process_selection(self):
        selected = [lbl.text() for lbl in self.image_labels if lbl.selected]
        print("Ausgewählte Bilder:", selected)

    def change_language(self, lang):
        self.current_lang = lang
        if hasattr(self, "btn_scan"):
            self.btn_scan.setText(self.langs[lang]["scan"])
            self.btn_internet.setText(self.langs[lang]["internet"])
            self.btn_sap.setText(self.langs[lang]["sap"])
            self.btn_ordner.setText(self.langs[lang]["ordner"])

    def show_folder_structure(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Ordnerstruktur")
        dialog.setGeometry(300, 300, 600, 400)

        layout = QVBoxLayout()
        model = QFileSystemModel()
        model.setRootPath("")
        tree = QTreeView()
        tree.setModel(model)
        layout.addWidget(tree)
        dialog.setLayout(layout)

        dialog.exec()

    def closeEvent(self, event):
        # Schließen verhindern – nur minimieren
        self.showMinimized()
        event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Globales Styling
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget {
            background-color: #ecf0f1;
            font-family: Arial;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
