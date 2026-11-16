import sys
import os

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QRadioButton, QLineEdit,
    QComboBox, QFrame, QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #181818;
    color: #e0e0e0;
    font-family: Segoe UI, sans-serif;
    font-size: 13px;
}
QLabel, QRadioButton, QSlider {
    background-color: transparent;
    border: none;
}
QRadioButton::indicator {
    width: 12px;
    height: 12px;
    border-radius: 8px;
}
QRadioButton::indicator:unchecked {
    border: 1px solid #666666;
    background-color: transparent;
}
QRadioButton::indicator:checked {
    border: 1px solid #c58af9;
    background-color: #c58af9;
}
QWidget#Sidebar {
    background-color: #202020;
    border-right: 1px solid #333333;
}
QWidget#TopBar {
    background-color: #202020;
    border-bottom: 1px solid #333333;
}
QLabel#SectionTitle {
    color: #888888;
    font-weight: bold;
    font-size: 11px;
}
QLabel#Canvas {
    background-color: #121212;
    color: #666666;
}
QPushButton {
    background-color: #2c2c2c;
    border: 1px solid #444444;
    padding: 7px;
    border-radius: 10px;
}
QPushButton:hover {
    background-color: #383838;
}
QPushButton#PrimaryBtn {
    background-color: #3a3a3a;
    border: 1px solid #666666;
    font-weight: bold;
}
QLineEdit, QComboBox {
    background-color: #141414;
    border: 1px solid #444444;
    padding: 5px;
    border-radius: 3px;
}
QFrame[frameShape="4"] { /* HLine separator */
    color: #333333;
}
QStatusBar {
    background-color: #202020;
    border-top: 1px solid #333333;
    color: #888888;
    font-size: 11px;
}
"""


def create_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    return line

def create_logo_pixmap(size=26):
    logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "img", "logo.png"
    )
    if os.path.exists(logo_path):
        return QPixmap(logo_path).scaled(
            size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    return QPixmap()


class ObjectRemoverWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FotoSzop")
        self.resize(1100, 700)
        self.setStyleSheet(DARK_QSS)

        # logo
        logo_pixmap = create_logo_pixmap(26)
        if not logo_pixmap.isNull():
            self.setWindowIcon(QIcon(logo_pixmap))
            logo_label = QLabel()
            logo_label.setPixmap(logo_pixmap)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- TOP BAR ---
        top_bar = QWidget()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        #title_label = QLabel("FotoSzop")
        #title_label.setStyleSheet("font-weight: bold; letter-spacing: 1px;")
        #top_layout.addWidget(title_label)
        top_layout.addStretch()
        root_layout.addWidget(top_bar)

        # --- MAIN WORKSPACE (SIDEBAR + CANVAS) ---
        workspace_layout = QHBoxLayout()
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        # Left Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(15, 15, 15, 15)
        side_layout.setSpacing(12)

        # START
        lbl_import = QLabel("START")
        lbl_import.setObjectName("SectionTitle")
        side_layout.addWidget(lbl_import)

        self.btn_load = QPushButton("Choose a file")
        self.btn_load.clicked.connect(self.on_load_image)
        side_layout.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save result")
        side_layout.addWidget(self.btn_save)

        side_layout.addWidget(create_separator())

        # SELECT
        lbl_select = QLabel("SELECT")
        lbl_select.setObjectName("SectionTitle")
        side_layout.addWidget(lbl_select)

        self.rb_brush = QRadioButton("Brush")
        self.rb_brush.setChecked(True)
        self.rb_wand = QRadioButton("Magic Wand")
        self.rb_livewire = QRadioButton("Live-Wire")
        side_layout.addWidget(self.rb_brush)
        side_layout.addWidget(self.rb_wand)
        side_layout.addWidget(self.rb_livewire)

        slider_header = QHBoxLayout()
        slider_header.addWidget(QLabel("Size"))
        self.lbl_slider_val = QLabel("30 px")
        self.lbl_slider_val.setAlignment(Qt.AlignRight)
        slider_header.addWidget(self.lbl_slider_val)
        side_layout.addLayout(slider_header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(lambda v: self.lbl_slider_val.setText(f"{v} px"))
        side_layout.addWidget(self.slider)

        self.btn_clear = QPushButton("Clear selection")
        side_layout.addWidget(self.btn_clear)

        side_layout.addWidget(create_separator())

        # auto clean
        lbl_clean = QLabel("AUTO CLEAN")
        lbl_clean.setObjectName("SectionTitle")
        side_layout.addWidget(lbl_clean)

        lbl_inprogress = QLabel("In progress...")
        #lbl_inprogress.setObjectName("SectionTitle")
        side_layout.addWidget(lbl_inprogress)

        # self.input_prompt = QLineEdit()
        # self.input_prompt.setPlaceholderText("Your prompt (e.g. red car)...")
        # side_layout.addWidget(self.input_prompt)

        # self.btn_detect_ai = QPushButton("Detect")
        # side_layout.addWidget(self.btn_detect_ai)

        side_layout.addWidget(create_separator())

        # ALGORITHM
        lbl_algo = QLabel("ALGORITHM")
        lbl_algo.setObjectName("SectionTitle")
        side_layout.addWidget(lbl_algo)

        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["Criminisi", "Telea"])
        side_layout.addWidget(self.combo_algo)

        self.btn_clean = QPushButton("Clean selection")
        self.btn_clean.setObjectName("PrimaryBtn")
        side_layout.addWidget(self.btn_clean)

        side_layout.addStretch()
        workspace_layout.addWidget(sidebar)

        # Right Canvas
        self.canvas = QLabel("NO IMAGE")
        self.canvas.setObjectName("Canvas")
        self.canvas.setAlignment(Qt.AlignCenter)
        workspace_layout.addWidget(self.canvas, stretch=1)

        root_layout.addLayout(workspace_layout, stretch=1)

        # --- STATUS BAR ---
        self.statusBar().showMessage("NO IMAGE  |  NOTHING SELECTED")

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self.statusBar().showMessage(f"LOADED: {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectRemoverWindow()
    window.showMaximized()
    sys.exit(app.exec())