import sys
import os
import time

import cv2
import numpy as np

# Umożliwia uruchomienie zarówno `python main.py`, jak i `python frontend/main_window.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QRadioButton, QLineEdit,
    QComboBox, QFrame, QFileDialog, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut

from backend.image_manager import ImageManager
from frontend.image_canvas import ImageCanvas, SUPPORTED_EXTENSIONS
from frontend.tools import BrushTool, MagicWandTool, LiveWireTool

OPEN_FILTER = "Images ({});;All files (*)".format(
    " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
)
SAVE_FILTERS = {
    "PNG (*.png)": ".png",
    "JPEG (*.jpg *.jpeg)": ".jpg",
    "WEBP (*.webp)": ".webp",
    "BMP (*.bmp)": ".bmp",
}

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
QPushButton {
    background-color: #2c2c2c;
    border: 1px solid #444444;
    padding: 7px;
    border-radius: 10px;
}
QPushButton:hover {
    background-color: #383838;
}
QPushButton:disabled {
    color: #555555;
    border-color: #333333;
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
QProgressBar {
    background-color: #141414;
    border: 1px solid #444444;
    border-radius: 3px;
    color: #e0e0e0;
    text-align: center;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #c58af9;
    border-radius: 2px;
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

        self.image_manager = ImageManager()
        self.current_path: str | None = None
        self.last_dir = ""
        self._displayed_image = None   # obraz aktualnie pokazany na płótnie (do pomijania zbędnych odświeżeń)
        self._worker = None            # wątek algorytmu rekonstrukcji (gdy działa)
        self._worker_started_at = 0.0

        # Płótno tworzymy na początku - narzędzia potrzebują do niego referencji
        self.canvas = ImageCanvas()

        # Narzędzia zaznaczania (kolejność = kolejność przycisków w panelu)
        self.tools = {
            "brush": BrushTool(self),
            "wand": MagicWandTool(self),
            "livewire": LiveWireTool(self),
        }
        self.active_tool = self.tools["brush"]

        # logo
        logo_pixmap = create_logo_pixmap(26)
        logo_label = QLabel()
        if not logo_pixmap.isNull():
            self.setWindowIcon(QIcon(logo_pixmap))
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
        top_layout.setSpacing(10)
        top_layout.addWidget(logo_label)
        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet("color: #aaaaaa;")
        top_layout.addWidget(self.lbl_title)
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
        self.btn_load.setToolTip("Open an image (Ctrl+O)")
        self.btn_load.clicked.connect(self.on_load_image)
        side_layout.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save result")
        self.btn_save.setToolTip("Save the edited image (Ctrl+S)")
        self.btn_save.clicked.connect(self.on_save_image)
        side_layout.addWidget(self.btn_save)

        history_row = QHBoxLayout()
        history_row.setSpacing(8)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.on_undo)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.on_redo)
        history_row.addWidget(self.btn_undo)
        history_row.addWidget(self.btn_redo)
        side_layout.addLayout(history_row)

        side_layout.addWidget(create_separator())

        # SELECT
        lbl_select = QLabel("SELECT")
        lbl_select.setObjectName("SectionTitle")
        side_layout.addWidget(lbl_select)

        self.rb_brush = QRadioButton("Brush")
        self.rb_brush.setToolTip("Brush (B)")
        self.rb_brush.setChecked(True)
        self.rb_wand = QRadioButton("Magic Wand")
        self.rb_wand.setToolTip("Magic Wand (W)")
        self.rb_livewire = QRadioButton("Live-Wire")
        self.rb_livewire.setToolTip("Live-Wire (L)")
        self.tool_buttons = {
            "brush": self.rb_brush,
            "wand": self.rb_wand,
            "livewire": self.rb_livewire,
        }
        for key, rb in self.tool_buttons.items():
            rb.toggled.connect(lambda checked, k=key: checked and self.set_tool(k))
            side_layout.addWidget(rb)

        slider_header = QHBoxLayout()
        self.lbl_slider_name = QLabel("Size")
        slider_header.addWidget(self.lbl_slider_name)
        self.lbl_slider_val = QLabel("")
        self.lbl_slider_val.setAlignment(Qt.AlignRight)
        slider_header.addWidget(self.lbl_slider_val)
        side_layout.addLayout(slider_header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_changed)
        side_layout.addWidget(self.slider)

        self.lbl_tool_hint = QLabel("")
        self.lbl_tool_hint.setWordWrap(True)
        self.lbl_tool_hint.setStyleSheet("color: #777777; font-size: 11px;")
        side_layout.addWidget(self.lbl_tool_hint)

        self.btn_clear = QPushButton("Clear selection")
        self.btn_clear.clicked.connect(self.on_clear_selection)
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
        self.btn_clean.clicked.connect(self.on_clean_selection)
        side_layout.addWidget(self.btn_clean)

        side_layout.addStretch()
        workspace_layout.addWidget(sidebar)

        # Right Canvas
        self.canvas.file_dropped.connect(self.load_image_from_path)
        self.canvas.zoom_changed.connect(self._update_zoom_label)
        self.canvas.cursor_moved.connect(self._update_cursor_label)
        workspace_layout.addWidget(self.canvas, stretch=1)

        root_layout.addLayout(workspace_layout, stretch=1)

        # --- STATUS BAR ---
        # Lewa strona: komunikaty tymczasowe; prawa strona: stałe informacje
        self.lbl_file_info = QLabel("NO IMAGE")
        self.lbl_selection_info = QLabel("NOTHING SELECTED")
        self.lbl_cursor_info = QLabel("")
        self.lbl_zoom_info = QLabel("")
        for lbl in (self.lbl_file_info, self.lbl_selection_info,
                    self.lbl_cursor_info, self.lbl_zoom_info):
            lbl.setStyleSheet("padding: 0 8px;")
            self.statusBar().addPermanentWidget(lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(180, 14)
        self.progress_bar.setVisible(False)
        self.statusBar().insertPermanentWidget(0, self.progress_bar)

        self._create_shortcuts()
        self.set_tool("brush")
        self._update_ui_state()

    # --- SKRÓTY KLAWISZOWE ---

    def _create_shortcuts(self):
        bindings = [
            (QKeySequence.Open, self.on_load_image),
            (QKeySequence.Save, self.on_save_image),
            (QKeySequence("Ctrl+0"), self.canvas.fit_to_window),
            (QKeySequence("Ctrl+1"), self.canvas.set_actual_size),
            (QKeySequence.ZoomIn, self.canvas.zoom_in),
            (QKeySequence("Ctrl+="), self.canvas.zoom_in),
            (QKeySequence.ZoomOut, self.canvas.zoom_out),
            (QKeySequence("Ctrl+D"), self.on_clear_selection),
            (QKeySequence.Undo, self.on_undo),
            (QKeySequence("Ctrl+Y"), self.on_redo),
            (QKeySequence("Ctrl+Shift+Z"), self.on_redo),
            (QKeySequence("B"), lambda: self.rb_brush.setChecked(True)),
            (QKeySequence("W"), lambda: self.rb_wand.setChecked(True)),
            (QKeySequence("L"), lambda: self.rb_livewire.setChecked(True)),
            (QKeySequence("["), lambda: self._nudge_slider(-1)),
            (QKeySequence("]"), lambda: self._nudge_slider(+1)),
        ]
        # Skróty widoku (zoom) działają zawsze; pozostałe są wyłączane, gdy działa algorytm
        view_slots = (self.canvas.fit_to_window, self.canvas.set_actual_size,
                      self.canvas.zoom_in, self.canvas.zoom_out)
        self._edit_shortcuts = []
        for seq, slot in bindings:
            sc = QShortcut(seq, self, activated=slot)
            if slot not in view_slots:
                self._edit_shortcuts.append(sc)

    # --- NARZĘDZIA ---

    def set_tool(self, key: str):
        tool = self.tools[key]
        if tool is not self.active_tool:
            self.active_tool.cancel()
        self.active_tool = tool
        if not self.tool_buttons[key].isChecked():
            self.tool_buttons[key].setChecked(True)   # np. przy wywołaniu ze skrótu

        uses_slider = tool.slider_label is not None
        for w in (self.lbl_slider_name, self.lbl_slider_val, self.slider):
            w.setVisible(uses_slider)
        if uses_slider:
            self.lbl_slider_name.setText(tool.slider_label)
            self.slider.blockSignals(True)
            self.slider.setRange(*tool.slider_range)
            self.slider.setValue(tool.value)
            self.slider.blockSignals(False)
            self._on_slider_changed(tool.value)

        self.lbl_tool_hint.setText(tool.hint)
        self.canvas.set_tool(tool)

    def _on_slider_changed(self, value: int):
        self.active_tool.value = value
        self.lbl_slider_val.setText(f"{value}{self.active_tool.slider_suffix}")
        self.canvas.update()   # odśwież podgląd pędzla

    def _nudge_slider(self, direction: int):
        """[ / ] - zmiana rozmiaru pędzla / tolerancji (krok rośnie z wartością, jak w Photoshopie)."""
        if not self.slider.isVisible():
            return
        step = max(1, self.slider.value() // 10)
        self.slider.setValue(self.slider.value() + direction * step)

    # Interfejs dla narzędzi (patrz frontend/tools.py)

    def begin_edit(self):
        self.image_manager.push_history()

    def mask_edited(self, rect=None):
        mask = self.image_manager.mask
        if rect is None:
            self.canvas.set_mask(mask)
        else:
            self.canvas.update_mask_region(mask, *rect)
        self._update_ui_state()

    # --- WCZYTYWANIE / ZAPIS ---

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", self.last_dir, OPEN_FILTER)
        if path:
            self.load_image_from_path(path)

    def load_image_from_path(self, path: str):
        if self.is_busy:   # np. plik upuszczony na płótno w trakcie działania algorytmu
            return
        try:
            self.image_manager.load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open image", f"Could not open:\n{path}\n\n{exc}")
            return

        self.active_tool.cancel()
        self.current_path = path
        self.last_dir = os.path.dirname(path)
        self.setWindowTitle(f"FotoSzop - {os.path.basename(path)}")
        self.lbl_title.setText(os.path.basename(path))
        self.refresh_canvas(reset_view=True)
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}", 4000)

    def on_save_image(self):
        if not self.image_manager.has_image:
            return

        stem = os.path.splitext(os.path.basename(self.current_path or "image"))[0]
        default_path = os.path.join(self.last_dir, f"{stem}_cleaned.png")
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Result", default_path, ";;".join(SAVE_FILTERS)
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += SAVE_FILTERS.get(selected_filter, ".png")

        try:
            self.image_manager.save_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot save image", f"Could not save:\n{path}\n\n{exc}")
            return

        self.last_dir = os.path.dirname(path)
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 4000)

    # --- HISTORIA ---

    def on_undo(self):
        # W trakcie rysowania konturu Live-Wire Ctrl+Z cofa ostatni punkt (jak w Photoshopie)
        if isinstance(self.active_tool, LiveWireTool) and self.active_tool.is_busy:
            self.active_tool.remove_last_anchor()
            return
        self.active_tool.cancel()
        if self.image_manager.undo():
            self.refresh_canvas()
            self.statusBar().showMessage("Undo", 1500)

    def on_redo(self):
        self.active_tool.cancel()
        if self.image_manager.redo():
            self.refresh_canvas()
            self.statusBar().showMessage("Redo", 1500)

    # --- ZAZNACZENIE / USUWANIE ---

    def on_clear_selection(self):
        if not self.image_manager.has_image:
            return
        self.active_tool.cancel()
        if np.any(self.image_manager.mask):
            self.image_manager.push_history()
            self.image_manager.clear_mask()
            self.refresh_canvas()

    def on_clean_selection(self):
        # W trakcie działania algorytmu ten sam przycisk służy jako "Stop"
        if self.is_busy:
            self.stop_inpainting()
            return

        mask = self.image_manager.mask
        if not self.image_manager.has_image or mask is None or not np.any(mask):
            self.statusBar().showMessage("Nothing selected - select an object first", 4000)
            return

        self.active_tool.cancel()
        method = self.combo_algo.currentText()
        if method.lower().startswith("criminisi"):
            self.start_criminisi()
            return

        # Telea - szybki, liczony od razu w wątku GUI
        self.statusBar().showMessage(f"Removing selection ({method})...")
        self.image_manager.push_history()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.image_manager.clean_selection(method=method)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.image_manager.discard_last_history()
            if isinstance(exc, NotImplementedError):
                QMessageBox.information(self, "Not available yet", str(exc))
            else:
                QMessageBox.critical(self, "Error", f"Object removal failed:\n\n{exc}")
            self.statusBar().clearMessage()
            return
        QApplication.restoreOverrideCursor()

        self.refresh_canvas()
        self.statusBar().showMessage(f"Selection removed ({method})", 4000)

    # --- CRIMINISI Z WIZUALIZACJĄ ---

    @property
    def is_busy(self) -> bool:
        return self._worker is not None

    def start_criminisi(self):
        # Import leniwy - prototyp algorytmu ładowany dopiero przy pierwszym użyciu
        try:
            from frontend.inpaint_worker import CriminisiWorker
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Cannot load the Criminisi module:\n\n{exc}")
            return

        im = self.image_manager
        self.image_manager.push_history()
        self._worker = CriminisiWorker(im.current_image, im.get_inpaint_mask(), parent=self)
        self._worker.progress.connect(self._on_inpaint_progress)
        self._worker.succeeded.connect(self._on_inpaint_succeeded)
        self._worker.failed.connect(self._on_inpaint_failed)
        self._worker.cancelled.connect(self._on_inpaint_cancelled)

        self._set_busy(True)
        self._worker_started_at = time.perf_counter()
        self.statusBar().showMessage("Criminisi: starting...")
        self._worker.start()

    def stop_inpainting(self):
        if self._worker is not None:
            self.btn_clean.setEnabled(False)
            self.statusBar().showMessage("Stopping...")
            self._worker.requestInterruption()

    def _on_inpaint_progress(self, image_bgr, remaining_mask, filled, total):
        # Wizualizacja: aktualny stan obrazu + fioletowa nakładka na jeszcze niewypełnionym obszarze
        self.canvas.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self.canvas.set_mask(remaining_mask)
        percent = round(100 * filled / total) if total else 100
        self.progress_bar.setValue(percent)
        self.statusBar().showMessage(f"Criminisi: {filled:,} / {total:,} px filled".replace(",", " "))
        if self._worker is not None:
            self._worker.ack_progress()

    def _finish_worker(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._set_busy(False)
        self._displayed_image = None   # płótno pokazywało podgląd - wymuś pełne odświeżenie
        self.refresh_canvas()

    def _on_inpaint_succeeded(self, result_bgr):
        self.image_manager.apply_inpaint_result(result_bgr)
        elapsed = time.perf_counter() - self._worker_started_at
        self._finish_worker()
        self.statusBar().showMessage(f"Selection removed (Criminisi) in {elapsed:.1f} s", 6000)

    def _on_inpaint_cancelled(self):
        self.image_manager.discard_last_history()
        self._finish_worker()
        self.statusBar().showMessage("Criminisi stopped - image unchanged", 4000)

    def _on_inpaint_failed(self, message: str):
        self.image_manager.discard_last_history()
        self._finish_worker()
        self.statusBar().clearMessage()
        QMessageBox.critical(self, "Error", f"Object removal failed:\n\n{message}")

    def _set_busy(self, busy: bool):
        """Blokuje edycję na czas działania algorytmu (zoom i przesuwanie widoku działają dalej)."""
        for w in (self.btn_load, self.btn_save, self.btn_undo, self.btn_redo, self.btn_clear,
                  self.combo_algo, self.slider, *self.tool_buttons.values()):
            w.setEnabled(not busy)
        for sc in self._edit_shortcuts:
            sc.setEnabled(not busy)

        self.btn_clean.setText("Stop" if busy else "Clean selection")
        self.btn_clean.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(busy)
        self.canvas.set_tool(None if busy else self.active_tool)

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.requestInterruption()
            self._worker.wait(5000)
        super().closeEvent(event)

    # --- ODŚWIEŻANIE WIDOKU ---

    def refresh_canvas(self, reset_view: bool = False):
        """Przerysowuje płótno na podstawie stanu ImageManager (obraz + warstwa maski)."""
        image = self.image_manager.current_image
        # Obraz przeliczamy tylko, gdy faktycznie się zmienił (np. Undo samej maski go nie zmienia)
        if reset_view or image is not self._displayed_image:
            self.canvas.set_image(self.image_manager.get_image_rgb(), reset_view=reset_view)
            self._displayed_image = image
        self.canvas.set_mask(self.image_manager.mask)
        self._update_ui_state()

    def _update_ui_state(self):
        has_image = self.image_manager.has_image
        mask = self.image_manager.mask
        selected_px = int(np.count_nonzero(mask)) if has_image and mask is not None else 0

        if self.is_busy:
            return   # stanem przycisków w trakcie pracy algorytmu zarządza _set_busy()

        self.btn_save.setEnabled(has_image)
        self.btn_undo.setEnabled(self.image_manager.can_undo)
        self.btn_redo.setEnabled(self.image_manager.can_redo)
        self.btn_clear.setEnabled(selected_px > 0)
        self.btn_clean.setEnabled(selected_px > 0)

        if has_image:
            h, w = self.image_manager.shape
            self.lbl_file_info.setText(f"{w} × {h} px")
            self.lbl_selection_info.setText(
                f"SELECTED: {selected_px:,} px".replace(",", " ") if selected_px else "NOTHING SELECTED"
            )
        else:
            self.lbl_file_info.setText("NO IMAGE")
            self.lbl_selection_info.setText("NOTHING SELECTED")
            self.lbl_zoom_info.setText("")
            self.lbl_cursor_info.setText("")

    def _update_zoom_label(self, zoom: float):
        self.lbl_zoom_info.setText(f"{zoom * 100:.0f}%")

    def _update_cursor_label(self, x: int, y: int):
        self.lbl_cursor_info.setText(f"X: {x}  Y: {y}" if x >= 0 else "")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectRemoverWindow()
    window.showMaximized()
    sys.exit(app.exec())