import math
import os

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QPen
from PySide6.QtWidgets import QWidget

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Kolor nakładki zaznaczenia (#c58af9 - akcent GUI) i jej przezroczystość
OVERLAY_COLOR = (197, 138, 249)
OVERLAY_ALPHA = 0.45


def _premultiplied_rgba(rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    a = round(alpha * 255)
    return np.array([round(c * a / 255) for c in rgb] + [a], dtype=np.uint8)


class ImageCanvas(QWidget):
    """
    Płótno do wyświetlania obrazu z zoomem, przesuwaniem i warstwą zaznaczenia.

    Zdarzenia myszy/klawiatury są przekazywane do aktywnego narzędzia
    (frontend/tools.py) we współrzędnych OBRAZU, więc narzędzia nie muszą
    nic wiedzieć o aktualnym powiększeniu ani przesunięciu widoku.

    Sterowanie widokiem:
      - kółko myszy            -> zoom względem kursora
      - środkowy przycisk      -> przesuwanie widoku
      - spacja + lewy przycisk -> przesuwanie widoku (jak w Photoshopie)
    """

    cursor_moved = Signal(int, int)     # pozycja kursora nad obrazem; (-1, -1) poza obrazem
    zoom_changed = Signal(float)        # aktualny zoom (1.0 = 100%)
    file_dropped = Signal(str)          # przeciągnięto plik obrazu na płótno

    MIN_ZOOM = 0.02
    MAX_ZOOM = 32.0
    ZOOM_STEP = 1.25
    FIT_MARGIN = 20

    BACKGROUND = QColor("#121212")
    PLACEHOLDER = QColor("#666666")
    BORDER = QColor("#333333")
    OVERLAY_PIXEL = _premultiplied_rgba(OVERLAY_COLOR, OVERLAY_ALPHA)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

        self.placeholder_text = "NO IMAGE\n\nChoose a file or drop an image here"

        # Warstwa obrazu
        self._pixmap: QPixmap | None = None
        self._scaled_cache: tuple[float, QPixmap] | None = None

        # Warstwa zaznaczenia (RGBA premultiplied, aktualizowana fragmentami)
        self._mask_rgba: np.ndarray | None = None
        self._mask_qimage: QImage | None = None
        self._mask_visible = False

        # Widok
        self._zoom = 1.0
        self._offset = QPointF(0, 0)    # lewy górny róg obrazu we współrzędnych widgetu
        self._fit_mode = True           # czy widok ma się dopasowywać przy zmianie rozmiaru okna

        # Interakcja
        self.tool = None
        self._space_held = False
        self._panning = False
        self._pan_last = QPointF()
        self._left_down = False

    # --- PUBLICZNE API: OBRAZ I MASKA ---

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def image_size(self) -> tuple[int, int] | None:
        """(szerokość, wysokość) wyświetlanego obrazu."""
        if self._pixmap is None:
            return None
        return self._pixmap.width(), self._pixmap.height()

    def set_image(self, rgb: np.ndarray | None, reset_view: bool = False) -> None:
        """
        Ustawia obraz do wyświetlenia (macierz RGB uint8, HxWx3).
        reset_view=True dopasowuje obraz do okna (używać przy wczytaniu nowego pliku).
        """
        if rgb is None:
            self._pixmap = None
            self._scaled_cache = None
            self.set_mask(None)
            self.update()
            return

        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        # .copy() - QImage nie może wskazywać na pamięć numpy, która zaraz zniknie
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._scaled_cache = None

        if reset_view or self._fit_mode:
            self.fit_to_window()
        self.update()

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Przebudowuje całą warstwę zaznaczenia z maski (uint8, 0 / 255)."""
        if mask is None:
            self._mask_rgba = None
            self._mask_qimage = None
            self._mask_visible = False
            self.update()
            return

        h, w = mask.shape[:2]
        if self._mask_rgba is None or self._mask_rgba.shape[:2] != (h, w):
            self._mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        self._fill_overlay(mask, 0, 0, w, h)

    def update_mask_region(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        """Aktualizuje tylko prostokąt [x0:x1, y0:y1] warstwy zaznaczenia (szybkie przy malowaniu pędzlem)."""
        h, w = mask.shape[:2]
        if self._mask_rgba is None or self._mask_rgba.shape[:2] != (h, w):
            self.set_mask(mask)
            return
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x0 >= x1 or y0 >= y1:
            return
        self._fill_overlay(mask, x0, y0, x1, y1)

    def _fill_overlay(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        region = self._mask_rgba[y0:y1, x0:x1]
        region[:] = 0
        region[mask[y0:y1, x0:x1] > 0] = self.OVERLAY_PIXEL

        self._mask_visible = bool(np.any(mask))
        h, w = self._mask_rgba.shape[:2]
        # Lekki wrapper (bez kopiowania) - bufor numpy trzymamy w self._mask_rgba
        self._mask_qimage = QImage(self._mask_rgba.data, w, h, w * 4,
                                   QImage.Format_RGBA8888_Premultiplied)
        self.update()

    # --- PUBLICZNE API: NARZĘDZIA ---

    def set_tool(self, tool) -> None:
        self.tool = tool
        self._update_cursor()
        self.update()

    # --- PUBLICZNE API: WIDOK ---

    def fit_to_window(self) -> None:
        """Dopasowuje obraz do okna (małych obrazów nie powiększa ponad 100%)."""
        if self._pixmap is None:
            return
        iw, ih = self._pixmap.width(), self._pixmap.height()
        avail_w = max(1, self.width() - 2 * self.FIT_MARGIN)
        avail_h = max(1, self.height() - 2 * self.FIT_MARGIN)
        self._zoom = min(avail_w / iw, avail_h / ih, 1.0)
        self._center_image()
        self._fit_mode = True
        self._scaled_cache = None
        self.zoom_changed.emit(self._zoom)
        self.update()

    def set_actual_size(self) -> None:
        """Zoom 100% wyśrodkowany."""
        if self._pixmap is None:
            return
        self._zoom = 1.0
        self._center_image()
        self._fit_mode = False
        self._scaled_cache = None
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self) -> None:
        self._zoom_at(self._zoom * self.ZOOM_STEP, self._widget_center())

    def zoom_out(self) -> None:
        self._zoom_at(self._zoom / self.ZOOM_STEP, self._widget_center())

    def widget_to_image(self, pos: QPointF) -> tuple[float, float]:
        """Przelicza punkt widgetu na współrzędne obrazu (float, mogą wychodzić poza obraz)."""
        return (
            (pos.x() - self._offset.x()) / self._zoom,
            (pos.y() - self._offset.y()) / self._zoom,
        )

    def image_to_widget(self, x: float, y: float) -> QPointF:
        """Przelicza współrzędne obrazu na punkt widgetu (do rysowania podglądów narzędzi)."""
        return QPointF(self._offset.x() + x * self._zoom, self._offset.y() + y * self._zoom)

    def pixel_center_to_widget(self, x: int, y: int) -> QPointF:
        """Środek piksela (x, y) we współrzędnych widgetu."""
        return self.image_to_widget(x + 0.5, y + 0.5)

    def is_inside_image(self, x: int, y: int) -> bool:
        return (
            self._pixmap is not None
            and 0 <= x < self._pixmap.width()
            and 0 <= y < self._pixmap.height()
        )

    # --- LOGIKA WIDOKU ---

    def _widget_center(self) -> QPointF:
        return QPointF(self.width() / 2, self.height() / 2)

    def _center_image(self) -> None:
        iw = self._pixmap.width() * self._zoom
        ih = self._pixmap.height() * self._zoom
        self._offset = QPointF((self.width() - iw) / 2, (self.height() - ih) / 2)

    def _zoom_at(self, new_zoom: float, anchor: QPointF) -> None:
        """Zmienia zoom tak, aby punkt obrazu pod `anchor` pozostał w miejscu."""
        if self._pixmap is None:
            return
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if math.isclose(new_zoom, self._zoom):
            return
        ix, iy = self.widget_to_image(anchor)
        self._zoom = new_zoom
        self._offset = QPointF(anchor.x() - ix * new_zoom, anchor.y() - iy * new_zoom)
        self._fit_mode = False
        self._scaled_cache = None
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _emit_cursor(self, ix: float, iy: float) -> None:
        x, y = int(math.floor(ix)), int(math.floor(iy))
        if self.is_inside_image(x, y):
            self.cursor_moved.emit(x, y)
        else:
            self.cursor_moved.emit(-1, -1)

    def _update_cursor(self) -> None:
        if self._panning:
            self.setCursor(Qt.ClosedHandCursor)
        elif self._space_held:
            self.setCursor(Qt.OpenHandCursor)
        elif self.tool is not None and self._pixmap is not None:
            self.setCursor(self.tool.cursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    # --- RYSOWANIE ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.BACKGROUND)

        if self._pixmap is None:
            painter.setPen(self.PLACEHOLDER)
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
            return

        iw, ih = self._pixmap.width(), self._pixmap.height()
        target = QRectF(self._offset.x(), self._offset.y(), iw * self._zoom, ih * self._zoom)

        # 1. Obraz
        if self._zoom < 1.0:
            # Pomniejszenie: wygładzona kopia trzymana w cache (przesuwanie jej nie unieważnia)
            if self._scaled_cache is None or self._scaled_cache[0] != self._zoom:
                scaled = self._pixmap.scaled(
                    max(1, round(iw * self._zoom)), max(1, round(ih * self._zoom)),
                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
                )
                self._scaled_cache = (self._zoom, scaled)
            painter.drawPixmap(target.topLeft(), self._scaled_cache[1])
        else:
            # Powiększenie: bez wygładzania, żeby było widać pojedyncze piksele
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        # 2. Zaznaczenie
        if self._mask_visible and self._mask_qimage is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.drawImage(target, self._mask_qimage, QRectF(self._mask_qimage.rect()))

        painter.setPen(QPen(self.BORDER, 1))
        painter.drawRect(target.adjusted(-1, -1, 0, 0))

        # 3. Podgląd narzędzia (obrys pędzla, kontur Live-Wire, ...)
        if self.tool is not None:
            painter.setRenderHint(QPainter.Antialiasing, True)
            self.tool.paint(painter, self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()

    # --- MYSZ ---

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        steps = event.angleDelta().y() / 120
        if steps:
            self._zoom_at(self._zoom * (self.ZOOM_STEP ** steps), event.position())
            if self.tool is not None:
                self.tool.hover(*self.widget_to_image(event.position()))
        event.accept()

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        pos = event.position()
        wants_pan = event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._space_held
        )
        if wants_pan:
            self._panning = True
            self._pan_last = pos
            self._update_cursor()
            return

        if event.button() == Qt.LeftButton:
            self._left_down = True
            if self.tool is not None:
                self.tool.press(*self.widget_to_image(pos), event.modifiers())

    def mouseMoveEvent(self, event):
        if self._pixmap is None:
            return
        pos = event.position()
        if self._panning:
            self._offset += pos - self._pan_last
            self._pan_last = pos
            self._fit_mode = False
            self.update()

        ix, iy = self.widget_to_image(pos)
        if self.tool is not None and not self._panning:
            self.tool.hover(ix, iy)
            if self._left_down:
                self.tool.drag(ix, iy, event.modifiers())
        self._emit_cursor(ix, iy)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._update_cursor()
            return
        if self._left_down and event.button() == Qt.LeftButton:
            self._left_down = False
            if self.tool is not None:
                self.tool.release(*self.widget_to_image(event.position()), event.modifiers())

    def mouseDoubleClickEvent(self, event):
        if self._pixmap is None or self.tool is None or event.button() != Qt.LeftButton:
            return
        self.tool.double_click(*self.widget_to_image(event.position()), event.modifiers())

    def enterEvent(self, event):
        # Aby spacja i klawisze narzędzi działały od razu po najechaniu na płótno
        self.setFocus(Qt.MouseFocusReason)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.cursor_moved.emit(-1, -1)
        if self.tool is not None:
            self.tool.leave()
        super().leaveEvent(event)

    # --- KLAWIATURA ---

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self._update_cursor()
            return
        if self.tool is not None and self._pixmap is not None:
            if self.tool.key_press(event.key(), event.modifiers()):
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._update_cursor()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space_held = False
        self._update_cursor()
        super().focusOutEvent(event)

    # --- DRAG & DROP ---

    @staticmethod
    def _dropped_image_path(mime) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path and os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS:
                return path
        return None

    def dragEnterEvent(self, event):
        if self._dropped_image_path(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._dropped_image_path(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.file_dropped.emit(path)
