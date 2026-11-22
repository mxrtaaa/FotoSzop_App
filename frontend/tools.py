"""
Narzędzia zaznaczania obiektu na płótnie.

Płótno (ImageCanvas) przekazuje do aktywnego narzędzia zdarzenia we
współrzędnych OBRAZU (float). Narzędzie modyfikuje maskę w ImageManager
i informuje okno główne (host) o zmianach.

Host musi udostępniać:
    host.image_manager          - instancja ImageManager
    host.canvas                 - instancja ImageCanvas
    host.begin_edit()           - wywoływane PRZED zmianą maski (zapis do historii Undo)
    host.mask_edited(rect)      - po zmianie maski; rect = (x0, y0, x1, y1) lub None (cała maska)
"""
import math

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPen, QPolygonF

ACCENT = QColor("#c58af9")
OUTLINE_DARK = QColor(0, 0, 0, 170)
OUTLINE_LIGHT = QColor(255, 255, 255, 230)


def _is_subtract(modifiers) -> bool:
    """Alt = odejmowanie od zaznaczenia (jak w Photoshopie)."""
    return bool(modifiers & Qt.AltModifier)


class SelectionTool:
    """Klasa bazowa - domyślnie wszystkie zdarzenia są ignorowane."""

    name = ""
    hint = ""
    cursor = Qt.CrossCursor

    # Konfiguracja suwaka w panelu bocznym (None = narzędzie nie używa suwaka)
    slider_label: str | None = None
    slider_range = (1, 100)
    slider_default = 30
    slider_suffix = ""

    def __init__(self, host):
        self.host = host
        self.value = self.slider_default   # aktualna wartość suwaka dla tego narzędzia

    @property
    def manager(self):
        return self.host.image_manager

    @property
    def canvas(self):
        return self.host.canvas

    @property
    def is_busy(self) -> bool:
        """True, jeśli narzędzie ma niezakończoną operację (np. otwarty kontur Live-Wire)."""
        return False

    def press(self, x: float, y: float, modifiers) -> None: ...
    def drag(self, x: float, y: float, modifiers) -> None: ...
    def release(self, x: float, y: float, modifiers) -> None: ...
    def double_click(self, x: float, y: float, modifiers) -> None: ...
    def hover(self, x: float, y: float) -> None: ...
    def leave(self) -> None: ...

    def key_press(self, key, modifiers) -> bool:
        """Zwraca True, jeśli narzędzie obsłużyło klawisz."""
        return False

    def cancel(self) -> None:
        """Przerywa niezakończoną operację (zmiana narzędzia, nowy obraz, Undo...)."""

    def paint(self, painter, canvas) -> None:
        """Rysuje podgląd narzędzia na płótnie (we współrzędnych widgetu)."""


# --------------------------------------------------------------------------- PĘDZEL

class BrushTool(SelectionTool):
    name = "Brush"
    hint = "Paint over the object.\nAlt + drag: erase   [ / ]: size"
    slider_label = "Size"
    slider_range = (1, 300)
    slider_default = 30
    slider_suffix = " px"

    def __init__(self, host):
        super().__init__(host)
        self._last: tuple[int, int] | None = None
        self._erase = False
        self._hover: tuple[float, float] | None = None

    @property
    def radius(self) -> int:
        # Rozmiar pędzla = średnica (jak w Photoshopie)
        return max(1, round(self.value / 2))

    def _dirty_rect(self, p1, p2):
        r = self.radius + 2
        return (min(p1[0], p2[0]) - r, min(p1[1], p2[1]) - r,
                max(p1[0], p2[0]) + r + 1, max(p1[1], p2[1]) + r + 1)

    def press(self, x, y, modifiers):
        if not self.manager.has_image:
            return
        pt = (int(math.floor(x)), int(math.floor(y)))
        self._erase = _is_subtract(modifiers)
        self.host.begin_edit()
        self.manager.draw_brush(pt[0], pt[1], self.radius, erase=self._erase)
        self._last = pt
        self.host.mask_edited(self._dirty_rect(pt, pt))

    def drag(self, x, y, modifiers):
        if self._last is None:
            return
        pt = (int(math.floor(x)), int(math.floor(y)))
        if pt == self._last:
            return
        self.manager.draw_brush_line(self._last, pt, self.radius, erase=self._erase)
        self.host.mask_edited(self._dirty_rect(self._last, pt))
        self._last = pt

    def release(self, x, y, modifiers):
        self._last = None

    def hover(self, x, y):
        self._hover = (x, y)
        self.canvas.update()

    def leave(self):
        self._hover = None
        self.canvas.update()

    def cancel(self):
        self._last = None

    def paint(self, painter, canvas):
        if self._hover is None:
            return
        px, py = math.floor(self._hover[0]), math.floor(self._hover[1])
        center = canvas.pixel_center_to_widget(px, py)
        r = max(2.0, self.radius * canvas.zoom)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(OUTLINE_DARK, 3))
        painter.drawEllipse(center, r, r)
        painter.setPen(QPen(OUTLINE_LIGHT, 1))
        painter.drawEllipse(center, r, r)


# --------------------------------------------------------------------------- RÓŻDŻKA

class MagicWandTool(SelectionTool):
    name = "Magic Wand"
    hint = "Click an area of similar color.\nAlt + click: subtract"
    slider_label = "Tolerance"
    slider_range = (0, 100)
    slider_default = 20
    slider_suffix = ""

    def press(self, x, y, modifiers):
        px, py = int(math.floor(x)), int(math.floor(y))
        if not self.canvas.is_inside_image(px, py):
            return
        self.host.begin_edit()
        self.manager.apply_magic_wand(px, py, tolerance=self.value,
                                      subtract=_is_subtract(modifiers))
        self.host.mask_edited(None)


# --------------------------------------------------------------------------- LIVE-WIRE

class StraightLinePathFinder:
    """
    Zastępczy "algorytm" Live-Wire: łączy punkty prostymi odcinkami
    (działa jak lasso wielokątne).

    Właściwy algorytm Live-Wire powinien implementować TEN SAM interfejs
    i zostać przekazany do LiveWireTool(path_finder=...):

        set_image(image_bgr)  - nowy obraz; tu np. liczymy mapę kosztów (Sobel)
        set_seed(point)       - użytkownik kliknął punkt; tu np. Dijkstra od tego punktu
        path_to(point)        - ścieżka od ostatniego seeda do `point` jako lista (x, y),
                                łącznie z oboma końcami; wołane przy KAŻDYM ruchu myszy,
                                więc musi być szybkie (np. odczyt z gotowej mapy poprzedników)
    """

    def __init__(self):
        self.seed: tuple[int, int] | None = None

    def set_image(self, image_bgr) -> None:
        pass

    def set_seed(self, point: tuple[int, int]) -> None:
        self.seed = point

    def path_to(self, point: tuple[int, int]) -> list[tuple[int, int]]:
        if self.seed is None:
            return [point]
        return [self.seed, point]


class LiveWireTool(SelectionTool):
    name = "Live-Wire"
    hint = ("Click to add points along the edge.\n"
            "Close: click the first point, double-click or Enter.\n"
            "Backspace: remove last point   Esc: cancel\n"
            "Alt while closing: subtract")
    slider_label = None
    CLOSE_DISTANCE_PX = 10      # odległość (na ekranie) od pierwszego punktu, przy której kontur się zamyka

    def __init__(self, host, path_finder=None):
        super().__init__(host)
        self.path_finder = path_finder or StraightLinePathFinder()
        self._finder_image = None                       # obraz, dla którego path_finder jest przygotowany
        self.anchors: list[tuple[int, int]] = []        # kliknięte punkty
        self.segments: list[list[tuple[int, int]]] = [] # ścieżki między kolejnymi punktami
        self._preview: list[tuple[int, int]] = []       # ścieżka od ostatniego punktu do kursora
        self._hover: tuple[float, float] | None = None

    @property
    def is_busy(self) -> bool:
        return bool(self.anchors)

    # --- pomocnicze ---

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.canvas.image_size
        return (min(max(int(math.floor(x)), 0), w - 1),
                min(max(int(math.floor(y)), 0), h - 1))

    def _near_first_anchor(self, x: float, y: float) -> bool:
        if len(self.anchors) < 3:
            return False
        fx, fy = self.anchors[0]
        dist_px = math.hypot((x - fx - 0.5) * self.canvas.zoom, (y - fy - 0.5) * self.canvas.zoom)
        return dist_px <= self.CLOSE_DISTANCE_PX

    def _prepare_finder(self):
        image = self.manager.current_image
        if image is not self._finder_image:
            self.path_finder.set_image(image)
            self._finder_image = image

    def _update_preview(self):
        if self.anchors and self._hover is not None:
            target = (self.anchors[0] if self._near_first_anchor(*self._hover)
                      else self._clamp(*self._hover))
            self._preview = self.path_finder.path_to(target)
        else:
            self._preview = []
        self.canvas.update()

    # --- zdarzenia ---

    def press(self, x, y, modifiers):
        if not self.manager.has_image:
            return
        if self._near_first_anchor(x, y):
            self.close_contour(modifiers)
            return

        pt = self._clamp(x, y)
        if self.anchors:
            if pt == self.anchors[-1]:
                return
            self.segments.append(self.path_finder.path_to(pt))
        else:
            self._prepare_finder()
        self.anchors.append(pt)
        self.path_finder.set_seed(pt)
        self._update_preview()

    def double_click(self, x, y, modifiers):
        if len(self.anchors) >= 3:
            self.close_contour(modifiers)

    def hover(self, x, y):
        self._hover = (x, y)
        if self.anchors:
            self._update_preview()

    def leave(self):
        self._hover = None
        self._preview = []
        self.canvas.update()

    def key_press(self, key, modifiers):
        if not self.anchors:
            return False
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if len(self.anchors) >= 3:
                self.close_contour(modifiers)
            return True
        if key == Qt.Key_Escape:
            self.cancel()
            return True
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            self.remove_last_anchor()
            return True
        return False

    # --- operacje ---

    def remove_last_anchor(self):
        if not self.anchors:
            return
        self.anchors.pop()
        if self.segments:
            self.segments.pop()
        if self.anchors:
            self.path_finder.set_seed(self.anchors[-1])
        self._update_preview()

    def close_contour(self, modifiers=Qt.NoModifier):
        if len(self.anchors) < 3:
            return
        closing = self.path_finder.path_to(self.anchors[0])
        polygon = [self.anchors[0]]
        for seg in self.segments + [closing]:
            polygon.extend(seg)

        self.host.begin_edit()
        self.manager.apply_livewire_polygon(polygon, subtract=_is_subtract(modifiers))
        self.cancel()
        self.host.mask_edited(None)

    def cancel(self):
        self.anchors.clear()
        self.segments.clear()
        self._preview = []
        self.canvas.update()

    # --- rysowanie ---

    def _polyline(self, canvas, points) -> QPolygonF:
        return QPolygonF([canvas.pixel_center_to_widget(x, y) for x, y in points])

    def paint(self, painter, canvas):
        if not self.anchors:
            return
        painter.setBrush(Qt.NoBrush)

        committed = [self.anchors[0]]
        for seg in self.segments:
            committed.extend(seg)
        poly = self._polyline(canvas, committed)
        painter.setPen(QPen(OUTLINE_DARK, 4))
        painter.drawPolyline(poly)
        painter.setPen(QPen(ACCENT, 2))
        painter.drawPolyline(poly)

        if self._preview:
            preview = self._polyline(canvas, self._preview)
            pen = QPen(OUTLINE_LIGHT, 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.drawPolyline(preview)

        # punkty kontrolne; pierwszy powiększony, gdy kursor jest na tyle blisko, by zamknąć kontur
        closable = self._hover is not None and self._near_first_anchor(*self._hover)
        for i, (ax, ay) in enumerate(self.anchors):
            c = canvas.pixel_center_to_widget(ax, ay)
            half = 6.0 if (i == 0 and closable) else 3.0
            painter.setPen(QPen(OUTLINE_DARK, 1))
            painter.setBrush(ACCENT if i == 0 else OUTLINE_LIGHT)
            painter.drawRect(QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half))
        painter.setBrush(Qt.NoBrush)
