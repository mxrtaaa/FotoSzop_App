import os
import cv2
import numpy as np


class ImageManager:
    """
    Zarządza stanem obrazu, maską zaznaczenia oraz wywołaniami algorytmów
    przetwarzania obrazu (niezależnie od warstwy GUI).
    """

    HISTORY_LIMIT = 30

    def __init__(self):
        self.original_image: np.ndarray | None = None  # Oryginalny obraz BGR
        self.current_image: np.ndarray | None = None   # Obraz po dotychczasowej edycji (BGR)
        self.mask: np.ndarray | None = None            # Maska binarna uint8 (0 = tło, 255 = do usunięcia)

        # Historia (Undo/Redo): pary (obraz, maska).
        # UWAGA: current_image nigdy nie jest modyfikowany w miejscu - każda operacja
        # tworzy nową macierz - dlatego obraz w historii trzymamy jako referencję
        # (bez kopiowania), a kopiujemy tylko maskę. Algorytmy podpinane do
        # clean_selection muszą zwracać NOWĄ macierz.
        self._undo_stack: list[tuple[np.ndarray, np.ndarray]] = []
        self._redo_stack: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def has_image(self) -> bool:
        return self.current_image is not None

    @property
    def shape(self) -> tuple[int, int] | None:
        """Zwraca wymiary obrazu jako (wysokość, szerokość)."""
        if not self.has_image:
            return None
        return self.current_image.shape[:2]

    # --- 1. OPERACJE WEJŚCIA / WYJŚCIA ---

    def load_image(self, path: str) -> tuple[int, int]:
        """Wczytuje obraz z dysku (z obsługą polskich znaków w ścieżce) i resetuje maskę."""
        stream = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Nie udało się zdekodować pliku obrazu: {path}")

        self.original_image = img_bgr.copy()
        self.current_image = img_bgr
        self.clear_mask()
        self._undo_stack.clear()
        self._redo_stack.clear()
        return self.shape

    def save_image(self, path: str) -> None:
        """Zapisuje aktualny obraz wynikowy na dysk (z obsługą polskich znaków)."""
        if not self.has_image:
            raise RuntimeError("Brak obrazu do zapisania.")

        ext = os.path.splitext(path)[1] or ".png"
        success, buffer = cv2.imencode(ext, self.current_image)
        if not success:
            raise RuntimeError("Błąd podczas kodowania obrazu do zapisu.")
        buffer.tofile(path)

    def reset_to_original(self) -> None:
        """Przywraca obraz do stanu sprzed wszystkich operacji usuwania."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.clear_mask()

    # --- HISTORIA (UNDO / REDO) ---

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.current_image, self.mask.copy()

    def push_history(self) -> None:
        """Zapamiętuje bieżący stan. Wywoływać PRZED każdą operacją zmieniającą obraz lub maskę."""
        if not self.has_image:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.HISTORY_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def discard_last_history(self) -> None:
        """Usuwa ostatni wpis historii (gdy operacja po push_history() się nie powiodła)."""
        if self._undo_stack:
            self._undo_stack.pop()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self.current_image, self.mask = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self.current_image, self.mask = self._redo_stack.pop()
        return True

    # --- 2. ZARZĄDZANIE MASKĄ I NARZĘDZIA SELEKCJI ---
    # erase/subtract=True usuwa obszar z zaznaczenia zamiast go dodawać (Alt w GUI).

    def clear_mask(self) -> None:
        """Zeruje maskę zaznaczenia."""
        if self.has_image:
            h, w = self.shape
            self.mask = np.zeros((h, w), dtype=np.uint8)

    def draw_brush(self, x: int, y: int, radius: int, erase: bool = False) -> None:
        """Dodaje kołowe zaznaczenie pędzla w punkcie (x, y) w oryginalnych pikselach."""
        if self.mask is None:
            return
        value = 0 if erase else 255
        cv2.circle(self.mask, (int(x), int(y)), int(radius), value, thickness=-1)

    def draw_brush_line(self, pt1: tuple[int, int], pt2: tuple[int, int], radius: int,
                        erase: bool = False) -> None:
        """Rysuje ciągłą linię pędzla między dwoma punktami (zapobiega 'kropkowaniu' przy szybkim ruchu myszy)."""
        if self.mask is None:
            return
        value = 0 if erase else 255
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        cv2.line(self.mask, p1, p2, value, thickness=max(1, int(radius * 2)))
        # Zaokrąglone końce - bez nich łączenia odcinków mają "ząbki"
        cv2.circle(self.mask, p2, int(radius), value, thickness=-1)

    def apply_magic_wand(self, x: int, y: int, tolerance: int = 20, subtract: bool = False) -> None:
        """Zaznacza spójny obszar o podobnym kolorze metodą Region Growing (floodFill)."""
        if not self.has_image:
            return

        h, w = self.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        # OpenCV floodFill wymaga maski większej o 2 piksele z każdej strony
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        diff = (tolerance, tolerance, tolerance)

        # Flaga FLOODFILL_MASK_ONLY sprawia, że floodFill nie modyfikuje samego zdjęcia, a jedynie wypełnia ff_mask
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(self.current_image, ff_mask, (int(x), int(y)), 255, diff, diff, flags)

        # Wycięcie marginesów i połączenie z główną maską aplikacji
        region = ff_mask[1:-1, 1:-1]
        if subtract:
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(region))
        else:
            self.mask = cv2.bitwise_or(self.mask, region)

    # --- 3. GŁÓWNE ALGORYTMY ---

    def apply_livewire_polygon(self, polygon_points: list[tuple[int, int]], subtract: bool = False) -> None:
        """Wypełnia zamknięty kontur wyznaczony przez algorytm Live-Wire."""
        if self.mask is None or len(polygon_points) < 3:
            return
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 0 if subtract else 255)

    def apply_bounding_box_mask(self, box: list[int], shape_dims: tuple[int, int] | None = None) -> None:
        """
        Konwertuje znormalizowany bounding box z Gemini [ymin, xmin, ymax, xmax] (0-1000)
        na prostokątne zaznaczenie w masce.
        """
        if self.mask is None or len(box) != 4:
            return
        h, w = self.shape
        ymin = max(0, int((box[0] / 1000.0) * h))
        xmin = max(0, int((box[1] / 1000.0) * w))
        ymax = min(h, int((box[2] / 1000.0) * h))
        xmax = min(w, int((box[3] / 1000.0) * w))

        cv2.rectangle(self.mask, (xmin, ymin), (xmax, ymax), 255, thickness=-1)

    def clean_selection(self, method: str = "telea", radius: int = 5) -> None:
        """
        Usuwa zaznaczony obszar (maskę) i rekonstruuje tło wybranym algorytmem.
        """
        if not self.has_image or self.mask is None or not np.any(self.mask):
            return

        dilated_mask = self.get_inpaint_mask()

        if method.lower().startswith("telea"):
            result = cv2.inpaint(
                self.current_image, dilated_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA
            )
        elif method.lower().startswith("criminisi"):
            # Import leniwy - prototyp ładowany dopiero przy pierwszym użyciu
            from backend.criminisi_runner import criminisi_inpaint
            result = criminisi_inpaint(self.current_image, dilated_mask)
        else:
            raise ValueError(f"Nieznany algorytm rekonstrukcji: {method}")

        self.apply_inpaint_result(result)

    def get_inpaint_mask(self) -> np.ndarray:
        """
        Maska przekazywana algorytmom rekonstrukcji: lekka dylatacja (poszerzenie o 1 px),
        aby usunąć obwódkę na krawędzi obiektu.
        """
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(self.mask, kernel, iterations=1)

    def apply_inpaint_result(self, result_bgr: np.ndarray) -> None:
        """Ustawia wynik rekonstrukcji jako aktualny obraz i czyści maskę (wynik musi być NOWĄ macierzą)."""
        self.current_image = result_bgr
        self.clear_mask()

    # --- 4. RENDEROWANIE PODGLĄDU DLA GUI ---

    def get_image_rgb(self) -> np.ndarray | None:
        """Zwraca aktualny obraz w RGB (bez nakładki maski - GUI rysuje ją osobną warstwą)."""
        if not self.has_image:
            return None
        return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

    def get_display_rgb(self, overlay_color: tuple[int, int, int] = (197, 138, 249), alpha: float = 0.45) -> np.ndarray | None:
        """
        Zwraca gotową macierz RGB z nałożoną półprzezroczystą maską zaznaczenia,
        gotową do bezpośredniej konwersji na QImage w PySide6.
        Domyślny kolor nakładki (#c58af9) pasuje do akcentu w Twoim GUI.
        """
        if not self.has_image:
            return None

        display_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

        if self.mask is not None and np.any(self.mask):
            overlay = display_rgb.copy()
            overlay[self.mask > 0] = overlay_color
            display_rgb = cv2.addWeighted(overlay, alpha, display_rgb, 1.0 - alpha, 0)

        return display_rgb