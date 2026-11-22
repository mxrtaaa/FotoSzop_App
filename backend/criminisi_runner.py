"""
Adapter algorytmu Criminisi dla aplikacji.

Algorytm NIE jest tu zaimplementowany - importujemy klasę CriminisiInpainting
z Algorythms_Prototypes/criminisi.py bez żadnych zmian w tamtym pliku.
Ten moduł tylko:
  - konwertuje obraz BGR (aplikacja) <-> RGB (prototyp),
  - pozwala wykonywać algorytm krok po kroku i podglądać stan pośredni
    (potrzebne do wizualizacji w GUI).

Z prototypu korzystamy z:
  metod:  __init__(image, mask, patch_size, search_radius), step()
  pól:    empty, n_unknown, img, unknown, mask0, y0, y1, x0, x1
Zmiana nazw którejkolwiek z tych rzeczy w prototypie wymaga poprawki tutaj.
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

_PROTOTYPES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Algorythms_Prototypes"
)


def _load_prototype():
    # Prototyp importuje `from sobel import ...`, więc jego folder musi być w sys.path
    if _PROTOTYPES_DIR not in sys.path:
        sys.path.append(_PROTOTYPES_DIR)
    path = os.path.join(_PROTOTYPES_DIR, "criminisi.py")
    spec = importlib.util.spec_from_file_location("criminisi_prototype", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_proto = _load_prototype()
CriminisiInpainting = _proto.CriminisiInpainting


class CriminisiRunner:
    """
    Wykonuje Criminisi krok po kroku.

        runner = CriminisiRunner(image_bgr, mask)
        while runner.step():
            preview = runner.snapshot_bgr()   # opcjonalnie, do wizualizacji
        result = runner.result_bgr()
    """

    def __init__(self, image_bgr: np.ndarray, mask: np.ndarray,
                 patch_size: int = 9, search_radius: int = 80):
        self._image_bgr = image_bgr
        self._algo = CriminisiInpainting(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), mask > 0,
            patch_size=patch_size, search_radius=search_radius,
        )
        self.total = 0 if self._algo.empty else int(self._algo.n_unknown)

    @property
    def remaining(self) -> int:
        return 0 if self._algo.empty else int(self._algo.n_unknown)

    @property
    def filled(self) -> int:
        return self.total - self.remaining

    def step(self) -> bool:
        """Wypełnia jedną łatkę. Zwraca False, gdy nie ma już czego wypełniać."""
        if self.remaining <= 0:
            return False
        return bool(self._algo.step())

    def snapshot_bgr(self) -> np.ndarray:
        """Aktualny stan obrazu (niewypełnione piksele są czarne). Zawsze nowa macierz."""
        result = self._image_bgr.copy()
        if self._algo.empty:
            return result
        a = self._algo
        crop_rgb = np.clip(np.rint(a.img * 255.0), 0, 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        view = result[a.y0:a.y1, a.x0:a.x1]
        view[a.mask0] = crop_bgr[a.mask0]      # znane piksele zostają bit-w-bit
        return result

    def remaining_mask(self) -> np.ndarray:
        """Maska (uint8, 0/255) pikseli, które jeszcze nie zostały wypełnione."""
        h, w = self._image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if not self._algo.empty:
            a = self._algo
            mask[a.y0:a.y1, a.x0:a.x1][a.unknown] = 255
        return mask

    def result_bgr(self) -> np.ndarray:
        return self.snapshot_bgr()


def criminisi_inpaint(image_bgr: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
    """Wersja bez wizualizacji: od razu zwraca wynik (nowa macierz BGR)."""
    runner = CriminisiRunner(image_bgr, mask, **kwargs)
    while runner.step():
        pass
    return runner.result_bgr()
