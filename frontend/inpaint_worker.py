import threading
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from backend.criminisi_runner import CriminisiRunner


class CriminisiWorker(QThread):
    """
    Uruchamia Criminisi w osobnym wątku (GUI nie zamarza) i co `update_interval`
    sekund wysyła stan pośredni do wizualizacji.

    Wątek pracuje na własnych kopiach danych - nie dotyka ImageManager.
    Wynik trafia do GUI sygnałem `succeeded`, a GUI decyduje, co z nim zrobić.
    """

    progress = Signal(object, object, int, int)   # obraz BGR, maska pozostałych px, wypełnione, wszystkie
    succeeded = Signal(object)                    # wynik BGR
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, image_bgr: np.ndarray, mask: np.ndarray,
                 update_interval: float = 0.08, parent=None):
        super().__init__(parent)
        self._image = image_bgr.copy()
        self._mask = mask.copy()
        self.update_interval = update_interval
        # GUI potwierdza (ack) odebranie podglądu - dopóki tego nie zrobi, nie wysyłamy
        # kolejnego. Bez tego przy dużych zdjęciach w kolejce zbierałyby się klatki.
        self._ready_for_frame = threading.Event()
        self._ready_for_frame.set()

    def ack_progress(self):
        """Wywoływane przez GUI po narysowaniu podglądu."""
        self._ready_for_frame.set()

    def _emit_progress(self, runner):
        self._ready_for_frame.clear()
        self.progress.emit(runner.snapshot_bgr(), runner.remaining_mask(),
                           runner.filled, runner.total)

    def run(self):
        try:
            runner = CriminisiRunner(self._image, self._mask)
            self._emit_progress(runner)

            last_emit = time.perf_counter()
            while not self.isInterruptionRequested():
                if not runner.step():
                    break
                now = time.perf_counter()
                if now - last_emit >= self.update_interval and self._ready_for_frame.is_set():
                    self._emit_progress(runner)
                    last_emit = now

            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self.succeeded.emit(runner.result_bgr())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
