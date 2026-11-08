from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

Sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float64)

Sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float64)

# convolve2d = correlate2d ze skręconym jądrem; liczymy je raz
_SX_FLIP = np.ascontiguousarray(Sobel_x[::-1, ::-1])
_SY_FLIP = np.ascontiguousarray(Sobel_y[::-1, ::-1])

_GRAY_W64 = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
_GRAY_W32 = _GRAY_W64.astype(np.float32)


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB/RGBA/skala szarości -> 2D float. float32 zostaje float32 (szybciej)."""
    f32 = image.dtype == np.float32
    if image.ndim == 2:
        return image if f32 else image.astype(np.float64)
    if image.ndim == 3:
        c = image.shape[2]
        if c == 1:
            return image[..., 0] if f32 else image[..., 0].astype(np.float64)
        if c in (3, 4):
            w = _GRAY_W32 if f32 else _GRAY_W64
            return image[..., :3] @ w          # jedno mnożenie macierzowe, bez pętli
    raise ValueError(f"Nieobsługiwany kształt obrazu: {image.shape}. Wymagane RGB lub skala szarości.")


def _prep(image: np.ndarray) -> np.ndarray:
    if image.dtype not in (np.float32, np.float64):
        image = image.astype(np.float32)
    return np.ascontiguousarray(image)


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Korelacja z zerowym paddingiem (identycznie jak wersja z pętlami), ale w C."""
    img = _prep(image)
    k = np.ascontiguousarray(kernel, dtype=img.dtype)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_CONSTANT)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return correlate2d(image, np.flipud(np.fliplr(kernel)))


def sobel_xy(gray: np.ndarray):
    """Szybkie gx, gy (bez magnitude/direction) - do użycia w pętli algorytmu."""
    img = _prep(gray)
    kx = _SX_FLIP.astype(img.dtype)
    ky = _SY_FLIP.astype(img.dtype)
    gx = cv2.filter2D(img, -1, kx, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.filter2D(img, -1, ky, borderType=cv2.BORDER_CONSTANT)
    return gx, gy


@dataclass
class SobelResult:
    gx: np.ndarray
    gy: np.ndarray
    magnitude: np.ndarray
    direction: np.ndarray


def sobell(image, pad_mode="edge") -> SobelResult:
    gx, gy = sobel_xy(to_gray(image))
    return SobelResult(gx=gx, gy=gy,
                       magnitude=np.hypot(gx, gy),
                       direction=np.arctan2(gy, gx))


def edge_map(image: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = sobell(image).magnitude
    return (magnitude > threshold).astype(np.uint8) * 255
