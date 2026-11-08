from __future__ import annotations
import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from sobel import sobel_xy, to_gray

_K3 = np.ones((3, 3), dtype=np.uint8)


def grow_mask(mask: np.ndarray, px: int = 2) -> np.ndarray:
    """Poszerza maskę o px pikseli w każdą stronę (okrągłe jądro)."""
    mask = np.asarray(mask).astype(bool)
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


class CriminisiInpainting:
    """
    Criminisi z lokalnymi aktualizacjami:
      * praca tylko na wycinku obrazu wokół maski (+ promień przeszukiwania),
      * gradienty i priorytety przeliczane tylko wokół ostatnio wypełnionej łatki,
      * dopasowanie łatki przez cv2.matchTemplate(TM_SQDIFF, mask) (FFT) zamiast
        ręcznego SSD na dziesiątkach tysięcy kandydatów,
      * prawdziwy front wypełniania (znane sąsiedztwo), a nie cała maska.
    Zalecany nieparzysty patch_size.
    """

    def __init__(self, image: np.ndarray, mask: np.ndarray,
                 patch_size: int = 9, search_radius: int = 80):
        image = np.asarray(image)
        self._was_gray = image.ndim == 2
        if self._was_gray:
            image = image[..., None]
        mask = np.asarray(mask).astype(bool)

        self.ps = patch_size
        self.half = patch_size // 2
        self.R = search_radius
        self.full_img = image
        self.full_mask = mask

        ys, xs = np.nonzero(mask)
        self.empty = ys.size == 0
        if self.empty:
            return

        # --- wycinek roboczy: maska + margines (promień szukania + łatka) ---
        m = self.R + self.ps
        H0, W0 = mask.shape
        self.y0, self.y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, H0)
        self.x0, self.x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, W0)

        self.img = image[self.y0:self.y1, self.x0:self.x1].astype(np.float32) / 255.0
        self.unknown = mask[self.y0:self.y1, self.x0:self.x1].copy()
        self.mask0 = self.unknown.copy()
        self.img[self.unknown] = 0.0
        self.H, self.W = self.unknown.shape
        self.nch = self.img.shape[2]
        if self.H < self.ps or self.W < self.ps:
            raise ValueError("Obraz mniejszy niż łatka.")

        self.conf = (~self.unknown).astype(np.float32)
        self.n_unknown = int(self.unknown.sum())

        # pozycje, w których łatka w całości leży w znanym obszarze (stałe w czasie)
        self.valid = ~sliding_window_view(self.unknown, (self.ps, self.ps)).any(axis=(2, 3))

        shp = (self.H, self.W)
        self.gx = np.zeros(shp, np.float32)
        self.gy = np.zeros(shp, np.float32)
        self.gmag = np.zeros(shp, np.float32)
        self.nx = np.zeros(shp, np.float32)
        self.ny = np.zeros(shp, np.float32)
        self.prio = np.full(shp, -1.0, np.float32)

        self._update_grad(0, self.H, 0, self.W)
        self._update_prio(0, self.H, 0, self.W)

    # ------------------------------------------------------------------ helpers
    def _ext(self, arr, a0, a1, b0, b1, dtype=np.float32):
        """Wycinek arr[a0:a1,b0:b1] rozszerzony o margines łatki; poza obrazem = 0."""
        pl, pr = self.half, self.ps - 1 - self.half
        ra0, ra1, rb0, rb1 = a0 - pl, a1 + pr, b0 - pl, b1 + pr
        out = np.zeros((ra1 - ra0, rb1 - rb0), dtype)
        sa0, sa1 = max(ra0, 0), min(ra1, self.H)
        sb0, sb1 = max(rb0, 0), min(rb1, self.W)
        out[sa0 - ra0:sa1 - ra0, sb0 - rb0:sb1 - rb0] = arr[sa0:sa1, sb0:sb1]
        return out

    def _update_grad(self, a0, a1, b0, b1):
        a0, a1, b0, b1 = max(a0, 0), min(a1, self.H), max(b0, 0), min(b1, self.W)
        e0, e1 = max(a0 - 2, 0), min(a1 + 2, self.H)
        f0, f1 = max(b0 - 2, 0), min(b1 + 2, self.W)

        unk = self.unknown[e0:e1, f0:f1]
        gray = to_gray(self.img[e0:e1, f0:f1])
        gray = np.where(unk, np.float32(0), gray).astype(np.float32)
        sx, sy = sobel_xy(gray)

        valid_grad = ~cv2.dilate(unk.astype(np.uint8), _K3).astype(bool)
        gx = np.where(valid_grad, sx / 8.0, 0.0).astype(np.float32)
        gy = np.where(valid_grad, sy / 8.0, 0.0).astype(np.float32)

        mx, my = sobel_xy(unk.astype(np.float32))
        nm = np.maximum(np.hypot(mx, my), 1.0)

        s = (slice(a0 - e0, a1 - e0), slice(b0 - f0, b1 - f0))
        self.gx[a0:a1, b0:b1] = gx[s]
        self.gy[a0:a1, b0:b1] = gy[s]
        self.gmag[a0:a1, b0:b1] = np.hypot(gx, gy)[s]
        self.nx[a0:a1, b0:b1] = (mx / nm)[s]
        self.ny[a0:a1, b0:b1] = (my / nm)[s]

    def _update_prio(self, a0, a1, b0, b1):
        a0, a1, b0, b1 = max(a0, 0), min(a1, self.H), max(b0, 0), min(b1, self.W)
        for s in range(a0, a1, 64):          # porcjami, żeby nie zjadać RAM-u przy pełnej inicjalizacji
            self._prio_block(s, min(s + 64, a1), b0, b1)

    def _prio_block(self, a0, a1, b0, b1):
        ps = self.ps
        h, w = a1 - a0, b1 - b0

        # front = nieznane piksele, które mają znanego sąsiada
        g0, g1 = max(a0 - 1, 0), min(a1 + 1, self.H)
        h0, h1 = max(b0 - 1, 0), min(b1 + 1, self.W)
        unk = self.unknown[g0:g1, h0:h1]
        known_dil = cv2.dilate((~unk).astype(np.uint8), _K3).astype(bool)
        front = (unk & known_dil)[a0 - g0:a1 - g0, b0 - h0:b1 - h0]

        # C: średnia pewność w łatce
        conf_ext = self._ext(self.conf, a0, a1, b0, b1)
        C = sliding_window_view(conf_ext, (ps, ps)).sum(axis=(2, 3)) / (ps * ps)

        # D: gradient w najsilniejszym znanym pikselu łatki vs normalna do frontu
        known_ext = ~self._ext(self.unknown, a0, a1, b0, b1, dtype=bool)
        gm = self._ext(self.gmag, a0, a1, b0, b1) * known_ext
        gm_w = sliding_window_view(gm, (ps, ps)).reshape(h, w, ps * ps)
        k = gm_w.argmax(axis=-1)[..., None]
        mx = np.take_along_axis(gm_w, k, axis=-1)[..., 0]
        gx_w = sliding_window_view(self._ext(self.gx, a0, a1, b0, b1), (ps, ps)).reshape(h, w, ps * ps)
        gy_w = sliding_window_view(self._ext(self.gy, a0, a1, b0, b1), (ps, ps)).reshape(h, w, ps * ps)
        gxx = np.take_along_axis(gx_w, k, axis=-1)[..., 0]
        gyy = np.take_along_axis(gy_w, k, axis=-1)[..., 0]
        D = np.where(mx > 0,
                     np.abs(-gyy * self.nx[a0:a1, b0:b1] + gxx * self.ny[a0:a1, b0:b1]),
                     1e-3)

        prio = C * (D + 1e-3)
        self.prio[a0:a1, b0:b1] = np.where(front, prio, -1.0)

    def _search(self, target, mask3, rmin, rmax, cmin, cmax):
        """Najlepsza łatka (SSD po znanych pikselach) w zadanym oknie; None jeśli brak kandydatów."""
        srch = np.ascontiguousarray(self.img[rmin:rmax, cmin:cmax])
        res = cv2.matchTemplate(srch, target, cv2.TM_SQDIFF, mask=mask3)
        v = self.valid[rmin:rmin + res.shape[0], cmin:cmin + res.shape[1]]
        res = np.where(v & np.isfinite(res), res, np.inf)
        i = int(np.argmin(res))
        if not np.isfinite(res.flat[i]):
            return None
        ii, jj = divmod(i, res.shape[1])
        return self.img[rmin + ii:rmin + ii + self.ps, cmin + jj:cmin + jj + self.ps]

    # ------------------------------------------------------------------ main
    def step(self) -> bool:
        k = int(np.argmax(self.prio))
        if self.prio.flat[k] < 0:
            return False
        r, c = divmod(k, self.W)
        ps, half, R = self.ps, self.half, self.R

        r0 = min(max(r - half, 0), self.H - ps)
        c0 = min(max(c - half, 0), self.W - ps)
        sl = (slice(r0, r0 + ps), slice(c0, c0 + ps))

        C = float(self.conf[sl].sum()) / (ps * ps)
        target = np.ascontiguousarray(self.img[sl])
        unk = self.unknown[sl].copy()
        mask3 = np.ascontiguousarray(
            np.repeat((~unk)[..., None].astype(np.float32), self.nch, axis=2))

        best = self._search(target, mask3,
                            max(0, r0 - R), min(self.H, r0 + R + ps),
                            max(0, c0 - R), min(self.W, c0 + R + ps))
        if best is None:                                  # fallback: cały wycinek
            best = self._search(target, mask3, 0, self.H, 0, self.W)
        if best is None:
            return False

        patch = self.img[sl]
        patch[unk] = best[unk]
        self.conf[sl][unk] = C
        self.unknown[sl] = False
        self.n_unknown -= int(unk.sum())

        # lokalne aktualizacje wokół wypełnionej łatki
        self._update_grad(r0 - 1, r0 + ps + 1, c0 - 1, c0 + ps + 1)
        self._update_prio(r0 - ps, r0 + 2 * ps, c0 - ps, c0 + 2 * ps)
        return True

    def run(self, verbose: bool = True):
        result = self.full_img.copy()
        if self.empty:
            return result[..., 0] if self._was_gray else result

        i = 0
        while self.n_unknown > 0:
            if not self.step():
                break
            if verbose and i % 50 == 0:
                print(f"Krok {i}: pozostało {self.n_unknown} nieznanych pikseli", flush=True)
            i += 1

        out = np.clip(np.rint(self.img * 255.0), 0, 255).astype(result.dtype)
        view = result[self.y0:self.y1, self.x0:self.x1]
        view[self.mask0] = out[self.mask0]               # znane piksele zostają bit-w-bit
        return view_to_out(result, self._was_gray)


def view_to_out(result, was_gray):
    return result[..., 0] if was_gray else result


def inpaint(image, mask, patch_size: int = 9, scale_factor: float = 1.0,
            grow: int = 0, search_radius: int = 100, verbose: bool = True):
    """
    grow - o ile pikseli poszerzyć maskę w każdą stronę przed inpaintingiem.
    scale_factor < 1 - liczenie na zmniejszonym obrazie (zwykle niepotrzebne po optymalizacji).
    """
    mask = grow_mask(mask, grow)
    h_orig, w_orig = mask.shape

    if scale_factor < 1.0:
        new_w = max(10, int(w_orig * scale_factor))
        new_h = max(10, int(h_orig * scale_factor))
        image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask.astype(np.uint8) * 255, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST) > 127
        if verbose:
            print(f"Skalowanie w dół: z {w_orig}x{h_orig} do {new_w}x{new_h}", flush=True)
    else:
        image_small, mask_small = image, mask

    result_small = CriminisiInpainting(image_small, mask_small, patch_size,
                                       search_radius).run(verbose)

    if scale_factor < 1.0:
        return cv2.resize(result_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return result_small
