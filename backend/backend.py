import numpy as np
from criminisi import inpaint


def remove_object(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 9,
    search_radius: int = 100
) -> np.ndarray:

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Obraz musi mieć 3 kanały RGB.")

    if image_rgb.dtype != np.uint8:
        raise ValueError("Obraz musi mieć typ uint8.")

    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("Maska i obraz muszą mieć ten sam rozmiar.")

    result = inpaint(
        image_rgb,
        mask.astype(bool),
        patch_size=patch_size,
        search_radius=search_radius,
        verbose=False
    )

    return result