"""Loading, saving and converting images.

Every function in this package works with images stored as ``float64`` NumPy
arrays with values in ``[0, 1]``:

* grayscale image -> shape ``(m, n)``
* colour image    -> shape ``(m, n, 3)``  (one matrix per RGB channel)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image file as a float array in [0, 1].

    Parameters
    ----------
    path:
        Path to an image file (PNG, JPEG, ...).
    grayscale:
        If True, convert to a single luminance channel.
    """
    img = Image.open(path)
    img = img.convert("L" if grayscale else "RGB")
    return to_float(np.asarray(img))


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save a float image (values are clipped to [0, 1]) as an 8-bit file."""
    Image.fromarray(to_uint8(array)).save(path)


def to_float(array: np.ndarray) -> np.ndarray:
    """Convert an 8-bit image to float64 in [0, 1]; float input is returned as a copy."""
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float64) / 255.0
    return np.asarray(array, dtype=np.float64).copy()


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8, clipping values outside the range."""
    return (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def rgb_to_gray(array: np.ndarray) -> np.ndarray:
    """Luminance (ITU-R BT.601) of an RGB float image."""
    if array.ndim == 2:
        return array.copy()
    weights = np.array([0.299, 0.587, 0.114])
    return array[..., :3] @ weights


def split_channels(array: np.ndarray) -> list[np.ndarray]:
    """Return the list of 2-D channel matrices of an image."""
    if array.ndim == 2:
        return [array]
    return [array[..., c] for c in range(array.shape[2])]


def merge_channels(channels: list[np.ndarray]) -> np.ndarray:
    """Inverse of :func:`split_channels`."""
    if len(channels) == 1:
        return channels[0]
    return np.stack(channels, axis=-1)
