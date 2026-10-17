"""Quality metrics for comparing an image with its compressed version."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(original: np.ndarray, approx: np.ndarray) -> float:
    """Mean squared error per pixel value."""
    return float(np.mean((original - approx) ** 2))


def psnr(original: np.ndarray, approx: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB: 10 log10(R^2 / MSE). Higher is better.

    Rough guide for 8-bit images: < 25 dB visibly degraded, 30-40 dB good, > 40 dB
    nearly indistinguishable.
    """
    err = mse(original, approx)
    if err == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / err))


def ssim(original: np.ndarray, approx: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity index in [-1, 1] (1 = identical); handles gray and RGB."""
    channel_axis = -1 if original.ndim == 3 else None
    return float(structural_similarity(original, approx, data_range=data_range,
                                       channel_axis=channel_axis))


def relative_frobenius_error(original: np.ndarray, approx: np.ndarray) -> float:
    """||A - B||_F / ||A||_F."""
    return float(np.linalg.norm(original - approx) / np.linalg.norm(original))


def frobenius_error(original: np.ndarray, approx: np.ndarray) -> float:
    """||A - B||_F (for RGB the norm runs over all three channels)."""
    return float(np.linalg.norm(original - approx))


def spectral_error(original: np.ndarray, approx: np.ndarray) -> float:
    """||A - B||_2: largest singular value of the error matrix (2-D only)."""
    return float(np.linalg.norm(original - approx, 2))
