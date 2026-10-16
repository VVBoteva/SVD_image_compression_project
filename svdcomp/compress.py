r"""Rank-k approximation of matrices and images.

Every SVD routine in this package follows the same *truncated SVD* interface::

    U, s, Vt = svd_fn(A, k)

with ``U`` of shape ``(m, k)``, ``s`` of shape ``(k,)`` (descending) and
``Vt`` of shape ``(k, n)`` so that ``A_k = U @ diag(s) @ Vt``.

Storage model: a rank-k approximation of an m x n matrix needs
``k`` left vectors (``k*m`` numbers), ``k`` right vectors (``k*n``) and ``k``
singular values, i.e. ``k (m + n + 1)`` numbers instead of ``m n``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .io import merge_channels, split_channels

SVDFunction = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]]


def numpy_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD computed with LAPACK (``numpy.linalg.svd``): the reference method."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


def reconstruct(U: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Return ``U diag(s) Vt`` (sum of rank-one terms sigma_i u_i v_i^T)."""
    return (U * s) @ Vt


def low_rank_approx(A: np.ndarray, k: int, svd_fn: SVDFunction = numpy_svd) -> np.ndarray:
    """Best rank-k approximation A_k of a 2-D matrix (Eckart-Young)."""
    _check_rank(A, k)
    return reconstruct(*svd_fn(A, k))


def compress_image(image: np.ndarray, k: int, svd_fn: SVDFunction = numpy_svd,
                   clip: bool = True) -> np.ndarray:
    """Compress a grayscale or RGB image by keeping k singular triplets per channel."""
    channels = [low_rank_approx(c, k, svd_fn) for c in split_channels(image)]
    out = merge_channels(channels)
    return np.clip(out, 0.0, 1.0) if clip else out


def storage_cost(m: int, n: int, k: int, channels: int = 1) -> int:
    """Number of stored values for a rank-k approximation: ``channels * k (m + n + 1)``."""
    return channels * k * (m + n + 1)


def compression_ratio(m: int, n: int, k: int) -> float:
    """Original size divided by compressed size, ``m n / (k (m + n + 1))``.

    Values > 1 mean the approximation is actually smaller than the image.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return (m * n) / (k * (m + n + 1))


def break_even_rank(m: int, n: int) -> int:
    """Largest k for which storing the rank-k factors is still cheaper than ``m n``."""
    return (m * n - 1) // (m + n + 1)


def energy_retained(s: np.ndarray, k: int) -> float:
    r"""Fraction of the squared Frobenius norm kept: sum_{i<=k} s_i^2 / sum_i s_i^2."""
    s2 = np.asarray(s, dtype=float) ** 2
    return float(s2[:k].sum() / s2.sum())


def theoretical_errors(s: np.ndarray, k: int) -> tuple[float, float]:
    r"""Errors predicted by Eckart-Young-Mirsky for the truncated SVD.

    Returns ``(spectral, frobenius)`` where::

        ||A - A_k||_2 = sigma_{k+1}
        ||A - A_k||_F = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)
    """
    s = np.asarray(s, dtype=float)
    tail = s[k:]
    spectral = float(tail[0]) if tail.size else 0.0
    return spectral, float(np.sqrt(np.sum(tail ** 2)))


def rank_for_energy(s: np.ndarray, energy: float) -> int:
    """Smallest k whose truncated SVD retains at least ``energy`` (in (0, 1])."""
    s2 = np.asarray(s, dtype=float) ** 2
    cumulative = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cumulative, energy - 1e-12) + 1)


def _check_rank(A: np.ndarray, k: int) -> None:
    if A.ndim != 2:
        raise ValueError("expected a 2-D matrix; use compress_image for RGB images")
    if not 1 <= k <= min(A.shape):
        raise ValueError(f"k must be in [1, {min(A.shape)}], got {k}")
