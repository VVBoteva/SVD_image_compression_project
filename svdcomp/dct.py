r"""Block-DCT compression: a simplified version of the JPEG idea, used as a baseline.

The image is cut into ``b x b`` blocks (``b = 8`` as in JPEG). Each block is
transformed by the orthonormal 2-D Discrete Cosine Transform (type II). The DCT
basis is *fixed* (cosines of increasing frequency), unlike the SVD basis, which
is computed from the image itself. Natural images put most of their energy in
the low frequencies, so most coefficients can be dropped.

Two strategies are provided:

* **zonal**: keep the first ``c`` coefficients of every block in JPEG zig-zag
  order (lowest frequencies). Positions are implicit, so the cost is
  ``c`` values per block.
* **threshold**: keep the ``N`` largest-magnitude coefficients in the whole
  image. This adapts better to content, but each kept value also needs its
  position, so we count **2 numbers per coefficient** (value + index).

Real JPEG also quantises the coefficients and entropy-codes them; that is
not modelled here. We compare *numbers stored* against the SVD's
``k (m + n + 1)``.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from .io import merge_channels, split_channels


def zigzag_order(b: int) -> np.ndarray:
    """Indices ``(row, col)`` of a ``b x b`` block in JPEG zig-zag order, shape ``(b*b, 2)``."""
    idx = [(i, j) for i in range(b) for j in range(b)]
    # Sort by anti-diagonal; alternate the direction on odd/even diagonals.
    idx.sort(key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))
    return np.array(idx)


def _to_blocks(A: np.ndarray, b: int) -> tuple[np.ndarray, tuple[int, int]]:
    m, n = A.shape
    pm, pn = (-m) % b, (-n) % b
    P = np.pad(A, ((0, pm), (0, pn)), mode="edge")
    M, N = P.shape
    blocks = P.reshape(M // b, b, N // b, b).swapaxes(1, 2)  # (rows, cols, b, b)
    return blocks, (m, n)


def _from_blocks(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    R, C, b, _ = blocks.shape
    P = blocks.swapaxes(1, 2).reshape(R * b, C * b)
    return P[: shape[0], : shape[1]]


def block_dct(A: np.ndarray, b: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Orthonormal 2-D DCT of every ``b x b`` block. Returns ``(coeffs, original_shape)``."""
    blocks, shape = _to_blocks(np.asarray(A, dtype=float), b)
    return dctn(blocks, axes=(2, 3), norm="ortho"), shape


def block_idct(coeffs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`block_dct`."""
    return _from_blocks(idctn(coeffs, axes=(2, 3), norm="ortho"), shape)


def n_blocks(m: int, n: int, b: int = 8) -> int:
    return (-(-m // b)) * (-(-n // b))


def dct_compress_zonal(A: np.ndarray, c: int, b: int = 8) -> tuple[np.ndarray, int]:
    """Keep the ``c`` lowest-frequency (zig-zag) coefficients per block.

    Returns ``(approximation, stored_values)``.
    """
    if not 1 <= c <= b * b:
        raise ValueError(f"c must be in [1, {b * b}]")
    coeffs, shape = block_dct(A, b)
    mask = np.zeros((b, b), dtype=bool)
    zz = zigzag_order(b)[:c]
    mask[zz[:, 0], zz[:, 1]] = True
    coeffs = coeffs * mask
    stored = c * coeffs.shape[0] * coeffs.shape[1]
    return block_idct(coeffs, shape), stored


def dct_compress_threshold(A: np.ndarray, n_keep: int, b: int = 8) -> tuple[np.ndarray, int]:
    """Keep the ``n_keep`` largest-magnitude block-DCT coefficients of the whole image.

    Returns ``(approximation, stored_values)`` with ``stored_values = 2 * n_keep``.
    """
    coeffs, shape = block_dct(A, b)
    flat = coeffs.ravel()
    n_keep = int(np.clip(n_keep, 1, flat.size))
    keep = np.argpartition(np.abs(flat), -n_keep)[-n_keep:]
    sparse = np.zeros_like(flat)
    sparse[keep] = flat[keep]
    return block_idct(sparse.reshape(coeffs.shape), shape), 2 * n_keep


def dct_compress_image(image: np.ndarray, budget: int, mode: str = "threshold",
                       b: int = 8, clip: bool = True) -> tuple[np.ndarray, int]:
    """Compress a gray/RGB image with roughly ``budget`` stored values **per channel**.

    ``mode="threshold"``: keep ``budget // 2`` coefficients per channel.
    ``mode="zonal"``: use the largest ``c`` with ``c * blocks <= budget`` (at least 1,
    so a budget below one value per block is exceeded; the true cost is returned).
    Returns ``(approximation, total_stored_values)``.
    """
    outs, total = [], 0
    for ch in split_channels(image):
        if mode == "threshold":
            approx, stored = dct_compress_threshold(ch, max(budget // 2, 1), b)
        elif mode == "zonal":
            c = int(np.clip(budget // n_blocks(*ch.shape, b), 1, b * b))
            approx, stored = dct_compress_zonal(ch, c, b)
        else:
            raise ValueError("mode must be 'threshold' or 'zonal'")
        outs.append(approx)
        total += stored
    out = merge_channels(outs)
    return (np.clip(out, 0.0, 1.0) if clip else out), total
