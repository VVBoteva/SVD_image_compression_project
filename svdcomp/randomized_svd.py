r"""Randomized truncated SVD (Halko, Martinsson & Tropp, 2011).

Algorithm
---------
1. Draw a Gaussian test matrix ``Omega`` of shape ``(n, k + p)``
   (``p`` = oversampling, typically 5-10).
2. Sample the range of A: ``Y = A Omega``. Optionally apply ``q`` power (subspace)
   iterations ``Y = (A A^T)^q A Omega`` to sharpen the spectrum when the singular
   values decay slowly. We re-orthonormalise between steps for stability.
3. Orthonormal basis ``Q`` of ``Y`` (QR decomposition): ``A ~ Q Q^T A``.
4. Small SVD of ``B = Q^T A`` (size ``(k+p) x n``): ``B = U_B S V^T``.
5. ``U = Q U_B``; keep the first k triplets.

Expected error bound (Halko et al., Thm. 10.5, q = 0, k >= 2, p >= 2)::

    E ||A - Q Q^T A||_F <= (1 + k/(p-1))^{1/2} (sum_{j>k} sigma_j^2)^{1/2}

i.e. within a small constant factor of the optimal Eckart-Young error. Note that
``Q Q^T A`` has rank ``k + p``; the rank-k truncation returned here is slightly worse.
Cost: ``O(mn(k+p))`` instead of ``O(mn min(m,n))`` for a full SVD.
"""

from __future__ import annotations

import numpy as np


def randomized_svd(A: np.ndarray, k: int, oversampling: int = 10, power_iters: int = 2,
                   seed: int | None = 0):
    """Truncated SVD ``(U, s, Vt)`` using random projections."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if not 1 <= k <= min(m, n):
        raise ValueError(f"k must be in [1, {min(m, n)}], got {k}")
    if oversampling < 0 or power_iters < 0:
        raise ValueError("oversampling and power_iters must be non-negative")

    rng = np.random.default_rng(seed)
    ell = min(k + oversampling, min(m, n))

    Omega = rng.standard_normal((n, ell))
    Q, _ = np.linalg.qr(A @ Omega)
    for _ in range(power_iters):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)

    B = Q.T @ A
    U_B, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    return U[:, :k], s[:k], Vt[:k, :]
