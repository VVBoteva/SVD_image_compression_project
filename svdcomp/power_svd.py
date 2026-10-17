r"""Truncated SVD from scratch: power iteration with deflation.

Idea
----
For ``B = A^T A`` (symmetric positive semi-definite, eigenvalues ``sigma_i^2``)
the power iteration

    v_{t+1} = B v_t / ||B v_t||

converges to the top right singular vector ``v_1`` as long as ``v_0`` has a
component along ``v_1`` and ``sigma_1 > sigma_2``. The error shrinks like
``(sigma_2 / sigma_1)^(2t)``. Then ``sigma_1 = ||A v_1||`` and
``u_1 = A v_1 / sigma_1``.

``B`` is never formed: we compute ``A^T (A v)``, which costs ``O(mn)`` per step.

To get the next triplet we *deflate*: remove what is already found.

* ``"explicit"``: replace ``A`` by ``A - sigma_1 u_1 v_1^T`` (Hotelling deflation).
  Its top singular triplet is ``(sigma_2, u_2, v_2)``.
* ``"orthogonal"`` (default): keep ``A`` and orthogonalise each iterate against the
  right singular vectors found so far. This is mathematically the same, but it
  does not accumulate the rounding errors of the subtracted matrices and it
  never modifies ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PowerSVDInfo:
    """Diagnostics returned by :func:`power_svd_with_info`."""

    iterations: list[int] = field(default_factory=list)  # iterations used per triplet
    converged: list[bool] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        return int(sum(self.iterations))


def dominant_triplet(A: np.ndarray, tol: float = 1e-10, max_iter: int = 5000,
                     v0: np.ndarray | None = None, V_prev: np.ndarray | None = None,
                     rng: np.random.Generator | None = None,
                     history: list[float] | None = None):
    """Largest singular triplet ``(u, sigma, v)`` of ``A`` by power iteration.

    Parameters
    ----------
    tol:
        Stop when ``||v_{t+1} - v_t|| < tol``.
    V_prev:
        ``(n, j)`` matrix of already-found right singular vectors; iterates are kept
        orthogonal to them (orthogonal deflation). ``None`` = no constraint.
    history:
        If given, the estimate of sigma at every iteration is appended to it.

    Returns
    -------
    u, sigma, v, n_iter, converged
    """
    rng = np.random.default_rng() if rng is None else rng
    n = A.shape[1]
    v = rng.standard_normal(n) if v0 is None else np.array(v0, dtype=float)
    v = _project_out(v, V_prev)
    v /= np.linalg.norm(v)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Av = A @ v
        w = _project_out(A.T @ Av, V_prev)
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:  # A v = 0: the remaining singular values are zero
            converged = True
            break
        v_new = w / norm_w
        if history is not None:
            history.append(float(np.linalg.norm(A @ v_new)))
        delta = np.linalg.norm(v_new - v)
        v = v_new
        if delta < tol:
            converged = True
            break

    Av = A @ v
    sigma = float(np.linalg.norm(Av))
    u = Av / sigma if sigma > 0 else np.zeros(A.shape[0])
    return u, sigma, v, it, converged


def power_svd_with_info(A: np.ndarray, k: int, tol: float = 1e-10, max_iter: int = 5000,
                        deflation: str = "orthogonal", seed: int | None = 0):
    """Truncated SVD by power iteration. Returns ``(U, s, Vt, info)``."""
    if deflation not in ("orthogonal", "explicit"):
        raise ValueError("deflation must be 'orthogonal' or 'explicit'")
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if not 1 <= k <= min(m, n):
        raise ValueError(f"k must be in [1, {min(m, n)}], got {k}")

    rng = np.random.default_rng(seed)
    U = np.zeros((m, k))
    s = np.zeros(k)
    V = np.zeros((n, k))
    info = PowerSVDInfo()
    work = A.copy() if deflation == "explicit" else A

    for j in range(k):
        V_prev = V[:, :j] if deflation == "orthogonal" and j > 0 else None
        u, sigma, v, it, ok = dominant_triplet(work, tol, max_iter, V_prev=V_prev, rng=rng)
        U[:, j], s[j], V[:, j] = u, sigma, v
        info.iterations.append(it)
        info.converged.append(ok)
        if deflation == "explicit":
            work -= sigma * np.outer(u, v)

    # Deflation finds the triplets in descending order in exact arithmetic;
    # sort anyway to guard against tiny rounding effects.
    order = np.argsort(s)[::-1]
    return U[:, order], s[order], V[:, order].T, info


def power_svd(A: np.ndarray, k: int, tol: float = 1e-10, max_iter: int = 5000,
              deflation: str = "orthogonal", seed: int | None = 0):
    """Truncated SVD ``(U, s, Vt)`` by power iteration with deflation."""
    U, s, Vt, _ = power_svd_with_info(A, k, tol, max_iter, deflation, seed)
    return U, s, Vt


def _project_out(x: np.ndarray, V: np.ndarray | None) -> np.ndarray:
    """Remove the components of x along the orthonormal columns of V (applied twice
    for numerical stability, "twice is enough")."""
    if V is None or V.shape[1] == 0:
        return x
    x = x - V @ (V.T @ x)
    return x - V @ (V.T @ x)
