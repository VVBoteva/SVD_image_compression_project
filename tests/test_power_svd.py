import numpy as np
import pytest

from svdcomp.compress import low_rank_approx, numpy_svd
from svdcomp.power_svd import dominant_triplet, power_svd, power_svd_with_info


def matrix_with_spectrum(m, n, s, seed=0):
    """Random matrix with prescribed singular values s."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, len(s))))
    V, _ = np.linalg.qr(rng.standard_normal((n, len(s))))
    return (U * s) @ V.T


@pytest.mark.parametrize("deflation", ["orthogonal", "explicit"])
def test_singular_values_match_numpy(deflation):
    A = matrix_with_spectrum(60, 40, np.geomspace(10, 0.1, 20))
    _, s_ref, _ = numpy_svd(A, 8)
    _, s, _ = power_svd(A, 8, deflation=deflation)
    assert np.allclose(s, s_ref, rtol=1e-8)


def test_on_random_matrix_top_values():
    A = np.random.default_rng(3).random((50, 40))
    _, s_ref, _ = numpy_svd(A, 5)
    _, s, _ = power_svd(A, 5)
    assert np.allclose(s, s_ref, rtol=1e-6)


def test_vectors_orthonormal_and_reconstruction():
    A = matrix_with_spectrum(30, 25, np.geomspace(5, 0.5, 10), seed=1)
    U, s, Vt = power_svd(A, 6)
    assert np.allclose(U.T @ U, np.eye(6), atol=1e-8)
    assert np.allclose(Vt @ Vt.T, np.eye(6), atol=1e-8)
    assert np.allclose((U * s) @ Vt, low_rank_approx(A, 6), atol=1e-8)


def test_exact_low_rank_matrix():
    """Rank-2 matrix: the 3rd singular value must come out as zero."""
    A = matrix_with_spectrum(20, 20, np.array([3.0, 1.0]))
    _, s, _ = power_svd(A, 3)
    assert np.allclose(s, [3.0, 1.0, 0.0], atol=1e-8)


def test_convergence_speed_depends_on_gap():
    """Bigger sigma_2/sigma_1 ratio -> more iterations."""
    fast = matrix_with_spectrum(40, 40, np.array([10.0, 1.0, 0.5]))
    slow = matrix_with_spectrum(40, 40, np.array([10.0, 9.5, 0.5]))
    *_, it_fast, _ = dominant_triplet(fast, rng=np.random.default_rng(0))
    *_, it_slow, _ = dominant_triplet(slow, rng=np.random.default_rng(0))
    assert it_fast < it_slow


def test_info_and_validation():
    A = np.random.default_rng(0).random((10, 8))
    *_, info = power_svd_with_info(A, 3)
    assert len(info.iterations) == 3 and all(info.converged)
    with pytest.raises(ValueError):
        power_svd(A, 9)
    with pytest.raises(ValueError):
        power_svd(A, 2, deflation="nope")
