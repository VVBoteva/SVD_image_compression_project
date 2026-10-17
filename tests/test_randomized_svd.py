import numpy as np
import pytest

from svdcomp.compress import numpy_svd, theoretical_errors
from svdcomp.randomized_svd import randomized_svd
from tests.test_power_svd import matrix_with_spectrum


def test_exact_on_low_rank_matrix():
    A = matrix_with_spectrum(80, 60, np.array([5.0, 3.0, 2.0, 1.0]))
    U, s, Vt = randomized_svd(A, 4, power_iters=0)
    assert np.allclose(s, [5, 3, 2, 1])
    assert np.allclose((U * s) @ Vt, A)


def test_matches_numpy_with_fast_decay():
    A = matrix_with_spectrum(100, 80, np.geomspace(100, 1e-3, 60))
    _, s_ref, _ = numpy_svd(A, 10)
    _, s, _ = randomized_svd(A, 10)
    assert np.allclose(s, s_ref, rtol=1e-6)


def test_power_iterations_improve_accuracy():
    """Slowly decaying spectrum: q > 0 brings the error close to the optimum."""
    A = matrix_with_spectrum(200, 150, 1.0 / np.arange(1, 151) ** 0.5, seed=2)
    k = 10
    s_all = np.linalg.svd(A, compute_uv=False)
    optimum = theoretical_errors(s_all, k)[1]
    errs = []
    for q in (0, 3):
        U, s, Vt = randomized_svd(A, k, oversampling=5, power_iters=q, seed=1)
        errs.append(np.linalg.norm(A - (U * s) @ Vt))
    assert errs[1] < errs[0]
    assert errs[1] < 1.05 * optimum
    assert all(e >= optimum - 1e-10 for e in errs)  # can never beat Eckart-Young


def test_orthonormal_factors():
    A = np.random.default_rng(0).random((40, 30))
    U, _, Vt = randomized_svd(A, 5)
    assert np.allclose(U.T @ U, np.eye(5))
    assert np.allclose(Vt @ Vt.T, np.eye(5))


def test_validation():
    A = np.ones((5, 4))
    with pytest.raises(ValueError):
        randomized_svd(A, 5)
    with pytest.raises(ValueError):
        randomized_svd(A, 2, oversampling=-1)
