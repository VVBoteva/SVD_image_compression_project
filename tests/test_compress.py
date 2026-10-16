import numpy as np
import pytest

from svdcomp.compress import (break_even_rank, compress_image, compression_ratio,
                              energy_retained, low_rank_approx, numpy_svd,
                              rank_for_energy, storage_cost, theoretical_errors)

rng = np.random.default_rng(0)


def test_full_rank_reconstruction_is_exact():
    A = rng.random((20, 15))
    assert np.allclose(low_rank_approx(A, 15), A)


def test_rank_of_approximation():
    A = rng.random((30, 25))
    assert np.linalg.matrix_rank(low_rank_approx(A, 7)) == 7


@pytest.mark.parametrize("k", [1, 3, 10])
def test_eckart_young_error_identities(k):
    """||A-A_k||_2 = s_{k+1} and ||A-A_k||_F = sqrt(sum_{i>k} s_i^2)."""
    A = rng.random((40, 30))
    s = np.linalg.svd(A, compute_uv=False)
    E = A - low_rank_approx(A, k)
    spec, frob = theoretical_errors(s, k)
    assert np.isclose(np.linalg.norm(E, 2), spec)
    assert np.isclose(np.linalg.norm(E, "fro"), frob)


def test_truncated_svd_beats_random_rank_k_matrices():
    """Sanity check of optimality: no random rank-k matrix does better."""
    A = rng.random((25, 20))
    k = 4
    best = np.linalg.norm(A - low_rank_approx(A, k), "fro")
    for _ in range(200):
        B = rng.standard_normal((25, k)) @ rng.standard_normal((k, 20))
        assert np.linalg.norm(A - B, "fro") >= best


def test_rgb_compression_shape_and_range():
    img = rng.random((16, 20, 3))
    out = compress_image(img, 5)
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1


def test_storage_and_ratio():
    assert storage_cost(100, 200, 10) == 10 * 301
    assert storage_cost(100, 200, 10, channels=3) == 3 * 10 * 301
    assert np.isclose(compression_ratio(100, 200, 10), 20000 / 3010)
    with pytest.raises(ValueError):
        compression_ratio(10, 10, 0)


def test_break_even_rank():
    m, n = 512, 512
    k = break_even_rank(m, n)
    assert storage_cost(m, n, k) < m * n <= storage_cost(m, n, k + 1)


def test_energy():
    s = np.array([3.0, 2.0, 1.0])
    assert np.isclose(energy_retained(s, 1), 9 / 14)
    assert energy_retained(s, 3) == 1.0
    assert rank_for_energy(s, 0.9) == 2
    assert rank_for_energy(s, 1.0) == 3


def test_invalid_rank():
    with pytest.raises(ValueError):
        low_rank_approx(rng.random((5, 5)), 6)
    with pytest.raises(ValueError):
        low_rank_approx(rng.random((5, 5, 3)), 2)


def test_numpy_svd_shapes():
    U, s, Vt = numpy_svd(rng.random((12, 9)), 4)
    assert U.shape == (12, 4) and s.shape == (4,) and Vt.shape == (4, 9)
