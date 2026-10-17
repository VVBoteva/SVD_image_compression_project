import numpy as np
import pytest

from svdcomp.dct import (block_dct, block_idct, dct_compress_image, dct_compress_threshold,
                         dct_compress_zonal, n_blocks, zigzag_order)

rng = np.random.default_rng(0)


def test_zigzag_start():
    zz = zigzag_order(8)
    assert zz[:6].tolist() == [[0, 0], [0, 1], [1, 0], [2, 0], [1, 1], [0, 2]]
    assert len({tuple(p) for p in zz}) == 64


@pytest.mark.parametrize("shape", [(16, 16), (21, 30)])
def test_block_dct_roundtrip(shape):
    A = rng.random(shape)
    coeffs, s = block_dct(A)
    assert np.allclose(block_idct(coeffs, s), A)


def test_dct_is_orthonormal_energy_preserving():
    A = rng.random((16, 24))
    coeffs, _ = block_dct(A)
    assert np.isclose(np.sum(coeffs ** 2), np.sum(A ** 2))


def test_constant_block_needs_one_coefficient():
    A = np.full((16, 16), 0.7)
    approx, stored = dct_compress_zonal(A, 1)
    assert np.allclose(approx, A)
    assert stored == 4


def test_full_zonal_is_lossless_and_error_decreases():
    A = rng.random((24, 24))
    assert np.allclose(dct_compress_zonal(A, 64)[0], A)
    errs = [np.linalg.norm(A - dct_compress_zonal(A, c)[0]) for c in (1, 8, 32)]
    assert errs[0] > errs[1] > errs[2]


def test_threshold_storage_and_exactness():
    A = rng.random((16, 16))
    approx, stored = dct_compress_threshold(A, 256)
    assert stored == 512 and np.allclose(approx, A)


def test_image_wrapper():
    img = rng.random((20, 20, 3))
    out, total = dct_compress_image(img, budget=100, mode="threshold")
    assert out.shape == img.shape and total == 3 * 100
    out, total = dct_compress_image(img, budget=90, mode="zonal")
    assert total == 3 * 10 * n_blocks(20, 20)
    with pytest.raises(ValueError):
        dct_compress_image(img, 10, mode="x")
