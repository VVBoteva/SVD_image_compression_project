import numpy as np

from svdcomp.metrics import (frobenius_error, mse, psnr, relative_frobenius_error,
                             spectral_error, ssim)

rng = np.random.default_rng(0)


def test_identical_images():
    a = rng.random((32, 32))
    assert mse(a, a) == 0
    assert psnr(a, a) == float("inf")
    assert np.isclose(ssim(a, a), 1.0)
    assert relative_frobenius_error(a, a) == 0


def test_psnr_known_value():
    a = np.zeros((10, 10))
    b = np.full((10, 10), 0.1)  # MSE = 0.01  ->  PSNR = 20 dB
    assert np.isclose(psnr(a, b), 20.0)


def test_ssim_rgb_and_degradation():
    a = rng.random((32, 32, 3))
    noisy = np.clip(a + 0.3 * rng.standard_normal(a.shape), 0, 1)
    assert ssim(a, noisy) < 0.9


def test_norm_errors():
    a = np.eye(3)
    b = np.zeros((3, 3))
    assert np.isclose(frobenius_error(a, b), np.sqrt(3))
    assert np.isclose(spectral_error(a, b), 1.0)
