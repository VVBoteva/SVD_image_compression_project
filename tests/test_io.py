import numpy as np

from svdcomp.io import (load_image, merge_channels, rgb_to_gray, save_image,
                        split_channels, to_float, to_uint8)


def test_float_uint8_roundtrip():
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(to_uint8(to_float(a)), a)


def test_to_uint8_clips():
    out = to_uint8(np.array([-0.5, 0.0, 1.0, 2.0]))
    assert out.tolist() == [0, 0, 255, 255]


def test_save_and_load(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.random((10, 12, 3))
    path = tmp_path / "x.png"
    save_image(img, path)
    back = load_image(path)
    assert back.shape == img.shape
    assert np.max(np.abs(back - img)) <= 0.5 / 255 + 1e-12


def test_gray_conversion_of_gray_pixel():
    img = np.full((4, 4, 3), 0.4)
    assert np.allclose(rgb_to_gray(img), 0.4)


def test_split_merge_roundtrip():
    img = np.random.default_rng(1).random((5, 6, 3))
    assert np.array_equal(merge_channels(split_channels(img)), img)
    gray = img[..., 0]
    assert np.array_equal(merge_channels(split_channels(gray)), gray)
