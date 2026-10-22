"""Generate the test images used in the notebook.

Natural images come from ``skimage.data`` (public-domain or freely licensed
sample images bundled with scikit-image). Synthetic images are generated here.

Run from the repository root:  python images/make_images.py
"""

from pathlib import Path

import numpy as np
from skimage import data

from svdcomp.io import save_image, to_float

OUT = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(42)

    # Natural images
    save_image(to_float(data.camera()), OUT / "cameraman.png")        # 512x512 gray photo
    save_image(to_float(data.astronaut()), OUT / "astronaut.png")     # 512x512 RGB photo
    save_image(to_float(data.text()), OUT / "text.png")               # 172x448 gray text

    # Synthetic: smooth gradient + circles (a low-complexity "cartoon")
    m, n = 256, 256
    y, x = np.mgrid[0:m, 0:n] / 255.0
    cartoon = 0.3 + 0.4 * x
    for cy, cx, r, v in [(0.3, 0.3, 0.15, 0.95), (0.65, 0.6, 0.22, 0.1), (0.25, 0.75, 0.1, 0.6)]:
        cartoon[(x - cx) ** 2 + (y - cy) ** 2 < r ** 2] = v
    save_image(cartoon, OUT / "cartoon.png")

    # Synthetic: checkerboard (exactly rank 2)
    checker = ((np.arange(m)[:, None] // 32 + np.arange(n)[None, :] // 32) % 2).astype(float)
    save_image(checker, OUT / "checkerboard.png")

    # Synthetic: uniform noise (no structure, should be incompressible)
    save_image(rng.random((m, n)), OUT / "noise.png")

    print("Images written to", OUT)


if __name__ == "__main__":
    main()
