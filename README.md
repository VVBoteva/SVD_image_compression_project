# Image Compression with the Singular Value Decomposition

*Python course final project.*

## Problem

A grayscale image of size m × n is a matrix with m·n entries. **Can we store far fewer numbers
and still reconstruct an image that looks almost the same? What is the best approximation for a
given storage budget, and how does it compare with JPEG-style compression?**

The answer comes from the **Singular Value Decomposition (SVD)** and the
**Eckart–Young–Mirsky theorem**. The theorem says that truncating the SVD to its k largest
singular values gives the *optimal* rank-k approximation of a matrix.

## What the project contains

| Part | Description |
|---|---|
| `notebook/svd_compression.ipynb` | **Main deliverable**: theory, proofs, experiments, conclusions |
| `svdcomp/` | Small Python package with the implementations |
| `tests/` | `pytest` test suite |
| `images/` | Test images |

Methods implemented and compared:

1. `numpy.linalg.svd`: the LAPACK reference
2. **Power iteration with deflation**: our own SVD, written from scratch
3. **Randomized SVD** (Halko, Martinsson & Tropp, 2011)
4. **Block DCT** (the transform used by JPEG) as a non-SVD baseline

## Main results

* **Eckart–Young verified.** The measured errors ‖A − A_k‖₂ and ‖A − A_k‖_F match σ_{k+1} and
  √(Σ_{i>k} σᵢ²) to about 10⁻¹⁴. No competing rank-k approximation beats the truncated SVD.
* **Compressibility = singular value decay.** A checkerboard is exact at k = 2 (64×). A photo reaches
  ≈ 29 dB PSNR at 5× and ≈ 33 dB at 2.6×. Random noise cannot be compressed at all.
* **Own SVD (power iteration)** matches NumPy to about 10⁻¹⁵. Its iteration count follows the
  predicted ln ε / (2 ln(σ_{j+1}/σ_j)), which makes it slow when singular values are close.
* **Randomized SVD** (p = 10, q = 2) comes within 1% of the optimal error and is the fastest method
  at every size tested. At 2048 × 2048 it is about 40× faster than LAPACK.
* **SVD vs DCT.** At an equal number of stored values, the thresholded block DCT gives 5–8 dB more PSNR on
  photos. Optimal among low-rank matrices does not mean optimal among all representations,
  which is why JPEG uses the DCT.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                                # run the tests
jupyter notebook notebook/svd_compression.ipynb
python images/make_images.py          # optional: regenerate the test images
```

Running the whole notebook takes about 2 minutes, most of it in the runtime benchmark (Experiment 8).

## Project structure

```
svdcomp/
  io.py              image loading/saving, float <-> uint8, channel handling
  compress.py        rank-k approximation, storage cost, compression ratio, Eckart-Young errors
  power_svd.py       SVD from scratch: power iteration + deflation
  randomized_svd.py  randomized SVD (Halko, Martinsson, Tropp)
  dct.py             8x8 block DCT compressor (zonal and threshold)
  metrics.py         MSE, PSNR, SSIM, matrix-norm errors
tests/               pytest suite for every module
images/              test images + script that generates them
notebook/            the main notebook
```

## References

* C. Eckart, G. Young, *The approximation of one matrix by another of lower rank*, Psychometrika, 1936.
* N. Halko, P.-G. Martinsson, J. A. Tropp, *Finding structure with randomness*, SIAM Review, 2011.
* L. N. Trefethen, D. Bau, *Numerical Linear Algebra*, SIAM, 1997.
* G. K. Wallace, *The JPEG still picture compression standard*, IEEE Trans. Consumer Electronics, 1992.
