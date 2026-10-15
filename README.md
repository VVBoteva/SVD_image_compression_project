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

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                                # run the tests
jupyter notebook notebook/svd_compression.ipynb
```
