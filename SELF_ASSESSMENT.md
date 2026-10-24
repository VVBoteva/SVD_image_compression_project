# Self-Assessment

**Project:** Image Compression with the Singular Value Decomposition
**Author:** Stanislav Botev
**Repository:** https://github.com/stakobotev/softuni_python_prj
**Main deliverable:** `notebook/svd_compression.ipynb`
**Additional deliverable:** `svdcomp/` — a small reusable Python library with a `pytest` test suite

---

## Summary

The project implements and analyses image compression via the truncated Singular Value
Decomposition. It is both an **algorithm implementation and analysis** project and a **math
explainer**: it proves the optimality of the truncated SVD (Eckart–Young–Mirsky), implements the
decomposition three ways (a from-scratch power iteration with deflation, a randomized SVD, and a
LAPACK reference), compares it against a block-DCT baseline (the idea behind JPEG), and validates
everything with eight experiments on structured, photographic, and noisy images.

All code runs without errors (`pytest`: 42/42 passing; the notebook re-executes end-to-end with no
error outputs). Sources are credited in the References section.

---

## Minimum criteria (fail-gates)

| Criterion | Status | Evidence |
|---|---|---|
| Legal / non-malicious | Met | Pure numerical linear algebra; nothing that affects another machine |
| Functional code | Met | `pytest` → 42/42; notebook re-executes with 0 error outputs |
| Deliverables submitted on time | Met | English Jupyter notebook + public GitHub repository link |
| ≥ 10 meaningful commits | Met | 14 commits, each scoped to one module or one notebook section |
| Academic integrity / sources credited | Met | Section 8 (References); test-image sources credited |

---

## Grading criteria (self-scored)

### 1. Problem understanding, formulation, and significance — **14 / 15**
Section 1 states the problem formally (find the closest rank-constrained approximation under a
storage budget), poses three explicit research questions, and defines "closeness" precisely via the
Frobenius and spectral norms. The significance (image redundancy, why low rank is plausible) is
motivated before any math. The real-world→mathematical translation (image = matrix) is correct and
stated explicitly.

### 2. Writing layout — **18 / 20**
Eight numbered sections with a table of contents, progressing theory → storage analysis → algorithms
→ experiments → conclusions → references. Methods and experiments receive the most space, as the
guidelines emphasise. Each experiment has a stated question, code, figures, and a written takeaway.
All math is typeset in LaTeX.

### 3. Mathematical understanding and research — **14 / 15**
Full statement and **proof** of the Eckart–Young–Mirsky theorem in both the spectral and Frobenius
norms, using Weyl's inequality for singular values as a lemma. The SVD, low-rank approximation,
storage/break-even analysis, power-iteration convergence rate, and the randomized-SVD error bound
are all derived or explained. Adjacent methods (DCT) are laid out for comparison. Six primary sources
are cited (Eckart–Young, Mirsky, Halko–Martinsson–Tropp, Trefethen–Bau, Strang, Wallace/JPEG).

### 4. Code quality — **14 / 15**
Clear separation of concerns across `svdcomp/` (I/O, compression, two SVD algorithms, DCT baseline,
metrics). Every SVD routine follows a single interchangeable interface `svd_fn(A, k) -> (U, s, Vt)`.
The code is documented, covered by a per-module `pytest` suite (42 tests), and depends only on a
minimal, fully-used set of libraries (verified: no undeclared and no unused dependencies).

### 5. Methods; data handling / analysis / processing — **19 / 20**
Eight experiments: singular-value decay vs compressibility, visual rank-k reconstructions, numerical
verification of Eckart–Young to machine precision, SVD vs block-DCT quality per stored number,
structure vs noise, convergence of the power iteration, accuracy of the randomized SVD, and running
time. Multiple methods are compared (own power iteration, randomized SVD, LAPACK, DCT) and validated
against NumPy. Quality is measured with MSE, PSNR, and SSIM. Assumptions and limitations (we count
stored numbers, not quantised bits) are stated.

### 6. Communication — **14 / 15**
The conclusion answers each of the three questions from Section 1 directly, reports an honest
**negative** result (the DCT beats the SVD on photographs, with the reason why), and lists four
concrete extensions (coefficient quantisation, YCbCr chroma subsampling, block-wise SVD,
Lanczos/Krylov solvers). The reusable `svdcomp` library is presented as a secondary deliverable.

**Suggested total: ~93 / 100.**

---

## Known limitations (stated honestly)

- Storage is measured in stored numbers, not quantised/entropy-coded bits, so the comparison with
  real codecs (JPEG) is about the transform, not the full pipeline.
- RGB channels are compressed independently, ignoring inter-channel correlation.
- The from-scratch power iteration is slow on images with clustered singular values; faster subspace
  or Krylov methods are noted as future work.
