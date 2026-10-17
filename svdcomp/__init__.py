"""svdcomp - image compression with the Singular Value Decomposition."""

from .compress import (break_even_rank, compress_image, compression_ratio, energy_retained,
                       low_rank_approx, numpy_svd, rank_for_energy, storage_cost,
                       theoretical_errors)
from .dct import dct_compress_image
from .power_svd import power_svd, power_svd_with_info
from .randomized_svd import randomized_svd

__version__ = "0.1.0"

__all__ = [
    "break_even_rank", "compress_image", "compression_ratio", "dct_compress_image",
    "energy_retained", "low_rank_approx", "numpy_svd", "power_svd", "power_svd_with_info",
    "randomized_svd", "rank_for_energy", "storage_cost", "theoretical_errors",
]
