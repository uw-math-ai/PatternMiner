"""Model components for the Schubert product-query transformer."""

from .product_query_transformer import ModelConfig, ProductQueryTransformer
from .losses import LossWeights, compute_losses
from .heads import coeff_to_bin, coeff_to_exact_small, N_BINS, MOD_PRIMES

__all__ = [
    "ModelConfig",
    "ProductQueryTransformer",
    "LossWeights",
    "compute_losses",
    "coeff_to_bin",
    "coeff_to_exact_small",
    "N_BINS",
    "MOD_PRIMES",
]
