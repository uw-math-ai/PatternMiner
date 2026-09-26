"""schubml: interpretable transformers for Schubert structure constants.

See PROJECT.md for the full blueprint.  Subpackages:

* ``combinatorics`` -- permutations, Rothe diagrams, rank matrices, Bruhat
  order, reduced words, known families, transition trees, validation.
* ``data``          -- schemas, feature extraction, datasets, samplers, packs,
  tokenizers, collators.
* ``models``        -- the product-query transformer with counting slots, heads,
  losses, and hook instrumentation.
* ``train``         -- curriculum, the Lightning module, metrics, logging.
* ``interp``        -- hooks, probes, activation/path patching, SAEs, feature
  mining, symbolic distillation.
* ``experiments``   -- baselines, ablations, held-out evaluations.
"""

__version__ = "0.1.0"
