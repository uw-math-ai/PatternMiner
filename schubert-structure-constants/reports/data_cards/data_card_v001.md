# Data card — v001

## Generation backend
- Backend: SageMath `SchubertPolynomialRing(ZZ)` v10.9 (`scripts/generate_products_sage.py`).
- Independent re-derivation (Monk's rule, Littlewood–Richardson, grading) via
  `schubml.combinatorics.validation` — see `tests/test_sage_products.py`.

## Mathematical conventions
- Permutations in one-line notation, 1-indexed; canonicalised by trimming trailing
  fixed points (`canonical_perm`). Length = #inversions. Bruhat order via rank matrices.

## Coverage
- Source products: S₃, S₄, S₅ exhaustive; S₆ sampled to 60,000 of 259,560 exhaustive
  pairs; S₇ sampled (8,000). Products of Sₙ reach S₁₁ (S₆) / S₁₃ (S₇), so `n_ctx=14`.
- **146,435** distinct permutations · **75,581** products · **1,592,553** triples
  (934,755 positive / 657,798 negative).
- Mean support size 12.37; max coefficient 6 (coeffs: 1→892,135, 2→39,357, 3→2,930,
  4→299, 5→31, 6→3).

## Negative sampling (hard zeros)
Tiers by # necessary conditions / positive-matching filters passed:
Z0 length-mismatch · Z1 fails Bruhat · Z2 grading+Bruhat · Z3 +descent/Rothe-shape
match · Z4 near-positive (length-preserving perturbation) · Z5 adversarial (exact
L/R descent sets + Rothe rows).

| tier | positive | Z0 | Z1 | Z2 | Z3 | Z4 | Z5 |
|------|---------:|---:|---:|---:|---:|---:|---:|
| count | 934,755 | 75,581 | 75,537 | 289,156 | 67,797 | 149,423 | 304 |

Hard zeros (Z2–Z5 = 506,680) dominate easy zeros (Z0+Z1 = 151,118); tier weights in
training further down-weight Z0/Z1 and up-weight Z3/Z4/Z5.

## Splits
- **product-disjoint** (no product pair across train/val/test): products 60,358 /
  7,630 / 7,593 → triples 1,272,767 / 160,079 / 159,707.
- **ambient-OOD**: train source S≤5, test S₆/S₇.
- **family holdouts**: grassmannian / boolean / vexillary.
- **hard-zero adversarial**: train Z0–Z3, test Z4/Z5.
- **coefficient-magnitude**: hold out products with max_coeff ≥ 6.

## Validation
`scripts/validate_dataset.py` → **ALL_OK**: 0 positives failing grading, 0 non-Z0
negatives failing grading, 0 negatives present in their product support, no
product-disjoint leakage.

## Known limitations
- S₆ is sampled (60k/259k); S₇/S₈ sparsely covered. High coefficients are scarce
  (max 6 — these small groups rarely produce large structure constants). Z5 is rare
  (304) — Z4 is the practical adversarial tier.
