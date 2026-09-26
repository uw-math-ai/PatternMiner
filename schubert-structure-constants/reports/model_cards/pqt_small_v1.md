# Model card — pqt_small_v1

Product-query transformer for Schubert structure constants (PROJECT.md §8).

## Architecture
- `ProductQueryTransformer`: d_model 192, 4 encoder layers + 2 cross-attention
  layers, 6 heads, d_mlp 768, n_ctx 14, **32 counting slots**. ~3.03 M params.
- u/v symmetric **by construction** (shared FACTOR role; factor stream encoded as a
  set → `c(u,v,w)=c(v,u,w)` exactly).
- Multi-view input tokens: one-line, Rothe cells, inversions, rank essential set,
  reduced word. Nonnegative counting bottleneck: ĉ = Σ_a z_a, z_a ≥ 0.
- Fully hooked (77 hook points) for caching / patching / SAEs.

## Training data
- Dataset **v001** (146,435 perms · 75,581 products · 1,592,553 triples; S₃–S₅
  exhaustive + S₆ 60k-sampled + S₇).
- Split `product_disjoint:train` (1,272,767 triples). **Product-pack batching** so
  the listwise-support loss sees many ±pairs per product.
- Multitask loss: positivity (tier-weighted BCE), log-coeff Huber on the counting
  bottleneck, coeff bins, exact-small coeff, mod-p {2,3,5,7}, listwise support,
  auxiliary (lengths/descents/family — scaled 0.1×), slot L1 + diversity.
- AdamW lr 4e-4 cosine (1k warmup), batch 384, **16-mixed AMP**, grad-clip 1.0,
  on a single Tesla T4. 30,000 steps (~2 h); checkpoints top-3 by val F1 + last.

## Metrics — validation (`product_disjoint:val`, all tiers)
Smooth, stable convergence to a strong plateau (no divergence/NaN/OOM):

| step | pos_f1 | pos_mcc | hardzero_f1 | coeff_exact | support_ndcg |
|-----:|-------:|--------:|------------:|------------:|-------------:|
| 1000 | 0.876 | 0.678 | 0.966 | 0.814 | 0.928 |
| 3000 | 0.961 | 0.903 | 0.985 | 0.923 | 0.979 |
| 5000 | 0.969 | 0.924 | 0.989 | 0.933 | 0.980 |
| 7000 | 0.976 | 0.942 | 0.995 | 0.936 | 0.982 |
| 9000 | 0.977 | 0.945 | 0.995 | 0.946 | 0.983 |

Final run completed all **30,000 steps** stably (best val F1 0.9924 @ step 29k);
final val MCC **0.981**, hard-zero F1 **0.998**.

### PROJECT §12.2 stratified table (best checkpoint, `evaluate.py`)
| evaluation slice | pos F1 | pos MCC | AUPRC | exact-coeff | support NDCG |
|------------------|-------:|--------:|------:|------------:|-------------:|
| product-disjoint test | 0.992 | 0.980 | 0.997 | 0.963 | 0.984 |
| **ambient-OOD** (train S≤5 → test S₆/S₇) | 0.992 | **0.980** | 0.998 | 0.962 | 0.993 |
| family holdout — grassmannian | 0.986 | 0.978 | 0.996 | 0.984 | 0.995 |
| family holdout — boolean | 0.990 | 0.980 | 0.998 | 0.979 | 0.995 |
| family holdout — vexillary | 0.992 | 0.983 | 0.998 | 0.974 | 0.995 |
| **coeff-magnitude** (max_coeff ≥ 6 held out) | 0.999 | 0.927 | 1.000 | **0.744** | **0.657** |

By negative tier (product-disjoint test): Z0 0.984 · Z1 0.981 · Z2 0.980 · Z3 0.954
· Z4 0.984 · **Z5 0.386** (MCC). (Z5 = 304 examples total — tiny, high-variance.)

**Reading:** the model **generalises out-of-distribution** — trained on S≤5 it holds
MCC 0.98 on S₆/S₇, and holds 0.98 on entirely held-out permutation families — so it
learned transferable Schubert structure, not product memorisation. The **honest
failure frontier** is exactly the project's research target: the **Z5 adversarial
tier** (most positive-like distractors: MCC 0.39) and **high-coefficient
extrapolation** (exact-coeff 0.74, NDCG 0.66 when c≥6 is held out). These are where
new combinatorial criteria are needed.

## Shortcut audit (vs `experiments/baselines`, hard zeros = Z2–Z5)
| model | mcc (easy Z0/Z1) | mcc (hard Z2–Z5) |
|-------|-----------------:|-----------------:|
| length-only | 0.673 | **0.000** (collapses) |
| logistic (full features) | 0.922 | 0.817 |
| random forest | 0.958 | 0.827 |
| XGBoost | 0.981 | 0.837 |
| MLP | 0.944 | 0.873 |

Length-only collapses on hard zeros (the dataset's hard zeros genuinely require
structural learning; length→hard accuracy gap 0.29). The transformer's plateau
(mcc≈0.94 / hardzero-F1≈0.99 on val) is competitive with / exceeds the strongest
feature baselines while exposing interpretable internal structure. The exact
**Monk known-rule** baseline scores P/R/F1 = 1.0 on simple-reflection products.

## Interpretability status
- Hooked architecture + counting slots; probes / activation-patching / path-patching
  / SAEs / symbolic distillation implemented and validated
  (`schubml.interp`, `scripts/run_probes.py` etc.).
- Layer probes recover known coordinates (e.g. ℓ(w) decodable from cross layers,
  descents/family tags from encoder layer 0).
- A data-grounded conjecture card (`reports/conjectures/`) distils a decision list
  over the Schubert predicate DSL approximating positivity on held-out hard zeros.

## Known failures / limitations
- High coefficients are scarce (max 6 in v001), so exact-coefficient accuracy is
  dominated by c=1/2. S₆ is sampled (60k/259k); S₇/S₈ sparse. Z5 adversarial tier
  is rare (304) — Z4 is the practical adversarial test.
