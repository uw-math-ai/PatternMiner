# PatternMiner

Code and data for the paper "PatternMiner: A Fusion of Large Language Models
and Small Interpretable Models for Open-ended Mathematical Discovery".

PatternMiner uses an LLM agent to construct mathematical datasets and train and
interpret small models. The agent tests conjectures suggested by these models
and uses counterexamples and hard negatives to select further training
examples. The paper applies it to five problems from Stanley's survey of
positivity in algebraic combinatorics (Problems 10, 11, 12, 16, and 18).

| Directory | Section | Appendix |
|---|---|---|
| `patternminer/` | 3 PatternMiner | |
| `schubert-structure-constants/` | 4.1 Schubert structure constants (Stanley's Problem 11) | A.1 |
| `kronecker-coefficients/` | 4.2 Kronecker coefficients (Stanley's Problem 10) | A.2 |
| `character-table-row-sums/` | 4.3 Character-table row sums (Stanley's Problem 12) | A.3 |
| `kazhdan-lusztig-coefficients/` | 4.4 Kazhdan–Lusztig coefficients (Stanley's Problem 16) | A.4 |
| `monomial-immanants/` | 4.5 Monomial immanants (Stanley's Problem 18) | A.5 |

## PatternMiner

PatternMiner alternates model training and interpretation with exact tests of
mathematical conjectures (Figure 1). An LLM agent chooses a prediction task and
uses a small model's learned features or computations to propose a conjecture.
Counterexamples prompt revisions to the conjecture, and hard-negative mining
supplies examples for the next round of training and interpretation.

The agent follows the prompts in `patternminer/prompts/`. They are Markdown
files and need no installation; `patternminer/README.md` gives the instruction
that starts a project.

| Section 3 | Prompt |
|---|---|
| Formulating a mathematical prediction task | `prompts/models.md` |
| Mining hard negatives | `prompts/experiment.md` |
| Training and evaluating small models | `prompts/models.md`, `guides/model-design.md` |
| Interpreting the learned computation | `prompts/mechanistic-interpretability.md`, `guides/mechanistic-interpretability.md` |
| Formulating and testing conjectures | `prompts/conjecture.md` |

`prompts/research.md` is the entry point for beginning, resuming, and revising
an investigation. `templates/` holds the research record the agent keeps
(`PROJECT.md`, `STATE.md`, `CONJECTURE.md`, `EXPERIMENT.md`), and `examples/`
holds records from the Schubert investigation of Section 4.1.

## Schubert structure constants (Stanley's Problem 11)

    schubert-structure-constants/
        schubml/     models, exact combinatorics, and interpretation code
        scripts/     dataset generation, training, and evaluation
        configs/     training configurations
        reports/     model cards, data cards, and ablation reports
        research/    dated experiment directories with scripts and results

The dated directories under `research/` import from one another through
relative paths, so the layout must be kept as is. The experiments used
Python 3.11, PyTorch 2.6.0, NumPy and lrcalc 2.1; the ML environment is in
`environment-ml.yml` and `requirements-ml.txt`. Exact Schubert products are
generated with SageMath 10.9 (`environment.yml`,
`scripts/generate_products_sage.py`).

| Paper | Source |
|---|---|
| Table 1: Schubert prediction models | `reports/model_cards/pqt_small_v1.md`, `reports/data_cards/data_card_v001.md`; selection transformer in `research/2026-09-14-full-circuits/` |
| Table 2: Exact tests used to refine the Schubert tasks | `reports/conjectures/ablation_baseline.md`, `research/2026-09-13-witness-accuracy/`, `research/2026-09-14-full-circuits/` |
| Table 3: Feature comparison on the same Schubert examples | `reports/conjectures/ablation_baseline.md`, `schubml/interp/sae.py` |
| Table 4: Samples for the fixed Schubert transformer | `research/2026-09-16-circuit-paths/` |
| Table 5: Confirmation results for Schubert interventions | `research/2026-09-16-circuit-paths/` |
| Table 6: Nonzero terms in the proper projections of the two Schubert seeds | `research/2026-09-13-witness-accuracy/SPLITS.json` |
| Table 7: Interventions on inputs with identical candidate statistics | `research/2026-09-16-circuit-paths/MATCHED_CASES.json` |

The circuit interventions (A.1.3) load the trained selection transformer from
`research/2026-09-14-full-circuits/artifacts/SECOND_REFINEMENT_READOUTS.pt`.
Each experiment directory lists its files and their SHA-256 digests in
`EVIDENCE_MANIFEST.json`.

## Kronecker coefficients (Stanley's Problem 10)

    python kronecker-coefficients/kronecker_analysis.py

requires NumPy, SciPy and scikit-learn. The implementation evaluates
characters by the Murnaghan–Nakayama rule and checks them against the
hook-length formula and orthogonality. It then trains the boosted-tree and
logistic-regression classifiers on 3 ≤ n ≤ 7 and evaluates them at n = 10, 11.

| Paper | Source |
|---|---|
| Table 8: Kronecker classification accuracy | `kronecker_analysis.py`; `experiments/kron-002-vanishing-baseline.md`, `kron-003-fresh-validation.md`, `kron-006-small-value-thresholds.md`, `kron-008-value-buckets.md` |
| Table 9: Kronecker tests of necessary conditions and coefficient bounds | `kronecker_analysis.py` (Rosas's bounds); `experiments/kron-002-vanishing-baseline.md`, `kron-008-value-buckets.md` |
| Table 10: Semigroup-inequality agreement for magnitude predictions | `experiments/kron-009-semigroup-property.md` |
| Table 11: Kronecker prediction outside the two special-shape bounds | `experiments/kron-006-small-value-thresholds.md`, `kron-012-dominance-order.md` |
| Table 12: Reversals in the comparison by distinct parts | `conjecture/kron-conjecture-001-complexity-gap.md` |
| Table 13: Association with Kronecker coefficients at n = 10, 11 | `conjecture/kron-conjecture-001-complexity-gap.md` |

## Character-table row sums (Stanley's Problem 12)

    python3 character-table-row-sums/tools/verify_e072.py

checks the saved record of the tableau cancellation experiment (A.3.1) and
the published files against their SHA-256 digests. It uses only the Python
standard library.

| Paper | Source |
|---|---|
| Table 14: Independent image selection in the tableau cancellation experiment | `research/e072-beta221/results.md`, `result.json.gz` |
| Table 15: Exact checks and feature ablations for the cancellation experiment | `research/e072-beta221/results.md`, `result.json.gz` |

The experiment script is `research/e072-beta221/historical_runner.py`, with
helpers in `research/src/`. The run protocol is
`research/e072-beta221/protocol.md`.

## Kazhdan–Lusztig coefficients (Stanley's Problem 16)

    python3.13 kazhdan-lusztig-coefficients/tools/verify_bundle.py
    python3.13 kazhdan-lusztig-coefficients/tools/replay.py --check-recorded

The first command checks file integrity and the saved results; the second
recomputes the exact checks and compares them with
`research/artifacts/publication_replay.json`. `--only e046` selects the
decision-stump experiment of A.4.1 and `--only coefficient-eight` the example
of Table 17. Both use only the standard library of Python 3.13.

| Paper | Source |
|---|---|
| Table 16: Decision stumps for vanishing of individual terms | `research/src/e046_replay.py`, `research/e046_config.json` |
| Table 17: A counterexample to a coefficient bound based on connectedness and (A.17) | `research/src/coefficient_eight.py` |

The code writes permutations zero-based; the paper writes them one-based.
`coefficient_eight.py` computes the polynomial of Table 17 with two
independent Kazhdan–Lusztig implementations, `exact_kl.py` and
`independent_kl.py`.

## Monomial immanants (Stanley's Problem 18)

    python3.13 monomial-immanants/tools/verify_bundle.py
    python3.13 monomial-immanants/tools/replay.py --check-recorded

`--only width-two` selects the exact decomposition experiments of A.5.2.

| Paper | Source |
|---|---|
| Table 18: Exact checks for the immanant decomposition | `research/src/width_two_controls.py`, `research/exact_config.json`, `research/artifacts/publication_replay.json` |
