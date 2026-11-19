# PatternMiner

Code and data for the paper “PatternMiner: Automated Mathematical Discovery Through LLM-Guided Training and Interpretation of Small Models”.

PatternMiner uses an LLM agent to construct mathematical datasets and train and
interpret small models. The agent tests conjectures suggested by these models
and uses counterexamples and hard negatives to select further training
examples. The paper applies it to five problems from Stanley's survey of
positivity in algebraic combinatorics (Problems 10, 11, 12, 16, and 18).

| Directory | Section | Appendix |
|---|---|---|
| `patternminer/` | 3 PatternMiner | |
| `schubert-structure-constants/` | 4.1 Schubert structure constants (Stanley's Problem 11) | B.1 |
| `kronecker-coefficients/` | 4.2 Kronecker coefficients (Stanley's Problem 10) | B.2 |
| `character-table-row-sums/` | 4.3 Character-table row sums (Stanley's Problem 12) | B.3 |
| `kazhdan-lusztig-coefficients/` | 4.4 Kazhdan–Lusztig coefficients (Stanley's Problem 16) | B.4 |
| `monomial-immanants/` | 4.5 Monomial immanants (Stanley's Problem 18) | B.5 |

## Ablations and direct LLM baselines

The [Schubert ablations](schubert-structure-constants/ablations/README.md) include the matched sampling comparison in B.1.5, the model-interpretation ablation in B.1.4 (Table 8), and the matched Schubert comparison in Appendix C. The [direct LLM baselines](direct-llm-baselines/README.md) give the protocol, the outcomes on all five problems (Table 22), and the two supplemental retries.

The [61 Schubert sessions in Table 1](schubert-structure-constants/research-sessions.csv) were reconstructed by question and recorded outcome. Related restarts and checks were grouped within each session. The other four counts came from experiment logs.

The [data](small-model-analyses/model_diagnostics_data.json) and [plotting script](small-model-analyses/plot_model_diagnostics.py) reproduce the Kazhdan–Lusztig and immanant model assessments in Appendix B.8 (Figures 12–14). Run the script with Python and Matplotlib. The five main investigations by PatternMiner used Claude Fable 5.

## PatternMiner

PatternMiner alternates model training and interpretation with exact tests of
mathematical conjectures (Figure 1). An LLM agent chooses a prediction task and
uses a small model's learned features or computations to propose a conjecture.
Counterexamples prompt revisions to the conjecture, and hard-negative mining
supplies examples for the next round of training and interpretation.

The workflow contains five prompts, four record templates, and three method guides. The agent follows the prompts in `patternminer/prompts/`. They are Markdown
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
| Table 3: Schubert prediction models | `reports/model_cards/pqt_small_v1.md`, `reports/data_cards/data_card_v001.md`; selection transformer in `research/2026-09-14-full-circuits/` |
| Table 4: Exact tests used to refine the Schubert tasks | `reports/conjectures/ablation_baseline.md`, `research/2026-09-13-witness-accuracy/`, `research/2026-09-14-full-circuits/` |
| Table 5: Feature comparison on the same Schubert examples | `reports/conjectures/ablation_baseline.md`, `schubml/interp/sae.py` |
| Table 6: Samples for the fixed Schubert transformer | `research/2026-09-16-circuit-paths/` |
| Table 7: Confirmation results for Schubert interventions | `research/2026-09-16-circuit-paths/` |
| Table 9: Nonzero terms in the proper projections of the two Schubert seeds | `research/2026-09-13-witness-accuracy/SPLITS.json` |
| Table 10: Interventions on inputs with identical candidate statistics | `research/2026-09-16-circuit-paths/MATCHED_CASES.json` |

The circuit interventions (B.1.3) load the trained selection transformer from
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
| Table 11: Kronecker classification accuracy | `kronecker_analysis.py`; `experiments/kron-002-vanishing-baseline.md`, `kron-003-fresh-validation.md`, `kron-006-small-value-thresholds.md`, `kron-008-value-buckets.md` |
| Table 12: Kronecker tests of necessary conditions and coefficient bounds | `kronecker_analysis.py` (Rosas's bounds); `experiments/kron-002-vanishing-baseline.md`, `kron-008-value-buckets.md` |
| Table 13: Semigroup-inequality agreement for magnitude predictions | `experiments/kron-009-semigroup-property.md` |
| Table 14: Kronecker prediction outside the two special-shape bounds | `experiments/kron-006-small-value-thresholds.md`, `kron-012-dominance-order.md` |
| Table 15: Reversals in the comparison by distinct parts | `conjecture/kron-conjecture-001-complexity-gap.md` |
| Table 16: Association with Kronecker coefficients at n = 10, 11 | `conjecture/kron-conjecture-001-complexity-gap.md` |

## Character-table row sums (Stanley's Problem 12)

    python3 character-table-row-sums/tools/verify_e072.py

checks the saved record of the tableau cancellation experiment (B.3.1) and
the published files against their SHA-256 digests. It uses only the Python
standard library.

| Paper | Source |
|---|---|
| Table 17: Independent image selection in the tableau cancellation experiment | `research/e072-beta221/results.md`, `result.json.gz` |
| Table 18: Exact checks and feature ablations for the cancellation experiment | `research/e072-beta221/results.md`, `result.json.gz` |

The experiment script is `research/e072-beta221/historical_runner.py`, with
helpers in `research/src/`. The run protocol is
`research/e072-beta221/protocol.md`.

## Kazhdan–Lusztig coefficients (Stanley's Problem 16)

    python3.13 kazhdan-lusztig-coefficients/tools/verify_bundle.py
    python3.13 kazhdan-lusztig-coefficients/tools/replay.py --check-recorded

The first command checks file integrity and the saved results; the second
recomputes the exact checks and compares them with
`research/artifacts/publication_replay.json`. `--only e046` selects the
decision-stump experiment of B.4.1 and `--only coefficient-eight` the example
of Table 20. Both use only the standard library of Python 3.13.

| Paper | Source |
|---|---|
| Table 19: Decision stumps for vanishing of individual terms | `research/src/e046_replay.py`, `research/e046_config.json` |
| Table 20: A counterexample to a coefficient bound based on connectedness and (B.16) | `research/src/coefficient_eight.py` |

The code writes permutations zero-based; the paper writes them one-based.
`coefficient_eight.py` computes the polynomial of Table 20 with two
independent Kazhdan–Lusztig implementations, `exact_kl.py` and
`independent_kl.py`.

## Monomial immanants (Stanley's Problem 18)

    python3.13 monomial-immanants/tools/verify_bundle.py
    python3.13 monomial-immanants/tools/replay.py --check-recorded

`--only width-two` selects the exact decomposition experiments of B.5.2.

| Paper | Source |
|---|---|
| Table 21: Exact checks for the immanant decomposition | `research/src/width_two_controls.py`, `research/exact_config.json`, `research/artifacts/publication_replay.json` |
