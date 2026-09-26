# P16 results and experimental record

Stanley's Problem 16 asks for a combinatorial interpretation of Kazhdan–Lusztig coefficients. This investigation addresses the related combinatorial-invariance question and the limitations of small predictors for individual recurrence terms. The [proof document](proofs.md) supplies conventions and complete arguments for the selected reductions.

## Contribution statements

| Result | Status | Evidence |
| --- | --- | --- |
| Two best E046 decision stumps each fit 30/33 training terms but both predict `(1,1)` on a withheld pair with labels `(0,1)`. | Exact negative result on a fixed curated dataset | `e046` replay; dataset and both rules are retained in the publication JSON |
| A length-thirteen invariance failure requires a canonical connected S₁₂ interval with exactly three relative cycles and leading coefficient at least 126. | Conditional obstruction theorem; no failure is constructed | [Coefficient fingerprint and T041](proofs.md#reduction-to-twelve-positions), using the cited length-twelve theorem and published spectrum |
| Such a failure forces a corrected leading-coefficient decrease of at least five along a path of at most 26 common descents. | Proved reduction | [T045](proofs.md#corrected-descent-gain); independent bounded recurrence controls |
| Every such corrected decrease is at most four. | **Open conjecture**, on the precise source class in the proof document | No rank-twelve proof or exhaustive verification is claimed |
| A connected genus-zero even-gap interval has polynomial `1+10q+29q²+29q³+8q⁴`. | Exact counterexample to a broader, simpler coefficient bound | Fresh agreement of two separately implemented KL engines |

The support-product and genus-heredity lemmas are also included, with their finite controls. The standard literature inputs are identified separately. Model failure helped select a different research direction; the reductions are algebraic arguments, not an algorithm extracted from a reliable learned model.

## E046: prediction task and exact model configuration

The target is whether the degree-four coefficient of an eligible gap-nine upper KL interval is positive. The dataset consists of 33 terms from five explicitly specified terminal permutation pairs, with 16 positive and 17 zero labels. The withheld assessment consists of two facets associated with the same fixed cover, with exact labels `(0,1)`. It is a deliberately chosen counterexample pair, not a random population sample. The exact permutations and candidate selection are in [placement_dataset.py](../research/src/placement_dataset.py).

| Configuration | Value |
| --- | --- |
| Model | One-feature decision stump; both threshold directions |
| Inputs | 14 integer features: ranks 1–4, interval size, global/local rank-defect volume and zero counts, cover position/value spans, simple-sidedness, and endpoint descent counts |
| Labels | Recomputed by integer KL recurrence; eligibility filters use Bruhat order and descents |
| Threshold candidates | Below the smallest observed value, and midpoints between consecutive distinct observed values, for each feature |
| Objective | Number of training classification errors; retain every tied best rule |
| Split | Five fixed training pairs; the two C016 rows are withheld from fitting |
| Randomness | None; no seeds, optimizer, neural layers, or hyperparameter search |
| Best rules | `r1 > 12.5` and `simple_sided > 0.5`, each predicting positive when true |
| Training outcome | Three errors for each rule |
| Withheld outcome | Both rules predict `(1,1)` instead of `(0,1)` |

The feature definitions are executable in `upper_features`; `TRAINING_CORES` fixes the five source pairs. The machine-readable [configuration](../research/e046_config.json) specifies the selected task. The publication replay exports every generated feature row and refits the thresholds after recomputing all labels. This reproduces the narrow historical failure, not a global impossibility theorem for classifiers or a generalization benchmark. No activation patching was performed in E046.

The [model figures](../graphs/README.md) show accuracy against a training-majority baseline and the exact predictions on the withheld pair, with reproducible PNG/PDF outputs and underlying counts.

Additional post hoc figures compare all 14 single-feature fits and refit after omitting each of the five source groups. All five refits select the same two rules. These analyses reuse the original rows, are recorded separately from the historical E046 experiment, and do not provide fresh generalization evidence.

## Exact controls and scope

| Replay name | Domain and check | What it establishes |
| --- | --- | --- |
| `support` | All 4,017 comparable intervals in S₁–S₅; 469 have multiple nontrivial support components | Cartesian-product vertex sets, KL/R factorization, rank bounds, and pure-descent preservation in this finite domain |
| `heredity` | Transposition words on at most four vertices in the disclosed ranges; intervals and endpoint facets through S₅; nested intervals through S₄ | Exact genus identities and hereditary controls in the disclosed domain |
| `descent` | 1,495 odd non-cover intervals through S₅; 1,571 leading recurrences; eight telescoping paths | Recurrence normalization, symmetry restrictions, and boundary handling; no S₁₂ coefficient estimate |
| `coefficient-eight` | One specified S₁₀ interval, independently evaluated by two KL engines | The polynomial and its connected, canonical, genus-zero geometry |

For `heredity`, word lengths run from zero through seven on two and three vertices, zero through six on four vertices, and only the empty word on one vertex. The source code fixes all bounds. For `coefficient-eight`, the zero-based endpoints are `x=(0,4,3,2,1,8,7,6,5,9)` and `w=(4,6,7,0,8,9,1,2,3,5)`. Their gap is ten and their relative permutation has two cycles. This refutes a bound of four for generic even-facet top coefficients; it does not refute the source-sensitive corrected-gain conjecture.

The [publication replay](../research/artifacts/publication_replay.json) records outputs and per-check elapsed wall times. Its 2026-09-25 runs are fresh replications of selected controls; they are not the historical discovery timings. [Historical descent](../research/artifacts/historical/descent.json) and [heredity](../research/artifacts/historical/heredity.json) outputs are retained verbatim. Universal conclusions depend on the mathematical proofs and cited inputs. The finite examples do not establish the unproved inequality or external novelty.

## Appendix use

Use the E046 configuration and outcomes for the model subsection; the coefficient-fingerprint, S₁₂ reduction, and corrected-descent argument for the mathematical subsection; and the explicit open conjecture and coefficient-eight example for limitations. The number of replay commands is not a count of research sessions or model-training runs. The unexported campaign contains additional experiments and unsuccessful routes; this package makes no claim to reproduce all of them.

## Manuscript coverage and accounting

This map checks the P16 claims in `main.tex` and `appendix.tex` from the supplied `patternmining.zip`, SHA256 `3ce3a1c757ce45bf263217da1048da200505c83a2e61a01f4e5463197aa1ff74`. It maps mathematical content rather than copying the manuscript's prose or figures.

| Manuscript label or claim | Repository evidence |
| --- | --- |
| `fig:kazhdan-lusztig`: the polynomial for `1234,3412` is `1+q` | `coefficient-eight` replay, `introductory_example` |
| `app:kl-learning`, `tab:kl-models` | E046 configuration and outcomes above; all feature rows, labels, and fitted rules in the `e046` replay |
| `thm:kl-known` | [Established references](proofs.md#established-references); the length-twelve theorem is an external input |
| `prop:kl-difference`, `eq:kl-r-difference`, `eq:kl-discrepancy` | [Coefficient difference at length thirteen](proofs.md#coefficient-difference-at-length-thirteen) |
| `lem:kl-factorization` | [Support factorization](proofs.md#support-factorization); `support` replay |
| `eq:kl-transpositions`, `lem:kl-position-bound`, `thm:kl-twelve` | [Reduction to twelve positions](proofs.md#reduction-to-twelve-positions) and [genus heredity](proofs.md#genus-heredity) |
| `eq:kl-descent`, `prop:kl-descent-reduction` | [Leading recurrence](proofs.md#leading-recurrence), [common descents](proofs.md#common-descents-and-cancellation), and [corrected descent gain](proofs.md#corrected-descent-gain) |
| `conj:kl-descent` | [Open conjecture](proofs.md#open-conjecture), explicitly unproved |
| `tab:kl-counterexample` | `coefficient-eight` replay and endpoints above; manuscript permutations are one-based, code permutations zero-based |
| `fig:app-kl-flow` | The E046 outcome and the proof chain above; the flowchart is not separate experimental evidence |

**Attribution correction for a later manuscript edit:** `app:kl-learning` currently says a rank-matrix criterion supplied the labels. In the retained E046 implementation, integer KL recurrence supplies the labels; rank and descent tests select eligible examples. The description above follows the executable source.

The [source map](../research/SOURCE_MAP.json) includes a compact `campaign_index` of 62 distinct experiment IDs, E001–E062, appearing in 63 Markdown reports. E052 has separate design and results reports under one ID. These are experiment records, not a measured number of sessions, executions, or model fits. The manuscript currently calls them 62 sessions; that terminology needs qualification. Indexed but unexported experiments are not certified by this bundle. E046 contains the published classifier; T039, T041, T042, T044, and T045 identify the selected mathematical arguments.
