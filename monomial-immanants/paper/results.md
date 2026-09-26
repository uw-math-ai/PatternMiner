# P18 results and exact experiments

Stanley's Problem 18 asks for Schur positivity of monomial immanants of Jacobi–Trudi matrices. The contribution here is a restricted infinite family and its constructive nonnegative character decomposition. The [proof document](proofs.md) defines the immanant/character normalization and supplies every supporting argument used in this export.

## Main statement

For every nonempty skew shape `D=mu/nu` of size `N`, every partition `theta` of `N` with second part at most two, and every monomial-immanant index `lambda` partitioning the matrix order,

```math
[s_\theta]\mathrm{Imm}_{\phi^\lambda}(H_D)\ge0.
```

The coefficient partition `theta` and immanant index `lambda` are different objects. The theorem allows all `(A,2^b,1^c)` with `A>=2` and arbitrary `b,c`, together with the known single-column case. There is no rank, skew-boundary, or matrix-index restriction.

The stronger construction expresses the Kostka permutation weight as a nonnegative integer combination of Hessenberg indicators after empty rows are removed. Averaging gives the Stanley–Stembridge character decomposition. The known hook theorem supplies background and the remaining column case; the positivity of the Stanley–Stembridge summands is a published input. The [integrated proof](proofs.md#integrated-proof) establishes the all-parameter statement through a content generating series, path factorization, a spare-entry argument, and tableau descents on the boundary.

## Exact verification experiments

The independent T053 checker uses a horizontal-strip recursion for Kostka numbers, generalized-binomial coefficient extraction for the proposed series, descending full-poset subtraction for decomposition coefficients, and a separate local-corner calculation. It uses exact integers throughout.

| Check | Fixed domain | Outcome |
| --- | --- | --- |
| Content generating series | Sizes 2–12; at most seven positive content entries; width-two coefficient shapes with first part at least two | 6,005 comparisons agree, including 2,263 with at least two tail twos and 34 all-ones contents |
| Path polynomial identities | Path sizes 1–7; all four endpoint choices | 28 exact identities agree |
| Skew-shape census | 1–5 nonempty rows; outer parts at most four; size 2–11; all partition pairs in that domain | 1,587 shape instances |
| Prefix fibers | All permutations in the specified shape domain | 17,767 feasible permutation checks |
| Corner geometry | Every legal cube in the census | 16,017 cubes and 66,345 loss-set vertex checks |
| Spare-entry bound | Applicable cubes after neutral directions are removed | 14,424 checks pass |
| Full-poset decomposition | Every allowed coefficient shape and Hessenberg function in the census | 776,891 coefficients are nonnegative; 224,196 involve at least two tail twos |
| Thin boundary | All-ones shape cases in the same domain | 526 agreements with the zero-based tableau-descent formula |

These finite checks validate implementations and conventions. The symbolic proof establishes the theorem beyond their bounds. The [historical output](../research/artifacts/historical/width_two.json) is preserved verbatim; the [publication replay](../research/artifacts/publication_replay.json) recomputes the same counts from the exported checker.

## Explicit six-row rectangle decompositions

For `D=(2,2,2,2,2,2)`, the [frozen witnesses](../research/artifacts/historical/rectangles.json) list every positive Hessenberg coefficient. The publication rectangle checker independently enumerates all 132 Hessenberg functions, recomputes the coefficients by full-poset subtraction, and checks each pointwise identity on all 720 permutations.

| Coefficient partition | Positive coefficients | Sum of coefficients | Permutations checked |
| --- | ---: | ---: | ---: |
| `(8,2,2)` | 38 | 50 | 720 |
| `(7,2,2,1)` | 51 | 75 | 720 |
| `(6,2,2,2)` | 42 | 50 | 720 |
| `(5,2,2,2,1)` | 33 | 40 | 720 |
| `(4,2,2,2,2)` | 15 | 15 | 720 |

The term counts refer to distinct Hessenberg indices, whose class functions need not be distinct. These examples illustrate the family; they do not replace its proof.

## E031: separate small-model diagnostic

E031 predicts whether a target comes from the retained genuine-shape census or from selected abstract obstruction examples. Its two inputs are the raw transposition and five-cycle class coordinates, `T_(2,1^4)` and `T_(5,1)`. This is an origin classification task, not a connectivity test. The 798 positive examples are the positive-five-cycle subset of a frozen 1,498-target census. The negative label means absence from that census; it does not certify disconnectedness of every translated full fiber.

The initial stump minimizes balanced classification error over the two features, every between-value threshold, and both leaf labels, with deterministic tie breaking. Genuine targets use a SHA256-based split; abstract examples use parent index modulo three, keeping all descendants of reserved parents out of training. The challenge set was deliberately mined. The revision adds 12 examples from different parents and fits one depth-two tree using class-balanced weighted Gini. There is no neural network, optimizer, random seed, or parameter sweep. The [model configuration](../research/origin_config.json), labeled [stump records](../research/artifacts/historical/origin_stump.json), and [tree records](../research/artifacts/historical/origin_tree.json) fix the experiment.

| Model | Training correct | Ordinary holdout correct | Mined challenge correct |
| --- | ---: | ---: | ---: |
| Stump: abstract iff transposition coordinate is at most 37 | 550/550 | 279/279 | 0/15 |
| Depth-two tree: abstract iff transposition coordinate is at most 60.5 and five-cycle coordinate exceeds 2.5 | 562/562 | 278/279 | 14/15 |

The ordinary holdout contains 268 genuine and 11 abstract examples; a constant-genuine prediction scores 268/279. The challenge contains 15 abstract examples. Its labels and errors were inspected before the revision, so improved performance on the reused challenge is not fresh generalization evidence. Both models fail the recorded zero-error reliability criterion.

The replay also reproduces 15 stump branch interventions, 15 tree root interventions, and 14 tree child interventions, including identity and unused-input controls. These directly inspect boolean comparisons in the fitted models. They do not establish a neural circuit or a mathematical origin criterion. The original exact counterexample preceded model fitting. No source record establishes that E031 discovered or supplied a premise of the T053 width-two theorem.

The `origin` replay freshly refits both models and reproduces their predictions and interventions on frozen labels. It checks target membership and group separation, but does not regenerate the original shape census or obstruction search. This limitation is recorded in the output and configuration.

The [model figures](../graphs/README.md) show accuracy against a training-majority baseline, both fitted decision boundaries, and the controlled branch interventions. PNG/PDF versions, generation code, and exact plotted counts are included.

Further figures show class-specific recall and balanced accuracy on the same ordinary holdout, plus exact coefficient distributions for the five rectangle witnesses. The latter are mathematical decompositions, not classifier outputs or new size-generalization experiments.

## Configuration and limits

The theorem's controls use exact enumeration and symbolic reasoning. The machine-readable [exact configuration](../research/exact_config.json) and checker source specify all finite domains. The E031 diagnostic is a separate contribution. Per-check wall times in the publication output measure the fresh replay, not historical discovery cost.

The full monomial-immanant conjecture, coefficient shapes with second part at least three beyond restricted results, and general decomposition-fiber connectivity remain open in this work. Internal review and exact agreement do not certify publication novelty or external peer review. For an appendix, use the main statement, proof map, finite-domain table, and one explicit rectangle witness; the remaining witnesses are supplementary examples.

## Manuscript coverage and accounting

This map checks the P18 claims in `main.tex` and `appendix.tex` from the supplied `patternmining.zip`, SHA256 `3ce3a1c757ce45bf263217da1048da200505c83a2e61a01f4e5463197aa1ff74`. It maps mathematical content rather than copying the manuscript's prose or figures.

| Manuscript label or claim | Repository evidence |
| --- | --- |
| `fig:immanants`: the example is twice the Schur function indexed by `(3)` | `rectangles` replay, `introductory_example` |
| `app:pm-immanants`, `eq:immanant-pairing` | [Statement and normalization](proofs.md#statement-and-normalization) |
| `thm:immanant-known` | [Established references](proofs.md#established-references-and-review); attributed external inputs |
| `app:pm-widthtwo`, `thm:immanant-positivity` | [Integrated proof](proofs.md#integrated-proof) and supporting lemmas |
| `eq:pm-content` | [Content series](proofs.md#content-series) |
| `eq:pm-cube` | [Prefix fibers](proofs.md#prefix-fibers) and [corner inversion](proofs.md#corner-inversion) |
| `eq:immanant-paths`, `eq:pm-pathproduct` | [Loss geometry](proofs.md#loss-geometry) and [path factors](proofs.md#path-factors) |
| Nonnegative remaining exponent | [Spare-entry argument](proofs.md#a-spare-large-entry-removes-every-nontrivial-denominator) |
| `eq:immanant-descents` | [All-ones boundary](proofs.md#the-all-ones-boundary-is-a-descent-count); executable code uses zero-based descents |
| `app:immanant-checks`, `tab:immanant-experiments`, `fig:app-immanants-flow` | Exact-domain and rectangle tables above; `width-two` and `rectangles` replay outputs |

**Attribution correction for a later manuscript edit:** the concluding sentence of Section 4.5 attributes the contribution to proofs aided by small-model training. The retained records support a separate E031 diagnostic, while the theorem follows from exact computations and symbolic arguments. This export does not claim a causal model-to-theorem link.

The [source map](../research/SOURCE_MAP.json) includes a compact `campaign_index` of 32 distinct experiment IDs appearing in 34 Markdown reports: E001–E034 with E016 and E032 absent from this report inventory. E031 has three reports under one ID. These are experiment records, not a measured number of sessions, executions, or model fits. The manuscript currently calls them 32 sessions; that terminology needs qualification. Indexed but unexported experiments are not certified by this bundle. The selected published models are under E031, and the theorem and its controls under T053.
