# Actual trial: circuit analysis and an obstruction to local information

13 September 2026. The agent developing this workflow used its research and
mechanistic-interpretability prompts on the Schubert project after the authors
authorized a bounded local investigation. This is an actual execution, not
the earlier manual scenario review. The agent had developed the prompts and
knew the project; this is not an independent or comparative evaluation.

The [project context](schubert-context.md) supplied the mathematical conventions,
current discoveries, the latest counterexample, and the attribution boundary
of the earlier model experiments. The frozen manuscript was preserved.

## The question changed after circuit interventions

The existing model receives an ordinary-support witness
`A_i=c^w_{s_i u,v}>0` and selects a left descent `j` of the second factor
with `B_j=c^w_{u,s_j v}>0`. It has two transformer layers with four attention
heads per layer. Each token records inverse-permutation positions near its
simple index and the supplied index. All real tokens participate in attention;
only left descents of the second factor can be selected as outputs.

Earlier tests had removed individual heads and changed the supplied witness.
The trial first checked all 256 head subsets on 3,000 selected examples, then
evaluated the selected configurations on 20,000 historical examples with
inputs from `S_9`. Outputs were computed stably and can lie in larger symmetric
groups. Circuit selection and evaluation used disjoint unordered input pairs;
both source populations had already been examined in earlier research.

| Intervention on the existing model | Valid first proposal on 20,000 examples |
|---|---:|
| Unmodified model | 87.650% |
| All attention-head contributions removed | 82.455% |
| Uniform attention over real tokens | 84.610% |
| Attention restricted to each token itself | 75.875% |

No proper head subset met the preset selection thresholds of 99% decision
agreement and centered-logit R-squared 0.95. The result concerns these
zero-ablation configurations; it does not exclude other model simplifications.
Individual supplied-witness patches left most of that behavior unexplained.
This prompted a more specific question: what information is absent from the
candidate tokens but available through the other tokens?

## Exact examples gave a mathematical obstruction

The agent searched for pairs with identical features at **every candidate**
but disjoint sets of valid outputs. Such a pair defeats any selector restricted
to those features, regardless of model size or training. The search found a
small example in `S_6`, then a stronger example in `S_7` for which the specified
classical filters also give identical outcomes.

For the stronger example, take
\[
u=3125764,\quad v=1453267,\quad i=2,
\quad w=3471562,\quad w'=3456712.
\]
The candidate left descents of `v` are 2 and 3. Both outputs have inverse
positions `(7,1,2)` at values 2,3,4, and the same length. Yet

| Output | Supplied coefficient `A_2` | `B_2` | `B_3` |
|---|---:|---:|---:|
| `3471562` | 1 | 1 | 0 |
| `3456712` | 1 | 0 | 1 |

Both candidate queries at both outputs pass the right-descent-union condition,
all implemented Grassmannian projected-dimension tests, and the implemented
Littlewood–Richardson row inequalities. Appending fixed points preserves this
obstruction for every `n>=7`. The claim concerns the specified local data and
the Boolean outcomes of these tests, not all information in rank matrices or
projected partitions.

Ordinary multiplication, independent localization, and pure Python polynomial
divided differences verified the coefficient tables. This is an exact result
about the current representation; no priority claim or new general Schubert
multiplication rule is asserted. The mathematical question arose from the
circuit trial; exact matching and Schubert calculus established the obstruction.

## The obstruction sharpened the circuit experiment

Matched pairs hold every candidate token fixed. Any difference in candidate
scores must therefore enter from other tokens through attention. The agent
first studied 58 historical pairs, then searched 229,875 remaining stored rows
with inputs from `S_9`, excluding the original 10,000 evaluation rows. This
found 458 further pairs across 71 unordered product groups. They are new
selected triples from a previously studied distribution, not untouched new
mathematical families or 458 independent statistical trials.

For each pair, fix one valid source index and one valid destination index and
measure their logit margin. Replace destination activations with source
activations and recompute downstream operations. Relative to the original
squared discrepancy between source and destination margins:

| Second-layer intervention, all heads | Reduction in margin discrepancy |
|---|---:|
| Replace attention patterns | 75.71% |
| Replace values | 27.23% |
| Replace patterns and values | 99.57% |
| Replace only keys at noncandidate tokens | 76.34% |
| Replace only candidate queries | −0.05% |

The reductions measure recovery of model margins. Interventions interact, so
the reductions need not add. Identity controls, complete restoration, and
equivalence of query/key substitution to pattern substitution all passed.
The principal pattern effect on these pairs comes from keys at tokens indexed
by simple reflections that are not candidate left descents. Which algebraic
quantity those keys encode remains unresolved.

The model selected correctly on both members of only 25 of the 458 pairs.
Context affects its computation, but it has not learned a reliable selection
rule on these difficult examples. The next useful target is to identify the
rank or inverse-position comparisons represented by those keys and test an
explicit mathematical predicate against the retained zeros.

## What the trial changed in the workflow

Inspection exposed untracked historical checkpoints and datasets. The trial
retained a small checkpoint and selected examples, then replayed the main
circuit analyses and all twelve coefficient-table entries in an isolated copy
without those historical files. All 57 retained score arrays reproduced
exactly; the query/key results and coefficient calculations agreed as well.

A native GPU forward-path discrepancy was detected and corrected before
interpretation. A later common-ambient-size assertion was corrected while
preserving completed coefficient checks. The failed runs remain in the record.
The prompts now explicitly cover artifact availability, superseded statements,
shared resource accounting, forward-equivalence checks, representation collisions,
common ambient conventions, and resuming valid partial checks.

This trial produced a useful change of question, an exact obstruction, and a
more specific model hypothesis. It does not estimate the workflow's advantage
over other research methods or its effectiveness on another mathematical topic.

Full records, code, retained inputs, controls, and reproduction instructions:
[dated Schubert trial](../../schubert-structure-constants/research/2026-09-13-autoresearch-circuits).
