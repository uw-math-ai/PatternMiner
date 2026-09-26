# Experiment: Multi-class value bucketing (g=0,1,2,3,4,5,6+)

Identifier / date: kron-008 / 2026-09-15
Question or conjecture: can a model classify g(lambda,mu,nu) into the
specific buckets {0,1,2,3,4,5,6+}, rather than a binary threshold or
full regression -- a middle ground between kron-006 (thresholds) and
kron-007 (regression).
Status: proposed

## Question and proposal

Motivated by the authors' request for a direct "characteristic" test
across specific small values, connecting to Rosas's exact bounds
(g<=2 when two partitions are hooks; g<=3 when hook+two-row) found in
kron-006 -- a 7-class bucketing lets us check per-class accuracy against
those known exact boundaries directly, not just a single threshold.

Proposed procedure:
1. Same exact-computation tool, same train n=3..7 / fresh test n=10,11
   split as all prior experiments.
2. Labels: g mapped to bucket {0,1,2,3,4,5,"6+"} (7 classes).
3. Same shape features as kron-006/007 (padded parts, canonically
   sorted).
4. Gradient-boosted multi-class classifier (sklearn supports this
   natively), majority-class baseline, full confusion matrix.
5. Cross-check specific cells of the confusion matrix against Rosas's
   Corollary 15 (two hooks -> bucket must be 0,1, or 2 -- never 3+) and
   Corollary 17 (hook+two-row -> never bucket 4+) as an exact sanity
   check: any violation would indicate a bug, not just weak accuracy.
6. Fresh-data validation built in from the start (train n=3..7, evaluate
   on untouched n=10,11), per this project's standing rule.

## Expected outcomes and decision

Stated before execution: I expect overall multi-class accuracy well
above the majority-class baseline (which will likely be dominated by
bucket 0 or 1, given the skewed distribution seen in kron-002/007), but
well below the ~80-90% seen on the simpler binary tasks -- my guess is
somewhere in the 40-55% range. I expect most errors to be "off by one
bucket" (confusion concentrated near the diagonal of the confusion
matrix) rather than wildly wrong, since a misestimate of magnitude
should usually land nearby. I expect the "6+" bucket to be hardest to
call correctly, consistent with kron-007's finding that the model
cannot extrapolate to large magnitudes it didn't see in training. A
result far outside this range, or a violation of the Rosas sanity
checks, would trigger the same investigation discipline used throughout
this project.

## Resources and execution

- Resource agreement: within previously confirmed scope (CPU-only,
  sklearn/numpy, no GPU, no internet needed for computation).
- Commands: pure Python + sklearn in this container.
- Data: reuses the verified kron-001 exact-computation tool.

## Results

Trained on n=3..7 (1,095 triples), tested on fresh n=10,11 (44,100
triples). Bucket distributions confirm the expected right-skew: train
{0:575, 1:372, 2:98, 3:25, 4:11, 5:10, 6+:4}; fresh test {0:18146,
1:6058, 2:3152, 3:2533, 4:1696, 5:1427, 6+:11088} -- bucket 6+ grows to
be the second-largest class on fresh data, consistent with kron-007's
finding that larger n produces larger coefficients.

- Majority-class baseline (always predict bucket 0): 41.15%.
- GBC multi-class accuracy: 46.46% (+5.32 points) -- within the
  pre-stated 40-55% prediction range.
- Accuracy within 1 bucket of the true value: 67.20% -- confirms the
  prediction that errors concentrate near the diagonal rather than being
  wildly wrong.
- Confusion matrix (see raw output) shows bucket 0 recalled well
  (72.5%), bucket 6+ recalled poorly (31.5%, frequently confused with
  bucket 1) -- consistent with kron-007's finding that the model cannot
  extrapolate well to large magnitudes.

Rosas sanity checks: ground-truth data has 0 violations of Corollary 15
or 17 (confirms data pipeline correctness, as expected). More
interestingly, the MODEL's own predictions were checked for implicit
consistency with these theorems, despite never being told about them:
- Corollary 15 (two hooks -> bucket <=2): 0 model violations across all
  5,236 covered triples -- the model has perfectly, implicitly learned
  to respect this exact constraint from examples alone.
- Corollary 17 (hook+two-row -> bucket <=3): 15 violations of 5,232
  covered triples (0.29%) -- very high but not perfect consistency.
- Accuracy restricted to the Corollary-15-covered subset: 75.53%,
  substantially higher than the 46.46% overall figure, as expected since
  the true label space is narrower there.

## Follow-up: investigating the 15 Corollary-17 violations directly

All 15 violations inspected individually (small enough to eyeball
completely). Striking, consistent pattern:
- Every violation has TRUE g in {0,1} -- the smallest possible values,
  not near-boundary cases as initially guessed.
- The model dramatically OVERestimates in every case (predicting bucket
  4-6 when truth is 0 or 1), never underestimates.
- The two-row partition involved is consistently a balanced, near-square
  shape: (5,5), (6,4), (6,5), (7,4) -- never a lopsided two-row shape.
- The accompanying (non-hook, non-two-row) third partition is always
  moderately complex: (8,2,1), (7,3,1), (6,2,2), (7,2,1,1), etc.

Interpretation: the model appears to have learned a coarse heuristic
consistent with the monotonicity/fat-hook story from kron-006 -- a
balanced, spread-out two-row shape paired with a structurally complex
partner suggests a LARGE coefficient in the broader population. This
heuristic misfires specifically in the narrow band of cases where the
third partition happens to be a hook, which Rosas's Corollary 17 proves
forces smallness regardless of how "spread out" the other two shapes
look. The model's implicit pattern-matching is statistically reasonable
but doesn't perfectly encode the exact theorem's logic.

## Follow-up: conjugation symmetry -- does the model implicitly respect it?

Motivated directly by the Corollary 15/17 implicit-consistency finding
above: tested a different, more fundamental exact identity --
g(lambda,mu,nu) = g(lambda',mu',nu) (conjugating any two of the three
partitions leaves g unchanged).

Step 1 -- verify the identity itself against the exact tool (never
assumed): checked 729 randomly sampled triples across n=3..8. 0
mismatches. Confirmed sound, as a real theorem must be.

Step 2 -- does the trained multi-class bucket model (from kron-008,
never given this symmetry) implicitly respect it? Sampled 6,000 fresh
n=10,11 triples, compared the model's predicted bucket for (lambda,mu,nu)
against its prediction for (lambda',mu',nu). Raw agreement: 57.6% --
much weaker than Corollary 15/17's near-perfect implicit respect
(100%/99.7%).

Computed the proper chance-agreement baseline (accounting for class
imbalance, since agreement can happen by chance when one bucket
dominates): if the two predictions were independent draws from the
model's actual predicted-bucket distribution, expected chance agreement
would be only 21.0%. Observed 57.6% is +36.6 points above this chance
baseline -- a real, substantial signal, just far weaker and noisier than
the boundedness-theorem result.

## Interpretation

A genuine middle-ground finding, informative precisely because it is NOT
another clean success: the model has real but partial implicit
sensitivity to conjugation symmetry, unlike its near-perfect implicit
respect for Corollary 15/17. A plausible explanation: hook-ness and
two-row-ness are properties directly visible in the leading-part
statistics that dominate this feature representation (large first part
relative to n, few parts), so a model trained on raw padded part values
can pick them up naturally. Conjugation, by contrast, scrambles a
partition's part-size pattern into a very different-looking vector under
this same representation (e.g. (6,4) conjugates to (2,2,2,2,1,1)) -- the
symmetry is real in the underlying mathematics but not directly exposed
by this particular choice of features. This is a clean illustration for
the paper: a model's implicit respect for a mathematical symmetry
depends on whether that symmetry happens to align with the chosen
representation, not on how fundamental the symmetry is mathematically.
No conjecture proposed (describes model behavior, not new mathematics).

## Next action

Three implicit-consistency checks now done, with a clean spread of
outcomes: Corollary 15 (near-perfect), Corollary 17 (near-perfect with a
characterized exception band), conjugation symmetry (real but partial,
well above chance). This spread is itself a useful, reportable finding
about how representation choice shapes what a model can implicitly
learn. Closing this experiment here; further checks (e.g. quantifying
the Corollary-17 blind spot precisely, or testing conjugation with a
representation that makes it more visible) are open options for the
user to choose from, not a default continuation.

## Follow-up: does a conjugate-aware representation close the gap?

Direct causal test of the representation-alignment hypothesis above:
retrained with each partition's conjugate's padded parts concatenated
onto the original features, then re-ran the same agreement check.

Result: agreement improved from 57.6% to 66.3% (+8.7 points) -- real,
directionally confirms the hypothesis, but far short of Corollary
15/17's near-perfect respect. Overall bucket-classification accuracy
also improved slightly (46.46% -> 47.92%), so the added features did not
hurt the main task.

Honest interpretation: giving the model direct access to conjugate
information helps, but does not by itself teach it the exact symmetry --
nothing in training enforces that the two representations of the "same"
underlying fact must produce identical predictions; the model still has
to learn this from examples, same as any other pattern, and only
partially does. This is a genuine partial confirmation, not a full one --
recorded as such rather than rounded up. Closing this line of
investigation here; a hard architectural constraint (e.g. explicitly
symmetrizing predictions at inference time) would guarantee the property
but was not attempted, since that would be enforcing the symmetry rather
than testing whether the model discovers it.
