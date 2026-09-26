# Experiment: Dominance-order representation for the g<=1/g<=2 residual

Identifier / date: kron-012 / 2026-09-15
Question or conjecture: does a fundamentally different representation
(pairwise dominance-order comparisons between the three partitions,
rather than raw padded part values) expose more signal in the isolated
g<=2 residual than the raw-value representation found (+6.5 points,
kron-006's final isolated deep-dive)?
Status: proposed

## Question and proposal

The only genuinely new representation idea flagged early in this project
and never attempted, rather than another variant of raw shape features.
Dominance order is a natural, standard way to compare partitions in
representation theory (lambda dominates mu iff every partial sum of
lambda's parts is at least the corresponding partial sum of mu's).

Proposed procedure:
1. For each of the 3 pairs among (lambda,mu,nu), compute the vector of
   partial-sum differences d_k = (sum of first k parts of A) - (sum of
   first k parts of B) for k=1..pad_length. This directly encodes the
   dominance relationship (all d_k>=0: A dominates B; all d_k<=0: B
   dominates A; mixed signs: incomparable) and compares the partitions
   against each other rather than describing them in isolation.
2. Isolate the same population as kron-006's final check: triples
   covered by neither Corollary 15 nor 17.
3. Same train n=3..7 / fresh test n=10,11 split, fresh-data validation
   from the start.
4. Compare directly to kron-006's raw-value isolated result (+6.5
   points over majority baseline on this same population).

## Expected outcomes and decision

Stated before execution: genuinely uncertain, unlike some prior
predictions where a specific theorem motivated a number. No literature
result found in this project specifically connects dominance order to
Kronecker coefficient thresholds (unlike Bessenrodt-Bowman for
fat-hook). Weak guess: somewhere in the 5-15 point range over majority,
i.e. plausibly better than the raw-value result, plausibly similar, and
I would not be shocked by close to zero either. Any outcome will be
reported plainly without adjusting the framing after the fact.

## Resources and execution

- Resource agreement: within previously confirmed scope (CPU-only,
  sklearn/numpy, no GPU, no internet needed for computation).
- Commands: pure Python + sklearn in this container.
- Data: reuses the verified kron-001 exact-computation tool.

## Results

Train (uncovered population, n=3..7): 348 triples. Fresh test (uncovered
population, n=10,11): 35,730 triples. Small(g<=2) rate: train 85.63%,
fresh test 53.23% -- large train/test distribution shift in class
balance for this narrow population, consistent with kron-006's original
finding on this same population.

- Majority baseline: 53.23%.
- GBC with dominance-order features: 54.86% (+1.63 points).
- Comparison: kron-006's raw-value representation on this SAME isolated
  population achieved +6.50 points.

The dominance-order representation performed WORSE than the raw-value
representation on this task, not better. The pre-stated prediction
(5-15 point range, genuinely uncertain) was too optimistic; the actual
result falls below that range.

Feature importance is dominated by d_AC_1, d_AC_0, d_AC_2 (partial-sum
differences between the first and third canonically-sorted partitions
at small k) plus n itself -- the model is using the dominance
information as intended, it simply does not carry more signal than the
raw values did for this specific population.

## Interpretation and next action

A clean, honest negative result, reported without reframing. The
"fundamentally different representation" idea, flagged early in this
project as untried, has now been tried and did not outperform the
existing approach. This does not mean dominance order is never useful
for Kronecker coefficients in general -- it is a standard, meaningful
partition comparison -- it means it did not help on THIS specific
narrow residual population with THIS specific model class and sample
size (348 training examples is small).

No conjecture proposed. This closes kron-012 and, with the authors'
request to move to a conclusion, closes the active experimental phase
of this Kronecker investigation.
