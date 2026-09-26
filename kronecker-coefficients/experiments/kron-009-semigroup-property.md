# Experiment: Implicit respect for the semigroup/monotonicity property

Identifier / date: kron-009 / 2026-09-15
Question or conjecture: does the trained model's predicted values
respect Bessenrodt-Bowman's semigroup property --
g(lambda,mu,nu)>0 and g(alpha,beta,gamma)>0 implies
g(lambda+alpha,mu+beta,nu+gamma) >= max(g(lambda,mu,nu),g(alpha,beta,gamma))
-- directly, rather than just correlating with its downstream
consequences (fat-hook enrichment, rank correlation) as in prior
experiments.
Status: proposed

## Question and proposal

This is more mechanistically direct than kron-006's fat-hook correlation
or kron-007's rank-correlation finding: those observed correlations
consistent with this theorem's downstream effects; this tests the
theorem's actual mechanism (partition addition monotonicity) directly.

Proposed procedure:
1. Verify the identity itself against the exact tool first, as always --
   sample pairs of triples, compute the componentwise partition sum
   (padding to equal length; sum of two weakly-decreasing sequences is
   weakly decreasing, so this stays a valid partition), confirm the
   inequality holds with zero violations.
2. Using the kron-007 regression model (predicts log(g+1)), test whether
   its PREDICTIONS respect the same inequality on fresh combined triples
   it was not trained on the sums of.
3. Compute a chance baseline the same way as kron-008's conjugation
   check, since naive agreement can be inflated by the model's general
   tendency to predict larger values for larger n.

## Expected outcomes and decision

Stated before execution: I expect real signal above chance (similar to
the conjugation finding, not zero), but I am uncertain whether it will
be closer to Corollary 15/17's near-perfect respect or the conjugation
check's partial respect. My weak guess: closer to partial, since this
property involves combining information from TWO separate triples
(more complex than a single-triple boundedness check), which seems
harder for a model with no explicit combination mechanism to pick up on
than the more local Corollary 15/17 properties.

## Resources and execution

- Resource agreement: within previously confirmed scope (CPU-only,
  sklearn/numpy, no GPU, no internet needed for computation).
- Commands: pure Python + sklearn in this container.
- Data: reuses the verified kron-001 exact-computation tool.

## Results

Step 1 -- verified the property itself against the exact tool: 35 valid
pairs (both source coefficients >0, sampled from n=3..6), 0 violations.
Confirmed sound.

Step 2 -- tested the kron-007 regression model's predictions on 1,614
fresh combinations (sources n=6..9, combined n up to 16, none of these
specific combined triples seen in training):
- Raw agreement (pred(combined) >= max of the two source predictions):
  78.1%.
- Chance baseline (shuffling which "max" pairs with which "combined"
  prediction): 74.2% -- much higher than the conjugation check's 21.0%
  chance baseline, because the model has learned a strong general trend
  that larger n predicts larger magnitude, and combined n is always
  larger than either source n by construction. This inflates naive
  agreement independent of whether the model understands the actual
  mechanism.
- Real signal above chance: only +3.9 points -- much weaker than the
  conjugation check's +36.6 points, and far below Corollary 15/17's
  near-total implicit respect.

This is notably weaker than the pre-stated prediction (expected
somewhere between Corollary 15/17's near-perfect respect and the
conjugation check's partial respect; actual result is close to chance).

## Interpretation and next action

An important, informative negative-leaning result, and a useful
methodological lesson in its own right: the fat-hook correlation
(kron-006) and the strong rank-correlation finding (kron-007) are
consistent with the ground-truth semigroup property holding in the data,
but this result shows the model has NOT specifically learned that
mechanism -- it has instead learned a much cruder proxy (larger n implies
larger magnitude) that happens to often satisfy this particular
inequality by coincidence of construction, not because it understands
partition addition or combines information from two separate inputs in
any meaningful way.

This sharpens the earlier correlational findings: the model's implicit
knowledge tracks surface statistics correlated with the theorem's
consequences (shape complexity, leading parts), not the theorem's actual
combinatorial mechanism. This is a genuinely useful methodological point
for the paper -- naive agreement checks without a proper chance baseline
would have overstated this finding by roughly 20x (3.9 points of real
signal dressed up as 78.1% raw agreement). No conjecture proposed. This
closes kron-009 as a complete, honest, cautionary result.
