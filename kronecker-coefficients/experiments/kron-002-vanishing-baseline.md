# Experiment: Elementary baseline for Kronecker vanishing prediction

Identifier / date: kron-002 / 2026-09-13
Question or conjecture: none yet -- exploratory pilot to see whether
simple partition-shape invariants carry any signal about vanishing of
g(lambda,mu,nu), before considering any larger model.
Status: proposed

## Question and proposal

Uncertainty addressed: do elementary, easily-computed invariants of three
partitions (padded part sizes, in a canonicalized order removing the
argument-permutation symmetry) carry detectable signal about whether
g(lambda,mu,nu) = 0, beyond the majority-class rate?

Proposed procedure:
1. Enumerate unordered triples (with repetition) of partitions of n, for
   n = 3..9, using combinations_with_replacement (avoids storing the same
   unordered triple multiple times, matching the target's symmetry).
2. Label each via the verified kron-001 tool: vanish = (g == 0).
3. Features: each partition zero-padded to a fixed length (9, the max n
   used here), the three padded vectors placed in a fixed canonical order
   (sorted lexicographically) so the representation does not depend on
   which partition is called lambda/mu/nu, plus n itself as an explicit
   feature.
4. Elementary baselines only, per model-design.md: logistic regression and
   gradient-boosted trees (scikit-learn), no neural network yet.
5. Split by n: train on n = 3..7, test on n = 8..9, to check
   generalization to unseen sizes rather than only interpolating within
   a size (per model-design.md's caution about equivalent-family overlap).

## Expected outcomes and decision

Exploratory/confirmatory mix, prediction stated before execution:
- Majority-class baseline (always predict the more common label) will
  have some accuracy > 50% since vanishing is unlikely to be a 50/50
  split, but this number alone means nothing.
- Prediction: the elementary invariants (raw padded part sizes) do NOT
  encode the actual algebraic vanishing criterion directly, so I expect
  the trained baselines to beat the majority-class rate by only a modest
  margin (a guess: somewhere in the range of a few to ~15 percentage
  points), not to approach near-perfect accuracy. A result far outside
  this range (either much closer to majority-class, or suspiciously
  near-perfect) would itself be informative and worth investigating
  rather than accepting at face value.
- If accuracy is well above baseline: worth inspecting which invariants
  the model weights most, as a lead for a real combinatorial criterion.
- If accuracy is at or barely above baseline: the representation is
  probably too impoverished (raw padded parts, not conjugate shapes,
  containment relations, or other classical partition statistics), which
  is itself a useful negative result pointing to the next representation
  to try -- not evidence that no learnable structure exists.

## Resources and execution

- Resource agreement: authors confirmed -- numpy + sklearn,
  CPU only, no GPU, no internet, n up to 9. No new agreement needed
  beyond what was just given.
- Commands: pure Python + sklearn in this container.
- Data: generated in-memory via the verified kron-001 code; no external
  file.
- Exactness: labels are exact (vanish iff exact integer g == 0); no
  approximation in the ground truth, only in the learned classifier.

## Results

Generated 8,079 unordered (lambda,mu,nu) triples for n=3..9 in <1s using
the verified kron-001 tool. Vanish rate hovered near 45-58% across n, no
degenerate class imbalance. Trained on n=3..7 (1,095 triples), tested on
n=8..9 (6,984 triples), unseen sizes.

- Majority-class baseline: 47.37% test accuracy.
- Logistic regression: 77.51% (+30.1 points over majority).
- Gradient-boosted trees: 89.09% (+41.7 points over majority).

This is FAR outside the pre-stated prediction range (a few to ~15 points).
Per the plan above, this triggered investigation rather than write-up as-is.

Follow-up 1 -- easy-case shortcut check: split test set by whether any of
the three partitions is the trivial (n) or sign (1^n) partition (for which
g is exactly determined by basic character orthogonality, no open
mathematics involved). 1,384 of 6,984 test triples ("easy") vs 5,600
("hard", neither present).
  - EASY subset: vanish rate 94.4%; majority baseline already 94.36%
    there; GBC only reaches 95.81% -- i.e. almost no help beyond the
    trivial rule on these cases, and logistic regression is actually
    *worse* than majority (86.2%) there.
  - HARD subset (80% of test data, no classical shortcut applies):
    majority baseline 35.75%; GBC reaches 87.43% (+51.7 points). This
    is not explained by the easy-case shortcut -- if anything the
    hard-subset result is more surprising than the aggregate number.

Follow-up 2 -- feature importance: GBC's importance is concentrated on
the largest and second-largest parts (position 0 and 1) of each of the
three canonically-ordered, zero-padded partitions (partA_pos0: 0.264,
partC_pos0: 0.162, partC_pos1: 0.136, partB_pos0: 0.094, ... top 4
features already ~66% of total importance). This pattern -- reliance on
leading parts across all three inputs -- is consistent with some form of
a leading-part / dominance-type necessary condition for vanishing, of the
kind already present in the Kronecker literature, rather than obviously
novel structure. It is not yet identified against a specific named
theorem; I have not verified a precise classical inequality this
corresponds to.

Follow-up 3 -- literature comparison (corrected after an initial mistake):
first attempt used a condition transcribed from a search snippet
("mu2+nu2 < lambda2+lambda3+lambda4") without checking its hypotheses; it
turned out to be Bravyi's inequality specialized to length(lambda)<=4,
length(mu),length(nu)<=2, applied here without that restriction. Soundness
check caught this immediately: it produced 2,816 false positives (claimed
vanishing on actually-nonvanishing triples) out of 6,198 firings -- an
unsound "necessary condition" is not a necessary condition, so this was
discarded rather than reported.

Fetched the source paper (Mishna-Rosas-Sundaram, arXiv:1811.10015) directly
to get the actual general statement (their Eqn. 17, a theorem of
Murnaghan, no length restriction): |lambda_bar| <= |mu_bar| + |nu_bar|,
where x_bar denotes x with its largest part removed. Since g is symmetric
in its three arguments, this must hold under every role permutation,
equivalently: letting c_x = n - x_1 for each partition x, g=0 is forced
whenever (c_lambda, c_mu, c_nu) violates the triangle inequality (largest
exceeds the sum of the other two).

Re-verified soundness: fired on 1,479 of 8,079 triples, 0 false positives
-- correct this time, as a real necessary condition must have zero.

On the hard subset: this corrected classical rule alone gives 73.59%
accuracy (fires on only 9.3% of triples, catches 26.1% of actual
vanishing cases) -- a real, substantial improvement over the 35.75%
majority baseline, but well short of GBC's 87.43%. Critically, of GBC's
predicted-vanish cases, only 25.3% overlap with what this classical rule
catches. GBC is not simply a noisy rediscovery of this specific known
inequality; it is picking up on something else, at least in part.

Follow-up 4 -- chasing the residual (the ~75% of GBC's vanish-predictions
not explained by the corrected classical rule): inspected the 1,145
hard-subset triples where GBC is correct, actual label is vanish, and the
classical rule is silent.

First hypothesis (naive generalization, motivated by feature importance
on the second-largest part): a "top-2" analogue of the triangle test,
using c2_x = n - x1 - x2 instead of c_x = n - x1. Soundness-checked over
the FULL 8,079-triple dataset (not just the residual subset that
motivated it, to avoid confirming a pattern only where it was found):
UNSOUND -- 1,344 false positives out of 4,014 firings. Discarded; not a
valid necessary condition as stated.

Second hypothesis: the residual is dominated by hook-shaped partitions
(at most one part >1), which have their own known combinatorial rule
(Blasiak's theorem) -- a different classical special case than the
triangle inequality. Quantified rather than asserted: 76.9% of the 1,145
residual triples contain a hook-shaped partition, versus a 58.4% base
rate across the whole hard subset. Real enrichment, but partial -- this
is evidence hooks contribute to the residual, not proof they explain all
of it. Not yet checked against Blasiak's actual formula (only checked
for hook-shape presence, not fetched/implemented the specific rule).

Follow-up 5 -- the no-hook residual, examined directly (Blasiak's actual
combinatorial rule was NOT implemented; user chose to skip that heavier
lift). Inspection of the 265 no-hook residual triples immediately
surfaced a visible pattern: repeated two-row partitions like (6,2).
Two-row shapes are ALSO a separately solved special case in the
literature (Rosas 2001, explicit Kronecker formula for two-row/hook
shapes). Quantified: 58.1% of the no-hook residual contains a
two-row-or-fewer partition, versus a 39.0% base rate in the whole hard
subset -- real enrichment, same pattern as the hook finding.

After removing both known special cases (hook-containing and
two-row-containing), 111 of the original 1,145 residual triples remain
(9.7% of the residual, about 1.6% of the full 6,984-triple hard test
set). Full funnel: 1,145 -> 265 (remove hook) -> 111 (remove two-row).

## Interpretation and next action

Two more known special cases (hooks, two-row shapes) together explain
roughly 90% of what initially looked like unexplained model behavior.
This is itself a useful, honest finding: most of GBC's apparent
"discovery" was re-deriving cases already solved in the literature, just
not by the one specific inequality checked first.

IMPORTANT METHODOLOGICAL CAVEAT, per experiment.md: the remaining 111
triples were found by successively filtering the SAME fixed test set on
which GBC's predictions were already known. This is adaptive selection,
not independent evidence -- the workflow explicitly warns against using
adaptively selected examples as evidence for a generalization claim. It
would be a mistake to report "111 mystery triples" as a discovery without
first checking whether this effect replicates on genuinely fresh data
(e.g. n=10/11, never touched by this filtering process) rather than on
the same n=8/9 test set that produced it. That check has NOT been run.
No conjecture is proposed from the 111 triples at this stage -- doing so
now would risk exactly the kind of overfitting-to-a-search this project's
own guidance warns about.
