# Experiment: Fresh-data check on n=10,11 -- does the residual replicate?

Identifier / date: kron-003 / 2026-09-13
Question or conjecture: does GBC's advantage over the combined
"triangle + hook + two-row" known-rules baseline survive on n=10,11 data
that was never touched by the earlier adaptive filtering (kron-002)?
Status: proposed

## Question and proposal

kron-002's 111-triple "unexplained residual" was found by successively
filtering the SAME n=8,9 test set GBC's predictions were already known
on -- adaptive selection, not independent evidence per experiment.md.
This experiment tests the effect on genuinely fresh data instead.

Proposed procedure:
1. Retrain GBC and logistic regression on n=3..7 (same training range as
   kron-002), but with padding length extended to 11 (not 9), since
   partitions of n=10,11 can have up to 10-11 parts and the original
   padding would silently truncate them. This changes the trained
   model's exact weights slightly relative to kron-002 but keeps the
   representation, training range, and procedure identical in kind.
2. Generate FRESH triples for n=10 and n=11 only -- never used in
   training, never used in the kron-002 filtering that produced the
   111-triple residual.
3. Build the combined known-rules classifier: predict vanish=1 if EITHER
   the corrected Murnaghan triangle condition fires, OR any partition in
   the triple is a hook, OR any partition is two-row-or-fewer; predict
   vanish=0 otherwise. (Hook/two-row are used here only as a proxy
   correlate of "classically explainable," not as an exact vanishing
   rule -- this is an approximation, stated explicitly, not Blasiak's or
   Rosas's exact formulas.)
4. Compare accuracy of majority-class, combined-known-rules classifier,
   and GBC on the fresh n=10,11 data (restricted to the "hard" subset
   excluding trivial/sign-partition triples, consistent with kron-002).

## Expected outcomes and decision

Genuinely uncertain, stated before execution:
- If GBC's advantage over the combined-rules baseline shrinks to near
  zero on fresh data, the 111-triple residual was likely an artifact of
  adaptive search on one fixed sample -- an important negative result,
  not a failure.
- If GBC still meaningfully beats the combined baseline on fresh n=10,11
  data, that is real evidence the residual is not just overfitting to
  the earlier sample, and would be worth further mechanistic follow-up.
- No specific percentage predicted -- this is explicitly the test of
  whether the earlier finding generalizes at all, not a confirmatory
  experiment with a precise expected effect size.

## Resources and execution

- Resource agreement: within previously confirmed scope (CPU-only,
  sklearn/numpy, no GPU, no internet). n=10,11 increases partition counts
  (p(10)=42, p(11)=56) so triple counts grow substantially versus n=8,9;
  expected to still run in well under a minute given the cached exact
  Kronecker tool.
- Commands: pure Python + sklearn in this container.
- Data: generated in-memory via the verified kron-001 tool; no external
  file.

## Results

First attempt used a flawed "combined known-rules" classifier (triangle
OR hook-present OR two-row-present -> predict vanish). This produced
11,377 false positives on 39,200 fresh triples -- caught immediately
because hook/two-row presence was never actually a sufficient condition
for vanishing, only a marker of "has a known exact formula" (which can be
positive or zero). This conflated two different kinds of fact and was
discarded; corrected to compare against the triangle rule alone, the only
actually-sound baseline, with hook/two-row used only descriptively as in
kron-002.

Corrected results, fresh n=10,11 hard subset (39,200 triples, vanish rate
34.17%, majority-baseline accuracy = vanish rate since majority class
flips to nonvanish here):
- Triangle-only baseline: 74.33% accuracy, 0 false positives (sound,
  verified).
- GBC (retrained on n=3..7 with padding extended to 11, same procedure as
  kron-002 otherwise): 81.46% accuracy.
- GBC's advantage over the sound triangle baseline: +7.14 points (versus
  +13.84 points on the original n=8,9 data).
- Residual (triangle silent, GBC correctly predicts vanish): 7,231 of
  39,200 (18.45%), comparable in proportion to kron-002's 20.4%.
- Of this residual: 61.1% contain a hook (vs 43.4% base rate) and 37.0%
  contain a two-row shape (vs 27.1% base rate) -- same direction and
  similar magnitude of enrichment as kron-002's n=8,9 finding.
- After removing both hook- and two-row-containing triples: 1,454 of
  39,200 (3.71% of the whole fresh hard subset) remain unexplained --
  a LARGER proportion than kron-002's 1.59% (111 of 6,984), not smaller.

Follow-up 6 -- characterizing the fresh 1,454-triple residual. Three-row-
or-fewer enrichment: OPPOSITE direction from hook/two-row -- only 39.8%
of the residual is three-row-or-fewer, versus a 63.5% base rate. The
residual is actually biased toward MORE parts (mean 5.81 vs 4.77 base,
median 6.0 vs 4.0), largely a mechanical consequence of already removing
the few-part (hook, two-row) families rather than a new discovery.

Follow-up 7 -- principled generalization of Murnaghan's rule: tested
whether the triangle inequality on c_k(x) = n - (sum of x's first k
parts) remains a sound necessary condition for k=2..6, motivated by the
one rule known to be true (k=1) rather than ad hoc pattern-matching.
Soundness-checked across ALL available data (train n=3..7 + fresh
n=10,11, 45,195 triples total): k=1 sound (0 false positives, as
expected -- this is just Murnaghan's theorem again). k=2 through k=6 all
UNSOUND, with false positive rates around 35-55% of firings. This natural
generalization does not hold.

## Interpretation and next action

Two more negative results, both informative: the residual is not a
"smaller special family" in the hook/two-row sense (opposite enrichment
direction), and it is not explained by a deeper-truncation version of the
one rule that does work. At this point, multiple reasonably-motivated
hypotheses (top-2 triangle, three-row family, deeper-truncation triangle)
have been tried and soundly rejected, while the underlying effect itself
(GBC's advantage over the sound classical baseline) remains real and
replicated on fresh data.

This is a genuine, unresolved research question at this point, not a
quick pattern-matching exercise -- consistent with these problems being
open for a reason. Further progress likely needs either: (a) actual
mechanistic inspection of what GBC/the tree ensemble computes on
specifically these triples (not just aggregate feature importance), or
(b) a different representation entirely (e.g. conjugate partitions,
pairwise dominance order comparisons between all three, rather than only
single-partition truncation statistics), or (c) accepting this as a
validated-but-uncharacterized lead and reporting it as such rather than
continuing an open-ended search. Recommending a check-in with the authors
given several specific hypotheses have now been exhausted.
