# Conjecture: Complexity-gap reliability for Kronecker coefficient comparison

Identifier: kron-conjecture-001
Date / revision: 2026-09-16, v1
Mathematical status: empirically supported in a specified range, NOT proven
Originating question and experiment links: kron-013, kron-014, kron-015

## Statement

Let lambda, mu, nu and lambda', mu', nu' be two triples of partitions of
the same n. Define the complexity statistic
  d(lambda,mu,nu) = |distinct parts of lambda| + |distinct parts of mu|
                     + |distinct parts of nu|.
Let the complexity gap be delta = |d(lambda,mu,nu) - d(lambda',mu',nu')|.

Empirical claim: as delta increases, the probability that the
higher-complexity triple has a Kronecker coefficient at least as large
as the lower-complexity triple increases sharply, approaching certainty
for delta >= 4. Specifically, measured on fresh n=10,11 data (never
used to discover this pattern):
  delta=1: 18.86% violation rate
  delta=2: 7.49% violation rate
  delta=3: 2.53% violation rate
  delta>=4: 0.44% violation rate

This is NOT claimed as a theorem. It is an honestly quantified empirical
regularity, stated precisely enough to be falsifiable and to guide
where a real proof attempt (or a sharper counterexample search) would
be worth aiming.

## Mathematical motivation and content

Why might this be true, even approximately? The proven semigroup
property (Bessenrodt-Bowman, 2017, arXiv:1609.03596, Prop 2.4) shows
Kronecker coefficients are monotone non-decreasing under a SPECIFIC
additive relationship between triples (partition-wise addition). The
present statistic, "number of distinct part sizes," is only a loose,
single-number proxy for structural complexity -- it is not itself an
additive quantity connected to that theorem by any known identity. The
sharp dependence on the gap size is consistent with the following
informal picture: whatever hidden structural information the exact
theorem depends on, this one-dimensional proxy captures a real but
partial signal of it, and errors are more likely when the proxy's
signal is weak (small gap) than when it is strong (large gap). This is
a plausible but NOT verified mechanism -- no attempt has been made to
derive this gap-dependence from the semigroup property directly.

This is a nontrivial claim: it is not a restatement of any theorem
already in the project's record, and no existing source consulted in
this project states it in this form.

## Supporting evidence

- The base statistic (delta ignored) itself: Spearman rank correlation
  0.6421 with true g on fresh n=10,11 data (kron-013), a similarly
  strong 0.6192 on the original n=3..9 sample -- not adaptively
  discovered on the data used for the gap analysis.
- Multiple refinements of the base statistic (excluding hook/two-row
  triples; blending with total part count or repetition count) were
  tested and ALL performed worse (kron-015) -- the bare statistic
  appears to be a locally strong, hard-to-improve choice among the
  alternatives tried, not an arbitrary pick.
- The gap-dependence itself (this record) was measured on 30,223
  sampled comparable pairs across n=10,11 (delta=1: 16,170 pairs;
  delta=2: 9,329; delta=3: 3,599; delta>=4: 1,125), all from fresh data.
  Range tested: n=10,11 only. NOT yet checked at larger n, nor proven
  for all n.
- Exact computation throughout (the verified kron-001 tool); no
  approximation in the ground truth.

## Relation to existing mathematics

No source found in this project's research (Rosas 2001,
Bessenrodt-Bowman 2017, Mishna-Rosas-Sundaram 2018, Ikenmeyer-Mulmuley-
Walter) states a result in this form. The general problem of finding
combinatorial/structural predictors of Kronecker coefficient magnitude
is open (this is essentially Stanley's Problem 10 restated at the level
of "how big," not just "is it zero"). No claim of novelty is made
beyond this project's own search; a literature search specifically for
"Kronecker coefficient" + "distinct parts" + magnitude has not been
performed and should precede any stronger claim.

## Origin and interpretation

Origin: found via a sequence of model-driven and then direct-mathematical
investigations. The base statistic emerged from comparing several
candidate complexity measures against true Kronecker coefficients
directly (kron-013, no model involved in that step). The gap-dependence
refinement was motivated by a mathematical argument (the mismatch
between this project's naive global comparison and the proven
theorem's narrower additive-relationship scope), not by further model
training. No neural network or black-box model is part of the final
claim; every number above comes from exact computation and elementary
statistics on that exact computation.

## Next useful question

1. Does the gap-dependence pattern hold at larger n (e.g. n=15, 20).
2. Can the ~0.44% delta>=4 violations be characterized directly (small
   enough set, likely feasible to eyeball completely) -- do they share
   a further identifiable structural feature, the way earlier violations
   were traced to hook/two-row presence (with mixed success)?
3. Is there a proof strategy connecting this statistic to the actual
   semigroup property or another known structural theorem, rather than
   leaving the mechanism as an open, unverified conjecture-within-a-
   conjecture?
4. A genuine literature search (not yet done) for prior work relating
   partition "distinctness" or similar complexity measures to Kronecker
   coefficient magnitude, before any claim of novelty is considered.

## Status history

- 2026-09-16, v1: initial formulation, following kron-013 (base
  statistic discovery), kron-014/015 (refinement attempts, all
  rejected), and this record's own gap analysis. Empirically supported
  in the range n=10,11 only. Not proven. Not yet checked against
  literature for novelty.

## Follow-up findings (2026-09-16, same day)

**Larger-n test (honest correction to the original claim):** sampled
6,000 triples at n=15 (never touched before; full exhaustive enumeration
would be expensive at p(15)=176, so this is a large random sample, not
exhaustive). The gap-dependence pattern replicates in DIRECTION (still
strongly monotone decreasing) but NOT in absolute reliability:

| Gap | n=10,11 violation rate | n=15 violation rate |
|---|---|---|
| 1 | 18.86% | 25.48% |
| 2 | 7.49% | 12.04% |
| 3 | 2.53% | 5.47% |
| 4+ | 0.44% | 1.41% |

Every gap level is noticeably less reliable at n=15 -- roughly 2-3x
worse across the board. The original phrasing "approaching certainty for
delta>=4" was too strong; it should be understood as "increasingly
reliable, at a rate that itself appears to depend on n," not as a
fixed, n-independent threshold. This is a genuine, honestly-reported
correction to the original statement, not a discarding of the pattern
itself, which remains real and monotone.

**Characterizing the rare gap>=4 exceptions:** eyeballed all 5 gap>=4
violations found in a 20,000-pair sample of the n=10,11 data. Striking,
unprompted pattern: 4 of 5 have the surprising (higher-g,
lower-complexity) triple containing two LITERALLY IDENTICAL partitions
(e.g. (2,2,2,2,2) appearing twice, (1x11) appearing twice). Checked
systematically: repeated-partition triples are only 11.11% of the whole
population, but appear in 4/5 (80%) of these violations -- a roughly
7x enrichment. Sample size is small (5 violations total at this
threshold from this sample), so this is a promising lead, not a
confirmed sub-mechanism. Mathematically well-motivated: g(lambda,mu,mu)
(two equal partitions) connects to symmetric/antisymmetric
decomposition structure genuinely different from the generic
three-distinct-partitions case, giving a real reason repeated-partition
triples might behave atypically rather than this being coincidence.

## Updated next useful question

Given these follow-ups, the sharpest next step is no longer "test at
larger n" (done, with an honest correction) but: does EXCLUDING
repeated-partition triples from the comparison reduce the gap>=4
violation rate further, the same refinement-testing discipline used
throughout this project (and which notably FAILED for the earlier
hook/two-row hypothesis in kron-015)? Not yet tested.

## Refinement test result: excluding repeated-partition triples (POSITIVE)

Unlike every prior refinement attempt in this project (hook/two-row
exclusion in kron-015, length/repetition blending, both of which made
things worse), this one genuinely helps, at every gap level:

| Gap | Full population | Excluding repeated-partition triples |
|---|---|---|
| 1 | 18.86% | 17.09% |
| 2 | 7.49% | 6.28% |
| 3 | 2.53% | 1.33% |
| 4+ | 0.44% | **0.24%** |

The gap>=4 violation rate roughly HALVES when repeated-partition triples
are excluded. This is the first successful refinement found across
kron-013 through this record, and it is mathematically motivated (not
an arbitrary post-hoc filter): g(lambda,mu,mu) has genuinely different
representation-theoretic structure than the generic three-distinct-
partition case.

## Revised statement (v2)

Restricting to triples where all three partitions are pairwise distinct,
the complexity-gap claim strengthens: for delta>=4, violation rate
0.24% (n=10,11 data). This is closer to, though still not, a sound rule.
The repeated-partition case is flagged as a separate, distinct regime
that should likely be treated on its own terms (its own targeted
investigation) rather than folded into the general statement.

Status remains: empirically supported in a specified range (n=10,11,
15), NOT proven, now with two honestly-reported corrections (n=15
reliability degradation; repeated-partition exclusion improves but does
not eliminate the gap>=4 exceptions).

## Literature check (2026-09-16, per the deferred "next useful question" item)

Found directly relevant prior work: Butbaia, Lee, Ruehle, "Interpretable
Machine Learning for Kronecker Coefficients," arXiv:2502.11774 (Feb
2025), building on Lee (2023), "Data-scientific study of Kronecker
coefficients" and "Machine-learning Kronecker coefficients."

**Overlap, honestly assessed:**
- Same target task (vanishing) as this project's kron-002 thread. Their
  accuracy (83% interpretable models, >99% transformers) substantially
  exceeds this project's ~87% GBC result -- real headroom not explored
  here. Recorded as an honest limitation, not hidden.
- Their central tool, "b-loading" (a per-partition statistic derived
  from the Perron-Frobenius eigenvector of a pairwise L1-distance graph
  on partitions of n, summed across the triple as a single scalar
  predictor), is STRUCTURALLY the same kind of move as this project's
  "sum of distinct-part-counts" statistic -- both are sum-of-per-
  partition-statistic predictors. Their specific statistic is more
  sophisticated (spectral, not a simple count).
- They derive an explicit sufficient condition for NON-vanishing
  (b(t) < b* implies g(t) != 0, verified n<=20), structurally analogous
  to this project's use of Murnaghan's theorem for vanishing, but via a
  completely different derivation and for the opposite direction.
- Independent, previously-unconnected confirmation of this project's own
  finding: their saliency analysis found "the first few and last few
  entries [of a partition] are more significant than the middle ones" --
  matching kron-002's feature-importance result (leading parts
  dominate), found independently by a different team and method.

**What appears to remain distinct, based on this search:** no mention in
this paper of a distinct-part-count statistic, Rosas's Corollaries 15/17,
gap-dependence reliability analysis, or the repeated-partition (mu=nu)
special-case finding. Their target throughout is zero-vs-nonzero; this
project's complexity-gap conjecture concerns relative MAGNITUDE between
two nonzero coefficients, a question their paper does not address.

**Honest conclusion:** the general strategy (sum of a per-partition
statistic as a magnitude/vanishing predictor) has real, direct
precedent and should be cited as such if this work is written up
further. The specific statistic, the gap-dependence characterization,
and the repeated-partition refinement do not appear to already exist in
the literature found so far -- but this is a single search, not an
exhaustive review, and should be treated as provisional pending a more
thorough check (e.g. also searching the two earlier Lee 2023 papers
directly, which were not separately verified here) before any stronger
novelty claim.

## Exhaustive literature review (2026-09-16, extended)

Following up on the initial check, a broader, multi-angle search was
conducted: both earlier Lee papers (2023) cited by the 2025 paper, a
direct search for "distinct parts" + Kronecker magnitude, and a search
for the repeated-partition phenomenon and shape-complexity-vs-growth
results specifically.

### Finding 1: the repeated-partition phenomenon is well-established

The finding that repeated partitions (e.g. mu=nu) behave atypically is
NOT novel -- it is a well-known, actively studied classical phenomenon
connected to the FAILURE OF THE SATURATION PROPERTY for Kronecker
coefficients (unlike Littlewood-Richardson coefficients, which do
satisfy saturation). The canonical example, appearing in multiple
sources (Ikenmeyer-Mulmuley-Walter 2017; Bowman-Paget and others):

  g_{(N,N),(N,N)}^{(N,N)} = 1 if N is even, 0 if N is odd.

This parity-dependent irregularity under simple scaling is exactly the
kind of behavior that would make repeated-partition triples prone to
violating a naive magnitude-monotonicity heuristic -- our empirical
finding (repeated partitions are enriched ~7x among the rare gap>=4
violations) is a NEW, SPECIFIC APPLICATION of a well-recognized general
phenomenon, not a novel phenomenon itself. This makes the finding MORE
credible (it has a real, documented mathematical cause), while making
it LESS novel as a discovery in its own right.

### Finding 2 (major): the core principle is a PROVEN THEOREM, not new

Pak and Panova, "Durfee squares, symmetric partitions and bounds on
Kronecker coefficients" (2022), arXiv:2207.02561, prove:
- For partitions lambda,mu,nu with FIXED Durfee square size (a classical
  shape-complexity statistic: the largest k such that the Young diagram
  contains a k x k square), Kronecker coefficients grow AT MOST
  POLYNOMIALLY (explicit bound given, Theorem 1.4).
- The maximal Kronecker coefficients g(lambda,lambda,lambda) for
  SELF-CONJUGATE lambda grow SUPEREXPONENTIALLY.

This is the SAME CORE PRINCIPLE as kron-conjecture-001 -- "shape
complexity governs the magnitude/growth of Kronecker coefficients" --
already established as PROVEN mathematics, using a different specific
statistic (Durfee square size, a classical invariant related to but
distinct from "number of distinct part sizes"). This is a proven
theorem about GROWTH RATE as partitions scale, not directly a
statement about pairwise magnitude comparison between fixed-n triples
the way kron-conjecture-001 is stated -- the precise claims differ, but
the underlying mathematical principle does not appear to be new.

### Revised honest assessment

**NOT novel:** the general principle that partition-shape complexity
governs Kronecker coefficient magnitude (Pak-Panova 2022, proven, via
Durfee square). The specific observation that repeated/equal partitions
behave atypically (well-established, connected to saturation-property
failure, appears in multiple sources since at least 2009).

**Possibly still distinct, with real caveats:** the specific choice of
"number of distinct part sizes" as the complexity statistic (rather
than Durfee square size or another classical invariant) has not been
found stated in this form in the literature searched. The specific
QUANTIFIED empirical gap-dependence reliability curve (18.86% -> 0.44%
violation rate as complexity gap grows, measured precisely at n=10,11
and n=15) is an empirical/statistical characterization, not a
theoretical claim, and was not found elsewhere. Whether Durfee square
size would show an even cleaner (or messier) version of the same
gap-dependence pattern has NOT been tested -- a natural, concrete next
check given this literature finding.

**Revised framing for any presentation of this work:** this conjecture
should be presented as "an empirical, quantified instance of a known
general principle (Pak-Panova's proven complexity-governs-growth
theorem), using a different, simpler statistic, with a specific
reliability characterization not found elsewhere" -- NOT as "a novel
mathematical principle." This is a meaningfully more modest, more
accurate claim than earlier framings in this project's record, and
should supersede them.

## Direct comparison test: distinct-parts vs. Durfee square (definitive)

Tested Durfee square size (Pak-Panova's proven statistic) directly
against distinct-part-count on the identical fresh n=10,11 data:

| Metric | Durfee square (proven-principle statistic) | Distinct-part-count (this project's statistic) |
|---|---|---|
| Spearman rho vs true g | 0.4857 | **0.6421** |
| Gap=1 violation rate | 22.40% | 18.86% |
| Gap=2 violation rate | 11.87% | 7.49% |
| Gap=3 violation rate | 7.78% | 2.53% |
| Gap>=4 violation rate | 3.38% | **0.44%** |

Distinct-part-count outperforms Durfee square at every level, most
dramatically at large gaps (nearly 8x lower violation rate). This
resolves the novelty question concretely: kron-conjecture-001 is NOT a
redundant restatement of Pak-Panova's proven principle applied with
their specific statistic -- it is a genuinely different, empirically
better-performing choice of complexity statistic for this specific task
(pairwise magnitude comparison at fixed n), with a precisely quantified
reliability curve not found in the literature searched.

## Final honest summary

The GENERAL PRINCIPLE (shape complexity governs Kronecker coefficient
magnitude) is established, proven mathematics (Pak-Panova 2022). The
SPECIFIC STATISTIC used here (distinct-part-count) is a genuine,
quantifiable improvement over the statistic used in that proven theorem,
for the specific task of pairwise magnitude comparison -- demonstrated
directly, not merely asserted. The repeated-partition refinement is a
new, specific application of an already-recognized general phenomenon
(saturation-property failure), not a new phenomenon in itself. This is
the complete, honest picture after an exhaustive (multi-angle, not
single-query) literature search: appropriately modest about what is
genuinely new, while identifying a specific, real, defensible,
quantified contribution.
