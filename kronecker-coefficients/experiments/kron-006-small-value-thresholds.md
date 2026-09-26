# Experiment: Small-value threshold prediction (g<=1, g<=2)

Identifier / date: kron-006 / 2026-09-14
Question or conjecture: mirroring the mentor's Schubert-project criteria
(c=0 and c<=2), test whether g(lambda,mu,nu) <= 1 and, separately,
g <= 2, can be predicted from partition shape -- a generalization of the
vanishing (g=0) task to slightly larger "small value" thresholds.
Status: proposed

## Question and proposal

Vanishing (g=0) showed strong, validated signal, largely explained by
known theorems. This asks whether "small" (g<=1, g<=2) is similarly
structured, or whether the structure is special to exact zero.

Proposed procedure, with fresh-data validation built in from the start
(applying the lesson from kron-004/005, where skipping this initially
let two false leads stand until a later, separate check):
1. Reuse the verified exact tool and the SAME raw padded-partition
   features that worked for vanishing (not the odd/even-count features,
   which failed for parity) -- direct comparability with kron-002.
2. Labels: (g <= 1) and (g <= 2), as two separate binary tasks.
3. Train on n=3..7. Evaluate on BOTH n=8,9 (as before) AND n=10,11
   (genuinely fresh, in the same run) -- so any surprising result is
   already checked against independent data before being reported, not
   after.
4. Report majority baseline, logistic regression, and GBC for both tasks
   on both evaluation sets side by side.

## Expected outcomes and decision

Stated before execution: given g=0 showed strong signal partly because
zero is a mathematically special value (forced by known inequalities),
I do not expect g<=1 or g<=2 to show equally strong signal -- being
"small but nonzero" is not obviously governed by the same kind of
forcing inequality. A modest effect (kron-004-like, a few points) that
then FAILS fresh validation would not be surprising given the parity
precedent. A strong, validated effect (kron-002-like) would be a genuine
new finding worth pursuing further. Either outcome will be reported
plainly; no result will be called real without surviving the n=10,11
check in this same run.

## Resources and execution

- Resource agreement: within previously confirmed scope (CPU-only,
  sklearn/numpy, no GPU, no internet needed for computation, same n
  range as prior experiments).
- Commands: pure Python + sklearn in this container.
- Data: generated in-memory via the verified kron-001 tool.

## Results

Train n=3..7 (1,095 triples), tested on both n=8,9 (6,984) and fresh
n=10,11 (44,100, generated in the same run before reporting anything).

**g<=1:**
| | n=8,9 | n=10,11 (fresh) |
|---|---|---|
| Majority baseline | 69.07% | 54.88% |
| Logistic regression | 76.92% (+7.85) | 72.11% (+17.23) |
| GBC | 87.41% (+18.34) | 79.27% (+24.39) |

**g<=2:**
| | n=8,9 | n=10,11 (fresh) |
|---|---|---|
| Majority baseline | 79.02% | 62.03% |
| Logistic regression | 80.70% (+1.68) | 69.84% (+7.80) |
| GBC | 87.41% (+8.39) | 72.22% (+10.18) |

Unlike parity, BOTH thresholds show an effect that does not shrink on
fresh data -- if anything grows (g<=1 grows from +18.3 to +24.4 points).
This is the opposite signature from overfitting.

Checked for the trivial/sign-partition shortcut found for vanishing:
excluding those cases does not reduce the effect (g<=1: +27.4 points
excluding easy cases, vs +24.4 including them; g<=2: +11.5 vs +10.2).
Not explained by that shortcut.

GBC train accuracy is very high (99.36% for g<=1, 100% for g<=2) --
flagged as worth caution, but the fresh-data result argues against pure
overfitting since fresh-data accuracy did not degrade.

Literature check: Gutschwager (2010), arXiv:0912.4411, "Reduced Kronecker
products which are multiplicity free or contain only few components,"
directly classifies which (reduced) Kronecker products satisfy g<=1
("multiplicity free") or have few components -- closely related to, but
not verified identical to, the per-triple threshold task tested here.
Also cites a classical monotonicity fact: g(lambda,mu,nu) is bounded and
weakly increasing as the first part of all three partitions grows,
which plausibly explains why "size minus first part"-style features
(already central to Murnaghan's vanishing rule) carry signal for this
threshold task too.

The exact criterion from Gutschwager's paper has NOT been read carefully
or implemented -- per this project's standing rule against citing an
unverified formula (the Bravyi mistake from kron-002), no specific rule
is claimed here, only that directly relevant literature exists and needs
proper reading before any stronger claim.

## Follow-up: fat-hook enrichment check (per user request to resume)

FIRST ATTEMPT (flawed diagnostic, caught and corrected): tested fat-hook
enrichment among "added-value" triples, defined as cases where the
model's prediction differs from majority AND is correct. Result:
fat-hook DEPLETED in this set (73.5%/60.3% vs 89.9% base rate).

This result was misleading due to a structural blind spot: since
majority baseline always predicts "small" (the dominant class), it is
mathematically impossible for a case to count as "added value" while
also being a correctly-identified true-small case -- agreeing with
majority never counts as "different from majority." A confusion-matrix
check confirmed this: 0 cases fell into "correctly identified SMALL,
model differs from majority" for either threshold, while the model in
fact has substantial real recall on true-small cases (84.1% for g<=1,
93.3% for g<=2) that the added-value metric was structurally blind to.

CORRECTED test: fat-hook rate measured directly among TRUE-SMALL vs
TRUE-LARGE cases (not filtered through the majority-diff lens):
- g<=1: 98.8% fat-hook among true-small cases (n=24,204) vs 89.9% base
  rate; 79.0% among true-large cases (n=19,896).
- g<=2: 98.7% fat-hook among true-small cases (n=27,356) vs 89.9% base
  rate.

This IS real, substantial enrichment in the hypothesized direction,
consistent with the Bessenrodt-Bowman semigroup/monotonicity connection
(fat-hook / concentrated shapes correlate with smaller coefficients,
matching that the coefficient grows monotonically as shapes become more
spread out under partition addition).

## Interpretation and next action

The original hypothesis was correct; the first test of it was flawed by
an easy-to-miss structural artifact of comparing against a majority
baseline that already agrees with the class it's being asked to
"add value" on. Correcting the diagnostic recovered a real, substantial,
literature-connected finding: ground-truth smallness of the Kronecker
coefficient correlates strongly with fat-hook partition shape, matching
known monotonicity theory. This is now the closest this project has come
to fully connecting an empirical model-driven finding to a specific
piece of existing theory for a task other than vanishing.

## Final soundness check: is fat-hook-absence a proven rule, or a strong correlation?

Tested directly: does "no fat-hook partition present" reliably force
g>threshold, the same rigor applied to Murnaghan's rule for vanishing?

- g<=1: 4,470 of 44,100 fresh triples have no fat-hook partition. Of
  those, 286 (6.40%) are STILL actually <=1 -- real counterexamples, not
  a sound necessary condition. As a standalone classifier, "predict large
  iff no fat-hook" has 93.60% precision on its firings -- strong, but not
  exact.
- g<=2: 4,470 triples with no fat-hook; 363 (8.12%) violation rate;
  91.88% precision.

Concrete counterexample: n=11, triple=((8,2,1),(4,3,2,2),(4,2,1,1,1,1,1)),
g=1 -- none of the three partitions is fat-hook, yet the coefficient is
still small.

## Major update: exact theorem found and verified (Rosas 2001, full paper read)

Read the full primary source (arXiv:math/0001084) rather than a snippet.
Found something much stronger than the fat-hook correlation: two EXACT,
proven corollaries directly relevant to g<=2 and g<=3:

- Corollary 15: if TWO of the three partitions are both hook shapes,
  g(lambda,mu,nu) in {0,1,2} ALWAYS -- no exceptions, a proven theorem.
- Corollary 17: if two of the three are a hook and a two-row shape,
  g(lambda,mu,nu) in {0,1,2,3} ALWAYS -- also proven.
- (Corollary 7, contrast: two two-row shapes together does NOT bound g
  -- it is provably unbounded in that case, e.g. via the family
  (3l,l)/(3l,l)/(3l,l) growing like ceil((l+1)/2).)

Soundness-checked both directly against our own verified exact tool
across all 8,079 triples for n=3..9:
- Corollary 15 fired on 2,308 triples: 0 violations.
- Corollary 17 fired on 2,262 triples: 0 violations.

Both are exactly sound, as proven theorems must be. This upgrades the
project's understanding from "fat-hook shape correlates with smallness"
(kron-006's earlier finding) to an exact, provable, verified explanation
for a real subset of the g<=2/g<=3 cases: specifically, any triple where
some PAIR of the three partitions are both hooks or a hook-and-two-row
combination is provably, exactly bounded, independent of the third
partition's shape.

## Comparing the exact theorem against the model's actual predictions

Tested how much of GBC's real skill on the g<=2 task is explained by
Corollary 15/17 specifically, rather than just verified as sound in
isolation.

Coverage: Corollary 15 fires on only 11.87% of fresh n=10,11 triples.

First diagnostic attempt was flawed (same class of mistake as the
fat-hook "added value" check earlier in this project): comparing a
"Corollary-15-fires -> small, else majority" classifier against majority
baseline gave a delta of exactly 0.0000 -- because majority baseline
ITSELF already predicts "small" for everything (small is the majority
class here), so a rule that also defaults to "small" when it doesn't
fire is mathematically identical to majority everywhere. Caught and not
reported as a real result.

Correct diagnostic: restrict to the 35,730 fresh triples NOT covered by
either Corollary 15 or 17, and compare GBC to majority baseline on JUST
that subset. Result: majority=53.23%, GBC=65.85% -- GBC still beats
majority by +12.63 points even with all exact-theorem-covered cases
removed. Additionally, of GBC's correct "small" predictions overall,
only 32.6% fall inside Corollary 15/17-covered territory.

## Deep dive into the uncovered territory (properly isolated this time)

Retrained fresh, from scratch, on ONLY the uncovered population (not
covered by Cor 15/17): train n=3..7 (348 triples), tested on fresh
n=10,11 (35,730 triples, isolated the same way).

Result: majority baseline 53.23%, GBC 59.72% (+6.50 points) -- smaller
than the +12.63 points found earlier using a model trained on the FULL
population and merely evaluated on this subset. This gap is itself
informative: part of that earlier-measured advantage likely came from
proxy signals learned on the abundant easy (covered) training data, not
structure specific to the hard population -- the same pattern seen in
the vanishing thread's isolated-population check, though unlike that
earlier mistake, THIS number is properly fresh-tested (train/test split
maintained throughout), not a training-accuracy artifact.

Recall is lopsided: 94.1% on true-small cases, only 20.6% on true-large
cases -- the model leans heavily toward predicting "small" and only
weakly discriminates the true-large minority in this population.
Feature importance shows no single dominant feature (A_p1, C_num_parts,
C_p3, C_c2 all contribute comparably) -- a messier signature than
Murnaghan's rule's clean reliance on leading parts.

Random inspection of true-small examples in this territory did not
suggest an obvious new pattern by eye.

## Final status

A real, modest, properly-validated effect (+6.5 points) exists in the
genuinely uncovered territory, smaller than earlier aggregate figures
suggested once correctly isolated. No further specific characterization
found via feature importance or manual inspection. This is a reasonable
stopping point for this deep-dive: the honest picture is now complete --
exact theorem (Rosas, narrow coverage), broader correlation (fat-hook,
Bessenrodt-Bowman), and a smaller-than-first-appeared but real residual
skill with no identified explanation, properly distinguished from an
artifact of training on the full population.

## Pipeline checkpoint (research.md step 5: revise the strategy)

Per research.md's explicit guidance -- "review the direction after ...
repeated inconclusive work" -- this thread has now had two rounds of
characterization attempts that did not converge on a clean pattern (the
fat-hook-in-added-value diagnostic error, corrected twice, and the
isolated deep-dive's undominated feature importance). This is the named
trigger condition for revising strategy, not a reason to force a further
guess.

Checked against conjecture.md's bar before closing: none of the
project's findings here qualify as a conjecture. Rosas's theorem is
already known (not new); the fat-hook correlation is a statistical
tendency, not a precise, quantified mathematical statement; the residual
has no describable pattern at all. Manufacturing a precise statement from
the undominated feature-importance signature would be exactly the kind
of arbitrary-exception conjecture research.md warns against. No
conjecture is proposed from this thread.

This closes the g<=1/g<=2 investigation as a complete, well-evidenced,
non-conjecture-yielding cycle. Per research.md step 1 ("choose a
question"), the next question is a genuine choice for the authors,
not a default continuation.
