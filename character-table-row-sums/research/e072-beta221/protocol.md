# E072: origin-sensitive beta221 matching pilot

Prospective protocol, 2026-09-21. Authorized by the authors' instruction to
follow the direct beta221 construction route. This replaces the proposed
generic LR classification experiment, which was never run. Local only.

## Question and scope

Can a small, transparent, learned priority rule send every negative object
of T221=(s22+s211)h2[s21]-s1*h3[s21] to a distinct positive object of the
same output, without a per-output matching lookup? Ordinary outputs have
degree ten and first part at most five; actual shapes are (5,nu), degree
fifteen. This is a bounded construction component, not the full row sum.

The complete four square origins and sixteen cube survivors are inherited,
not learned. Generate all eight positive LR families and all 56 negative
corner objects, including the six virtual sources retained separately.
Cross-check LR generators against a separate brute-force word predicate.
Check all 30 ordinary signed counts against full h3[s221] via MN and JT.
Do not query a virtual shape as an ordinary ambient shape.

## Learning task

Every same-output positive object is locally legal. The nontrivial target
is simultaneous injectivity, not binary edge validity. Do not supervise
against an arbitrarily chosen matching as the sole correct answer.
Fit an integer linear priority scorer by bounded coordinate search to
minimize collisions between independently chosen images. It is a small
learned ranking model, not a neural network or mechanistic circuit study.

Features compare old and new skew cells, their entries, the added corner,
and named primitive origins. No output-shape ID, coefficient or arbitrary
copy index is a feature. Keep fixed lexicographic tie-breaking explicit.
All alternatives are scored; independent argmax choices receive no
collision repair. A separate lexicographic global assignment is an
existence/lookup baseline only, not evidence for a natural rule.

Split by the complete output partition using SHA256('E072:'+comma-separated
parts) modulo five: 0 validation, 1 test, 2/3/4 training. All sources and
candidates of an output remain together. All counts are generated for
integrity checks, but only training/validation choices select weights.
The author has prior symbolic exposure to this finite family; this is
algorithmically held-out testing, not blind mathematical discovery or
size/family extrapolation.

Three seeds (1729,2718,31415), 600 coordinate proposals each. Weights
are integers in [-4,4], initially zero. Proposals change one coordinate
by +/-1, with 10% random coordinate resets. Accept only a strict
lexicographic improvement in (training collisions, sum absolute weights).
Retain the best weights per seed; select by validation collisions, then
training collisions, L1 norm and seed. No post-test retraining in this run.
This deliberately small optimizer can stall; failure does not refute
existence of a suitable priority rule.

Baselines: constant lexicographic priority; fixed geometric overlap and
entry-agreement priority; 100 deterministic random weight vectors;
lexicographic global assignment. Report extra colliding sources, fully
injective output groups, and source counts separately for each split.
Finite reliability gate: zero missing choices and zero collisions across
all 50 ordinary sources, including test. Only a pass permits promotion to
a candidate finite map, still requiring separate mathematical review.

Interpretation: report all nonzero weights and leave-one-feature-out
ablation on validation and test. Select the strongest validation ablation
before inspecting test ablation effects. This is exact feature dependence
of a linear priority rule, not activation patching or proof of an abstract
mathematical algorithm. Retain complete collision witnesses and a repaired
lookup certificate separately. Do not attribute inherited mathematics or
posthoc matching to the learner.

## Resources, checks and outputs

One process/thread, CPU only; the finite integer search does not justify
GPU allocation. No external model/API/network/install. Use the existing
RunBudget, at most 90 wall/CPU seconds, 512 MiB address space, and at most
500,000 additional artifact bytes within the unchanged cumulative 100 MB
cap. Existing remaining artifact space is 704,454 bytes. No frozen source
is edited or added under src; new code is under scripts.

Pin all 120 existing source files and the symbolic inputs before/after.
Preserve the sole attempt and failures. Save complete objects, candidate
features, split assignments, weights, predictions, ablations, exact checks,
inverse lookup and survivors in compressed JSON, with a SHA256 sidecar.
The source and protocol are hashed before execution and saved in the record.
No independent reviewer is claimed for this new pilot; root preflight and
executable cross-checks do not equal E071's independent review status.
Scientific failures and an unsuccessful learned rule are reportable results.
