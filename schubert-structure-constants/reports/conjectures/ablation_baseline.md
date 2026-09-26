# Ablation — what did the transformer + SAE add beyond plain predicate mining?

*Answers the expert's §10 request (`../EXPERT_RESPONSE.md`):
"the mechanistic contribution still needs an ablation … the decisive metrics
should be sample complexity, candidate rank, time to conjecture, rate of
rediscovering hidden known theorems."  All numbers reproduced by
`scripts/ablation_baseline.py`; raw metrics in
[`ablation_baseline.json`](ablation_baseline.json).  Run 2026-07-16, CPU-only.*

## Protocol and faithfulness

We re-ran the discovery conditions of the three laws found by the loop —
descent-union (v001 era), C1 (v002 era), C2 (v003 era) — **without the
model**, on the same data the historical runs used, with the same feature
battery (`schubml/interp/residual_features.py`: 21 enriched + 15 config
features, + `n_extra_descents` encoding the descent-union set condition; 37
features, 74 single-feature regions).  Faithfulness checks (bit-exact
replications of the historical artifacts):

* discovery positives sample: **59,979** (= `residual_mine_v2.json` exactly);
* C1-era positive scan: **125,773** (= C1 card's "initial scan" row exactly);
* C2 kills on the discovery sample: **133**; on the full 328,419-zero pool:
  **942** (= the recorded "942 C2-firing zeros");
* depth-5 tree ceiling on the same sample: bal-acc **0.743** (= recorded 0.743).

Zero pools are themselves model-free (sampler + classical filters), so this
isolates the mining step.  The v002-era pool = 31,184 test-split v002 zeros
passing grading+Bruhat+descent-union; the v001-style pool = the 93,347
passing grading+Bruhat only; the v003 pool = the 328,419 v003 residual zeros.
Junk assessment uses a **held-out positive pool of 83,689** ground-truth
positives from files/lines never touched by the mining samples (S6–S8;
every S9 line was already consumed by the discovery run).

## 1. Candidate rank (the expert's key question)

Single-feature zero-violation mining (the exact `residual_mine_v2.py`
protocol, kill threshold 30):

| era / target | candidates K | rank of law by kills | rank by simplicity | law kills | candidates surviving held-out validation |
|---|---:|---:|---:|---:|---|
| v003 / **C2** (`ess_n_both_tight > 0`) | **2** | **1** | 1 | 133 (0.30%) | 2/2 — both ARE C2 (duplicate feature) |
| v002 / **C1** (`ess_n_slack0 > 0`) | **17** | **4** | — | 637 (2.04%) | **4/17 — all four are C1 or C2** (each twice, duplicate features) |
| v001 / **descent-union** (`n_extra_descents > 0`) | **26** | **1** | — | 62,163 (66.6%) | 5/26 — du + the C1/C2 duplicates |

Findings:

* **C2 is found at rank 1, and the mathematician does not drown.**  The
  candidates outranking C1 at v002 (`rothe_n_rows > 5`, `ell_w > 27`, …) are
  size artifacts of the S5–S6 positive sample; a single held-out validation
  pass (83,689 larger positives — the step the pipeline always ran anyway)
  kills 13/17 of the v002 list and leaves **only the true laws standing** in
  all three eras.
* **C2 was already visible one era early**: on the v002 pool it appears at
  rank 2 (781 kills) — an iterated model-free miner (mine → validate → filter
  → re-mine) would have reached both C1 and C2 from v002-era data alone.
* One protocol nuance: the historical mechanism-independence flag marks C2
  `False` (perturb 0.11% vs random 0.85% kill rate).  A miner using that flag
  as a hard gate would have **rejected** C2; accepting it required the
  judgment that zero-violation over positives is the load-bearing property.

## 2. Sample complexity (5 random zero-subsamples per size; detection = 0
positive violations and >= 10 kills against the full positive sample)

| n zeros | C2 (v003 pool) | C1 (v002 pool) | descent-union (v001 pool) |
|---:|---|---|---|
| 500 | 0/5 (kills 0–4) | 3/5 (kills 7–15) | 5/5 (~320 kills) |
| 2,000 | 1/5 (kills 3–13) | 5/5 | 5/5 |
| 8,000 | 5/5 (kills 14–32), rank 1 | 5/5, rank 4 | 5/5, rank 1 |
| 32,000 | 5/5 (~92 kills), rank 1 | 5/5 (full pool 31,184: 637 kills) | 5/5 |
| full (328,419) | 5/5 (942 kills), rank 1 | — | 5/5 (62,163 kills) |

C2's region is rare (0.3% of residual zeros): plain mining needs **~8k
residual zeros** for reliable detection; C1 needs ~1–2k; descent-union is
detectable from a few hundred.  The historical 45k-zero sample was comfortably
above threshold.

## 3. Exhaustive low-complexity enumeration (association-rule burden)

All depth-<=2 conjunctions of integer-threshold literals (`f >= t`, `f <= t`,
t in 0..4) over the same battery; survivor = 0 positive violations and >= 10
kills:

| era | predicates enumerated | survivors K | distinct kill-sets | junk rate vs held-out | law rank by kills | law rank by simplicity |
|---|---:|---:|---:|---:|---:|---:|
| v003 (C2) | 23,653 | 364 | 92 | 1.4% | 19 | **1** |
| v002 (C1) | 31,878 | 845 | 183 | 0.5% | 189 (C2: rank 2) | 4 |
| v001 (du) | 34,453 | 1,786 | 628 | 0.7% | **1** | 2 |

* The kill-count ordering buries C1 under 188 conjunctions — but ~all of them
  are refinements/duplicates of the C2/C1 regions themselves (v003: 325/364
  survivors are killed-subsets of C2's region).  Under the natural simplicity
  ordering (depth, then threshold magnitude) C2 is **rank 1** and C1 rank 4.
* The false-discovery burden against held-out positives is small (0.5–1.4%)
  *on this battery* — because the battery is tiny (37 features) and heavily
  curated.  The burden numbers say the drowning risk on the curated features
  is modest; they say nothing about an uncurated DSL.
* Notable by-product: the strongest v003 survivor,
  `both_tight_full_row >= 1 AND both_tight_full_col <= 0` (1,253 kills, 0
  violations on 83,689 held-out positives), is a **genuine law** the pipeline
  never wrote down: a full both-tight interior row at i means
  u([i]) = v([i]) = w([i]), i.e. equal Grassmannian shadows
  \(\Lambda_i(u)=\Lambda_i(v)=\Lambda_i(w)=\nu\); MP\(_i\) then requires
  \(\rho\subseteq\nu\) with \(|\rho|=2|\nu|\), forcing \(\nu=\emptyset\) (the
  second literal excludes that block case).  So exhaustive mining surfaces
  *more* true shadow corollaries than the pipeline extracted — each requiring
  exactly the kind of triage/proof effort C2 received.

## 4. Random-feature control (SAE specificity)

On the cached `pqt_residual_v1` readout activations of the 59,441-triple
matched v004 set (the setting where SAE #653/#857 were found and later named
as the graded shadow-infeasibility score):

| feature set (1024 each) | max zero-AUC | # > 0.95 | # > 0.9 | best \|rho\| with MP-infeasibility |
|---|---:|---:|---:|---:|
| SAE codes | 0.9854 (#653) | **2** (cos 0.997 — one direction) | 3 | 0.587 |
| random linear projections | 0.9855 | 110 | 337 | 0.480 |
| random ReLU projections | 0.9788 | 34 | 148 | **0.586** |
| raw coordinates (256) | 0.9743 | 40 | — | — |

Honest reading: **random projections of the model's activations match the SAE
on both separation and shadow-correlation.**  The information is in the
trained representation, not uniquely in the SAE basis; a random direction
would also have exposed the rho ~ 0.59 shadow correlation.  What the SAE
added is *concentration*: a discrete shortlist (2 features above 0.95, and
they are the same direction) versus a diffuse continuum of 110+ partially
redundant random directions with no privileged ordering.  The load-bearing
artifact is the transformer's representation; the SAE is a target-selection
convenience on top of it.

## 5. ML ceilings on the same features (expert items 2–3)

Same v003 discovery sample, 70/30 split: depth-5 tree bal-acc 0.743 / MCC
0.487; **boosted trees 0.790 / MCC 0.574**; sparse logistic 0.754 / MCC 0.503
— versus the residual transformer's MCC **0.961**.  At the survivor frontier
the same gap: model pairwise win 0.9825 on count-defying pairs vs 0.8015 for
gradient boosting on all 95 hand features (`deep_forensics.json`).  Boosted
trees on the same features do *not* close the gap; the model separates far
beyond anything expressible in the battery, and that gap is measured, not
assumed.

## 6. The expert's four metrics, answered

| metric | plain mining (measured) | pipeline (recorded) |
|---|---|---|
| sample complexity | C2: ~8k residual zeros; C1: ~2k; du: <500 | same data sufficed (45k / 40k samples) |
| candidate rank | C2 1/2; C1 4/17 (range mining) or 19/364 and 189/845 by kills, 1 and 4 by simplicity (depth-2); after held-out validation only the laws survive | C2 surfaced as the unique candidate pair in the historical run |
| time to conjecture | ~2 min featurize + <1 s mine + ~2 s depth-2 per era (CPU), **given** the residual pool and the battery | per round: dataset build (~0.5 h) + 7–8 h GPU training + probing/SAE/mining (hours) |
| rediscovery rate | 3/3 laws rediscovered (du rank 1; C1 rank 4->2 after validation; C2 rank 1); plus one unextracted true law (equal-shadow row) | 3/3 via the loop, historically |

## 7. Verdict (the honest synthesis)

**Plain zero-violation mining on this feature battery finds C1, C2 and
descent-union without the model, cheaply and at high rank — the model was
not necessary for the *extraction step* of these three laws.**  Once the
residual pool and the 37-feature battery exist, mining is a few CPU minutes,
C2 is rank 1 of 2 candidates, and a single held-out validation pass removes
all junk.  Anyone claiming the transformer was indispensable *for these
specific predicates* is overclaiming, and the depth-2 enumeration even shows
one true shadow corollary the pipeline left on the table.

What the measured record says the model actually contributed:

1. **Dataset-construction guidance (the residual crank).**  The pools mined
   here are the pipeline's own artifacts: each era's zero pool exists because
   the previous model round validated the previous law hard enough to filter
   on it.  The mining step is trivial *conditional on* that ladder; the
   ladder itself was driven by the loop.  (An iterated model-free miner could
   plausibly have run the same crank for C1/C2 — measured: C2 already
   appears at rank 2 from v002-era data — but that is a counterfactual about
   discipline, not a measurement.)
2. **Feature-battery guidance.**  The enriched/config battery was written
   after the v002 model's circuit analysis localized the residual signal in
   rank-slack geometry at cross-attention (probes 0.72 -> 0.95).  The mining
   is only as good as its features, and the features encode where the model
   said to look.  This confound is post-hoc untestable and is the main
   caveat on "mining alone suffices."
3. **The existence proof of remaining signal — the model's clearest
   irreplaceable role.**  MCC 0.961 vs a 0.79 boosted-tree ceiling on the
   identical features is a measured guarantee that *undiscovered structure
   beyond the battery* exists; it is what justified building the pipe-dream
   battery, the survivor crank, and ultimately the shadow work.  No mining
   ceiling can certify its own incompleteness; the model does exactly that,
   era after era (currently: 0.986 vs 0.879 on the multi-cut frontier).
4. **Naming targets where the DSL is exhausted.**  At the survivor frontier
   the hand vocabulary correlated with the model's zero-detectors at only
   rho ~ 0.21; the shadow hypothesis raised this to 0.59 and named the
   feature.  The random-projection control shows the SAE basis per se was
   not the magic — the trained representation was — but the model-side
   evidence (a *nameable, monotone* score that survived every surrogate
   test) is what made the Grassmannian-shadow reading compelling enough to
   pursue to proof.

Calibrated one-liner for the papers: *the transformer is not needed to
extract low-complexity laws from a curated feature battery — zero-violation
mining does that in minutes; it is needed to (a) certify that signal remains
when the battery is exhausted, (b) decide which residual to build next, and
(c) point the feature engineering — and those three roles are exactly where
every law in this project actually came from.*

## Limitations

* The battery is model-era-curated; a fully blind ablation would need a
  pre-registered feature DSL, which no longer exists to run.
* Held-out junk validation uses 83,689 positives (S6–S8); the pipeline's own
  standard was 3.4–3.8M positives.  Junk rates here are upper-bound-honest
  for the small thresholds enumerated (integer 0..4) but range-bound
  candidates (e.g. `ell_w > 27`) already show the size-artifact failure mode.
* The random-feature control conditions on the trained model's activations;
  it quantifies SAE-vs-random within the pipeline, not model-vs-no-model
  (that comparison is the ML-ceilings table).

## Reproduce

```
PYTHONPATH=$PWD taskset -c 1,2 nice -n 8 python -u scripts/ablation_baseline.py --stage featurize-v003
...                                                                --stage featurize-v002
...                                                                --stage mine
...                                                                --stage sample-complexity
...                                                                --stage depth2
...                                                                --stage random-control
...                                                                --stage report
```

Stage fragments cache under `$ABLATION_CACHE` (scratchpad `ablation/`);
consolidated metrics: `reports/conjectures/ablation_baseline.json`.
