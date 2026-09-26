# Schubert witness selection: training and circuit analysis

The model passed all twelve development criteria at 99.52% or better. It then failed the independent difficult-pair assessment: 96.54% in the S9 population and 96.59% in S10, despite ordinary accuracy above 99.5% in every assessed population. Independent qualification remains unmet. The original formal circuit discovery and confirmation populations remain unopened.

This continuation contains actual training, exact data generation and causal interventions. The principal results are in RESULTS_SO_FAR.md; the intervention definitions, controls and limitations are in CIRCUIT_RESULTS.md. [FRESH_ASSESSMENT.json](FRESH_ASSESSMENT.json) records the failed independent evaluation. Subsequent evaluations on those populations are development evidence.

The research loop produced a useful representation change. An attention-path analysis suggested relative permutations, leading to 24 coordinates describing the images of simple roots under x y^{-1}, for x,y among u,v,w. The matched representation experiment improved the weakest development accuracy from 95.65% to 97.67%; subsequent refinement reached 99.52%. Removing the learned additional projection introduces 110 errors and corrects nine on the retained supplied-input diagnostic. The older neuron selection retains a substantial causal contribution after this representation change, but it does not give a Schubert multiplication rule.

The exact corpus contains 5,502,579 new training triples, including 71 geometrically screened errors confirmed by complete products. The combined corpus has 12,054,685 training triples and 47,972 difficult pairs. The failed assessment rows are not added to it. A subsequent matched experiment improves the minimum across fifteen development criteria from 97.23% to 98.00% by imposing independence of the supplied index. Adding simple-root action coefficients to attention and training further reaches 98.20%. This final continuation has no matched control for additional training. FINAL_MODEL_SELECTION.json records the selected checkpoint.

The full development reevaluation gives 284/289 difficult pairs in S9 and 980/998 in S10. The fixed circuit replay retains 79.88% of the mean isolated-path effect through the earlier 32 neuron pairs. By contrast, removing the added attention coefficients loses three correct answers and fixes two among 10,138 supplied inputs. The figure presents the development progression, failed independent assessment and circuit transfer separately.

The population names S7 through S10 refer to the sampled factors. Stable product outputs may require a larger common symmetric group; encoded triples have a common ambient size N at most 17.

Separately, [INCIDENCE_TRANSLATION_DEGREES.md](INCIDENCE_TRANSLATION_DEGREES.md) proves that the simple-root translation degrees of the incidence hypersurface, multiplied by the generic incidence degree, equal the corresponding ordinary Schubert coefficients. Thus the generic conormal supports equal the coefficient supports. This note makes no external priority claim. The manuscript is unchanged.

## Reproducing the retained computations

The recorded environment uses Python 3.11, PyTorch 2.6.0 with CUDA 12.4, NumPy 2.4.6 and lrcalc 2.1. The neural runs used one local NVIDIA T4. The mathematical and circuit scripts record checkpoint, source, data and output hashes. Training scripts retain snapshots of the implementation used by each run; these snapshots take precedence over later interface additions to the live modules.

The canonical experiment directory is this directory. `context.py` imports read-only helpers from the frozen predecessor without redirecting its paths or resource ledger. The complete research repository is needed for those dependencies.

Compressed exact data are retained in `artifacts/`. Verify them without constructing caches:

```bash
python rebuild_cache.py --verify-only
```

Restore the exact local data caches with:

```bash
python rebuild_cache.py
```

The default destination is `cache/data`. On the recorded machine, `cache` is an ignored symlink to the existing mounted data volume. A different machine can use a local directory or `--destination /path/to/cache/data`. Reconstruction checks every archive and concatenated-data hash and makes no exact-product queries. Frozen predecessor caches have their own reconstruction procedures.

Use `richardson_model.load(path)` for retained predictors. It supports the original additional representation, its zero-projection bypass, the relative-root features, independence of the supplied index and the additional root-action attention comparisons. `artifacts/root_action_training.pt` is the final development selection. `artifacts/qualified_development.pt` is the exact checkpoint used in the failed independent assessment. Neither is independently qualified.

`replay_selected_path.py` reuses the recorded neuron ranking and random subsets on a completed checkpoint report. It verifies native/explicit agreement, identity replacement and complete-input restoration. It is a development transfer test. A new run must use a fresh output name and remain within the recorded resources; the frozen assessment must not be reopened as independent evidence.

## Evidence and resources

[PRIOR_EVIDENCE_INTEGRITY.json](PRIOR_EVIDENCE_INTEGRITY.json) verifies 978 prior manifest entries and the unchanged manuscript. CORPUS_SEPARATION_AUDIT.json, FINAL_TRAINING_AUDIT.json and CACHE_ARCHIVE_AUDIT.json record the data checks. The final evidence manifest records the versioned files and cumulative resource audit when this continuation is frozen.

The resource agreement is cumulative across continuations. It allows 39,600 charged GPU seconds, 50,400 total timed seconds, 85,000 exact-product queries and 24 GB of experiment storage, with no paid services. The ledger retains failed attempts and the intentional interruption after development qualification. It is not reset by starting a new experiment directory.
