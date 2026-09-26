# Stanley Problem 18: monomial immanants

This package records the width-two coefficient positivity theorem, exact checks supporting its proof, and a separate E031 small-model diagnostic with its failures. For every nonempty skew shape, the associated immanant character has a nonnegative Stanley–Stembridge decomposition whenever the coefficient partition has second part at most two. Coefficient partitions outside that class remain outside the theorem.

- [Results and exact experiment domains](paper/results.md)
- [Statement, normalization, and complete proof with supporting lemmas](paper/proofs.md)
- [Five figures: model diagnostics and exact decomposition distributions](graphs/README.md)
- [Source provenance](research/SOURCE_MAP.json) and [fresh exact replay](research/artifacts/publication_replay.json)

## Verify or reproduce

Python 3.13 and its standard library are sufficient. From the repository root:

```bash
python3 monomial-immanants/tools/verify_bundle.py
python3 monomial-immanants/tools/replay.py --check-recorded
```

The first command checks publication integrity, syntax, imports, local links, and saved evidence. The second reruns the finite mathematics and model fitting and compares deterministic outputs with the recorded publication replay. Select `--only width-two`, `--only rectangles`, or `--only origin` for one check. Each has a 600-second cap. No data download, external model, GPU, or private checkout is required.

Replay prints JSON by default. `--output /path/to/new-result.json` writes a new file and refuses to overwrite existing results. The historical artifacts remain unchanged.

## Scope and provenance

The source campaign checkpoint is `[withheld for anonymous review]`. This export contains the T053 proof, necessary supporting lemmas, independent finite checker, and five explicit rectangle decompositions. The original internal review accepted the proof after a descent-index clarification. External peer review and mathematical priority remain unresolved.

The source map distinguishes historical results, adapted read-only checkers, and formatted proof excerpts, and indexes the campaign's experiment reports. The publication replay freshly reproduces the selected exact claims. It does not turn the finite census into an all-parameter proof. The E031 replay reuses frozen labels; it does not regenerate their originating census. Fiber-connectivity searches, unrelated exploratory branches, private paths, and orchestration material are excluded. This is a focused export, not all 32 experiment IDs from the source campaign. No learned model is credited with the theorem. See the [manuscript coverage map](paper/results.md#manuscript-coverage-and-accounting) for claims, experiment locations, and attribution corrections.
