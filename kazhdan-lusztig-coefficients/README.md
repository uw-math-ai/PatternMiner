# Stanley Problem 16: Kazhdan–Lusztig coefficients

This package records a failed threshold-learning experiment, proved structural reductions for a possible length-thirteen failure of combinatorial invariance, and exact controls supporting those reductions. The remaining coefficient inequality is an **open conjecture**.

- [Results, experiment details, and model configuration](paper/results.md)
- [Statements, proofs, dependencies, and open conjecture](paper/proofs.md)
- [Four figures: accuracy, failures, feature comparisons, and source sensitivity](graphs/README.md)
- [Source provenance](research/SOURCE_MAP.json) and [fresh exact replay](research/artifacts/publication_replay.json)

## Verify or reproduce

Python 3.13 and its standard library are sufficient. No data download, external model, GPU, or private checkout is required. From the repository root:

```bash
python3 kazhdan-lusztig-coefficients/tools/verify_bundle.py
python3 kazhdan-lusztig-coefficients/tools/replay.py --check-recorded
```

The first command checks file integrity, syntax, imports, local links, and the saved evidence without running mathematical experiments. The second recomputes all five selected exact checks and compares deterministic results with the publication replay. The interval computations may take several minutes. Use `--only e046`, `--only support`, `--only heredity`, `--only descent`, or `--only coefficient-eight` to select one check. Each check has a 600-second cap; an interrupted or timed-out command is a failure, not additional evidence.

Replay writes to stdout by default. `--output /path/to/new-result.json` writes a new file and refuses to overwrite existing data. Historical results stay separate from the publication replay.

## Scope and provenance

The source campaign checkpoint is `[withheld for anonymous review]`. This is a selected export supporting the paper's P16 claims, not the entire 62-ID campaign. It includes the E046 diagnostic and the controls for support factorization, genus heredity, descent recurrence, and the C018 coefficient-eight example. Later model diagnostics, incomplete searches, and unrelated exploratory branches are excluded.

Established length-twelve invariance and Warrington's coefficient spectrum are attributed to their sources. The deductions have internal research reviews; external peer review and publication novelty remain unresolved. Finite verification does not prove the remaining rank-twelve bound. The source map distinguishes unchanged mathematical definitions, adapted entry points, proof excerpts, and historical artifacts. Private paths, conversations, and orchestration files are excluded.

The [manuscript coverage map](paper/results.md#manuscript-coverage-and-accounting) locates each P16 claim in the supplied archive and records the label-attribution correction. The source map also indexes all 62 experiment IDs and their original report paths; these are not verified session or execution counts.
