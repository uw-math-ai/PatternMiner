# E072: beta221 construction-directed learning pilot

Start with [results](results.md) or the [paper summary](../../paper/beta221_learning_pilot.md).

**The learned map failed injectivity.** Training/validation/test excess
collisions were0/1/5. The test random-score median was4, so no learning
advantage is established. Exact checks support97 positive objects,
50 negative objects and47 finite-lookup survivors across30 ordinary targets.
This is a local component of the Problem12 counting-construction route,
not a solution of the problem or a new geometric injection.

## What is included

- The [prospective protocol](protocol.md), [root preflight](root-preflight.md)
  and results report. Independent E072 code/result review is still pending.
- The frozen complete gzip record, SHA256 sidecar and all model weights,
  candidate features, choices, collisions, exact-check records and lookup.
- The original runner snapshot and the two core tableau/object modules.
- The [signed reduction](beta221-working-route.md), its [review](working-route-review.md),
  the [negative domain](negative-corner-domain.md) and its [review](negative-domain-review.md).
  These earlier reviews do not certify the new learning experiment.
- [Export provenance](EXPORT.json) distinguishing verbatim files from two
  publication path normalizations. Original hashes are not rewritten.

## Reproduce the saved-record audit

From the repository root, using Python3.10 or later:

```bash
python3 character-table-row-sums/tools/verify_e072.py
python3 character-table-row-sums/tools/verify_bundle.py --check-imports
```

The first command recomputes predictions from saved feature matrices and
weights, checks split isolation, selection, collision witnesses, inverses,
survivor complements, and available source pins. It performs no training,
LR enumeration, coefficient query or scientific rerun. It does not establish
the correctness of the underlying feature construction or exact oracle.

## Historical replay limits

`historical_runner.py` is an unchanged source snapshot, NOT a runnable
entry point in this directory. Its original project layout, budget agreement,
and complete120-source snapshot are intentionally not copied wholesale.
The saved record retains127 original hash pins, including references to
excluded files. This export checks the included subset, not all127 files.
The historical command in the report describes the original run only.

A new end-to-end scientific replication would require a scoped dependency
export, fresh approved budget/protocol, and a new run identifier. This packet
does not claim that replication or import closure of the historical runner.
Do not run it against the frozen publication artifacts.

The E039–E071 archive is now available separately through the
continuation index, not duplicated in this
E072 supplement. No unrelated conjecture results, private meeting text,
external model or local workspace history is included. The complete
gzip record is retained verbatim, not pruned to remove failed assessments.
