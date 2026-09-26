#!/usr/bin/env python
"""Run the weak + classical-ML baselines and print a summary table.

PROJECT.md 11 / 12.1: trains majority, length-only, length+descents/Rothe
logistic, logistic-regression, random-forest, XGBoost and MLP classifiers on the
``product_disjoint:train`` split and evaluates them on ``product_disjoint:test``,
reporting positivity metrics on the easy (Z0/Z1) and hard (Z2--Z5) zero slices
*separately* so shortcut learning is visible.  A shortcut audit compares the
length-only baseline to the best full-feature model on the hard slice.

Runs under the ``schubml`` env.  Example:

    python scripts/run_baselines.py --data-dir data/processed/v001 \
        --out runs/baselines/v001_metrics.json
"""

from __future__ import annotations

import argparse

from schubml.experiments.baselines import format_summary, run_baselines
from schubml.experiments.heldout import known_rule_baseline_monk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True,
                    help="processed dataset dir (with permutations/triples parquet)")
    ap.add_argument("--out", default=None, help="path to write the metrics JSON")
    ap.add_argument("--max-train", type=int, default=200000,
                    help="cap on number of training triples (default 200000)")
    ap.add_argument("--max-test", type=int, default=100000,
                    help="cap on number of test triples (default 100000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-monk", action="store_true",
                    help="also run the exact Monk known-rule support baseline")
    args = ap.parse_args()

    results = run_baselines(
        args.data_dir, args.out,
        max_train=args.max_train, max_test=args.max_test, seed=args.seed,
    )
    print(format_summary(results))

    if args.with_monk:
        monk = known_rule_baseline_monk(args.data_dir)
        print()
        print("Monk known-rule support baseline (PROJECT.md 11.4):")
        print(f"  products matched : {monk['n_products']}")
        print(f"  support P/R/F1   : {monk['support_precision']:.3f} / "
              f"{monk['support_recall']:.3f} / {monk['support_f1']:.3f}")
        print(f"  coeff exact acc  : {monk['coeff_exact_acc']:.3f}")

    if args.out:
        print(f"\n[baselines] wrote {args.out}")


if __name__ == "__main__":
    main()
