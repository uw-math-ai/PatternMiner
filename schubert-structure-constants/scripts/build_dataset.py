#!/usr/bin/env python
"""Build the processed dataset (tables + negatives + splits + packs).

Runs under the ``schubml`` env.  Example:

    python scripts/build_dataset.py \
        --raw data/raw/s3.jsonl:3 data/raw/s4.jsonl:4 data/raw/s5.jsonl:5 \
              data/raw/s6_shard0.jsonl:6 data/raw/s6_shard1.jsonl:6 \
              data/raw/s6_shard2.jsonl:6 data/raw/s7_sampled.jsonl:7 \
        --out data/processed/v001 --workers 4
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from schubml.data.build import NegConfig, SplitConfig, build_dataset


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True,
                    help="list of path:source_n entries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--with-transition", action="store_true")
    ap.add_argument("--n-random-words", type=int, default=8,
                    help="random reduced words per perm (lower = less build memory)")
    ap.add_argument("--z2", type=int, default=4)
    ap.add_argument("--z3", type=int, default=4)
    ap.add_argument("--z4", type=int, default=2)
    ap.add_argument("--z5", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=120)
    ap.add_argument("--ambient-train-max-n", type=int, default=5)
    ap.add_argument("--max-per-source", nargs="*", default=None,
                    help="caps like 6:60000 7:8000 to down-sample a source S_n")
    args = ap.parse_args()

    paths_with_n = []
    for entry in args.raw:
        path, _, n = entry.rpartition(":")
        paths_with_n.append((path, int(n)))

    neg_cfg = NegConfig(
        per_tier={"Z0": 1, "Z1": 1, "Z2": args.z2, "Z3": args.z3, "Z4": args.z4, "Z5": args.z5},
        max_candidates=args.max_candidates,
    )
    split_cfg = SplitConfig(ambient_ood_train_max_n=args.ambient_train_max_n)

    max_per_source = None
    if args.max_per_source:
        max_per_source = {}
        for item in args.max_per_source:
            n, cap = item.split(":")
            max_per_source[int(n)] = int(cap)

    build_dataset(
        paths_with_n, args.out,
        neg_cfg=neg_cfg, split_cfg=split_cfg,
        backend="sage", backend_version="10.9",
        commit=git_commit(),
        datetime_utc=datetime.now(timezone.utc).isoformat(),
        n_workers=args.workers, with_transition=args.with_transition,
        n_random_words=args.n_random_words, max_per_source=max_per_source,
    )


if __name__ == "__main__":
    main()
