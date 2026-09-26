#!/usr/bin/env python
"""Audit the split-tag distribution over an existing ``triples.parquet``.

PROJECT.md 6.7 / 16 Milestone 2: ``build_dataset`` writes mathematically
meaningful splits as per-triple ``split_tags`` (``product_disjoint:train/val/test``,
``ambient_ood:train/test``, ``family_<fam>:train/test``, ``coeff_mag:train/test``,
``hardzero:train/test``).  This wrapper does not *create* splits -- it audits the
ones already produced by reporting, for every split-tag family, the count of
triples carrying each value, broken down by positive / negative.

A leakage sanity check is included: the count of product ids that appear under
more than one ``product_disjoint`` value (should be zero -- splits are
product-disjoint by construction).

Runs under the ``schubml`` env.  Example:

    python scripts/build_splits.py --triples data/processed/v001/triples.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict


def _split_family(tag: str) -> str:
    """``"product_disjoint:train"`` -> ``"product_disjoint"``."""
    return tag.split(":", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triples", required=True, help="triples.parquet to audit")
    ap.add_argument("--out", default=None, help="optional path to write the audit JSON")
    args = ap.parse_args()

    import pandas as pd

    # split_tags is stored JSON-encoded; product_id/positive are native scalars.
    df = pd.read_parquet(args.triples, columns=["split_tags", "positive", "product_id"])
    n = len(df)
    print(f"[splits] {n} triples in {args.triples}")

    # per split-tag-family -> value -> {total, positive, negative}
    families: dict = defaultdict(lambda: defaultdict(lambda: {"total": 0, "positive": 0, "negative": 0}))
    # product-disjoint leakage check: product_id -> set of pd values
    pd_values_by_product: dict = defaultdict(set)

    positives = df["positive"].to_numpy()
    product_ids = df["product_id"].to_numpy()
    for i, raw_tags in enumerate(df["split_tags"].to_numpy()):
        tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
        is_pos = bool(positives[i])
        for tag in tags:
            fam = _split_family(tag)
            cell = families[fam][tag]
            cell["total"] += 1
            cell["positive" if is_pos else "negative"] += 1
            if fam == "product_disjoint":
                pd_values_by_product[int(product_ids[i])].add(tag)

    leaked = [pid for pid, vals in pd_values_by_product.items() if len(vals) > 1]

    audit = {
        "triples_path": os.path.abspath(args.triples),
        "n_triples": n,
        "n_positive": int(positives.sum()),
        "n_negative": int(n - positives.sum()),
        "split_families": {
            fam: {tag: dict(cell) for tag, cell in sorted(vals.items())}
            for fam, vals in sorted(families.items())
        },
        "product_disjoint_leakage": {
            "n_products": len(pd_values_by_product),
            "n_products_in_multiple_splits": len(leaked),
            "leaked_product_ids_sample": leaked[:20],
        },
    }

    # human-readable print
    for fam, vals in audit["split_families"].items():
        line = "  ".join(f"{tag}={cell['total']}" for tag, cell in vals.items())
        print(f"[splits] {fam:<18} {line}")
    leak_n = audit["product_disjoint_leakage"]["n_products_in_multiple_splits"]
    print(f"[splits] product-disjoint leakage: {leak_n} products in >1 split "
          f"({'OK' if leak_n == 0 else 'LEAK!'})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"[splits] wrote audit -> {args.out}")


if __name__ == "__main__":
    main()
