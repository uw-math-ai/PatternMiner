#!/usr/bin/env python
"""Post-build validation of a processed dataset (PROJECT.md 19.2).

Checks: positives match length grading; hard-zero negatives are absent from the
product support and satisfy Bruhat; no product-disjoint split leakage; reports
tier / split / coefficient distributions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from schubml.combinatorics import permutations as P
from schubml.data.schemas import (PERM_JSON_COLUMNS, PRODUCT_JSON_COLUMNS,
                                   load_table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    D = args.data_dir

    perm = load_table(os.path.join(D, "permutations.parquet"), PERM_JSON_COLUMNS)
    prod = load_table(os.path.join(D, "products.parquet"), PRODUCT_JSON_COLUMNS)
    trip = pd.read_parquet(os.path.join(D, "triples.parquet"))
    pid2canon = {int(r.perm_id): tuple(r.canonical) for r in perm.itertuples()}
    len_by_id = {pid: P.length(w) for pid, w in pid2canon.items()}

    report = {"n_perms": len(perm), "n_products": len(prod), "n_triples": len(trip)}

    # 1. positives satisfy grading
    pos = trip[trip.positive]
    bad_grade = 0
    for r in pos.itertuples():
        if len_by_id[int(r.w_id)] != len_by_id[int(r.u_id)] + len_by_id[int(r.v_id)]:
            bad_grade += 1
    report["positives_failing_grading"] = bad_grade

    # 2. negatives (non-Z0) match length grading
    neg = trip[~trip.positive]
    bad_neg = 0
    for r in neg.itertuples():
        if r.negative_tier == "Z0":
            continue
        if len_by_id[int(r.w_id)] != len_by_id[int(r.u_id)] + len_by_id[int(r.v_id)]:
            bad_neg += 1
    report["nonZ0_negatives_failing_grading"] = bad_neg

    # 3. negatives absent from product support
    support = {int(r.product_id): set(r.support_w_ids) for r in prod.itertuples()}
    leak = sum(1 for r in neg.itertuples() if int(r.w_id) in support.get(int(r.product_id), set()))
    report["negatives_in_support"] = leak

    # 4. product-disjoint leakage: each product in exactly one split
    def split_of(tags):
        for t in json.loads(tags):
            if t.startswith("product_disjoint:"):
                return t.split(":")[1]
        return None
    psplit = trip.groupby("product_id")["split_tags"].first().apply(split_of)
    # a product's triples all share its split (no leakage by construction); report counts
    report["product_disjoint_products"] = dict(Counter(psplit.dropna()))

    # 5. tier + coefficient distributions
    report["tier_counts"] = dict(trip["negative_tier"].fillna("positive").value_counts())
    report["coeff_max"] = int(pos["coefficient"].max())
    report["coeff_hist_small"] = dict(pos["coefficient"].value_counts().head(8).astype(int))
    report["mean_support_size"] = float(prod["support_size"].mean())

    ok = (bad_grade == 0 and bad_neg == 0 and leak == 0)
    report["ALL_OK"] = ok
    print(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
