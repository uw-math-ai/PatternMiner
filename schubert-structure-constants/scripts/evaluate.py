#!/usr/bin/env python
"""Evaluate a trained model and produce the required evaluation table.

PROJECT.md 12.2: random / product-disjoint / hard-zero Z2 / Z3 / Z4-Z5 /
ambient-OOD / family-holdout / coefficient-magnitude / support-ranking, with
metrics stratified by negative tier.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from schubml.data.datasets import build_perms_by_id, load_triple_arrays
from schubml.data.labels import TIERS
from schubml.data.schemas import PERM_JSON_COLUMNS, load_table
from schubml.train.inference import load_run, predict
from schubml.train.evaluation_contract import validate_contract
from schubml.train.logging import save_json, write_model_card
from schubml.train.metrics import (coefficient_metrics, mod_accuracy,
                                    positivity_metrics, support_ranking_metrics)


def metrics_block(pred, mask=None):
    if mask is not None:
        pred = {k: v[mask] for k, v in pred.items() if isinstance(v, np.ndarray) and len(v)}
    if len(pred.get("positive", [])) == 0:
        return {"n": 0}
    out = {}
    out.update({f"pos_{k}": v for k, v in positivity_metrics(pred["pos_logit"], pred["positive"]).items()})
    bin_true = None
    out.update({f"coeff_{k}": v for k, v in coefficient_metrics(
        pred["c_hat"], pred["coefficient"].astype(int),
        pred.get("bin_pred") if len(pred.get("bin_pred", [])) else None, bin_true).items()})
    modp = {p: pred[f"mod{p}_pred"] for p in (2, 3, 5, 7) if len(pred.get(f"mod{p}_pred", []))}
    out.update(mod_accuracy(modp, pred["coefficient"].astype(int)))
    out.update({f"support_{k}": v for k, v in support_ranking_metrics(
        pred["support_score"], pred["positive"], pred["coefficient"], pred["product_id"]).items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True, help="training output dir")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=120000)
    ap.add_argument("--split", default=None,
                    help="Test tag; defaults to the recorded training regime's test tag")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "eval")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(args.run, "run_config.json")) as f:
        run_config = json.load(f)
    train_tag = run_config.get("args", {}).get("train_split", "")
    selected_split = args.split or train_tag.rsplit(":", 1)[0] + ":test"
    contract = validate_contract(run_config, args.data_dir, selected_split)

    model, mcfg, tok = load_run(args.run, args.ckpt, args.device)
    perm_df = load_table(os.path.join(args.data_dir, "permutations.parquet"), PERM_JSON_COLUMNS)
    perms_by_id = build_perms_by_id(perm_df)
    trip = os.path.join(args.data_dir, "triples.parquet")

    def run(split):
        validate_contract(run_config, args.data_dir, split)
        arr = load_triple_arrays(trip, split=split)
        if args.limit and len(arr["w_id"]) > args.limit:
            idx = np.random.default_rng(0).choice(len(arr["w_id"]), args.limit, replace=False)
            arr = {k: v[idx] for k, v in arr.items()}
        return predict(model, arr, perms_by_id, tok, n_count_queries=mcfg.n_slots, device=args.device)

    results = {}
    # Only evaluate the regime actually excluded by this checkpoint's training.
    pd_pred = run(selected_split)
    selected_key = selected_split.replace(":", "_")
    results[selected_key] = metrics_block(pd_pred)
    results["evaluation_contract"] = contract
    # by tier within product-disjoint test
    results["by_tier"] = {}
    for tid, tname in enumerate(TIERS):
        mask = (pd_pred["tier_id"] == tid)
        # include positives with each negative tier for a meaningful pos/neg split
        m = mask | (pd_pred["positive"] > 0)
        results["by_tier"][tname] = metrics_block(pd_pred, m)
    # hard-zero tiers explicitly
    for label, tids in [("hardzero_Z2", [3]), ("hardzero_Z3", [4]), ("hardzero_Z45", [5, 6])]:
        m = np.isin(pd_pred["tier_id"], tids) | (pd_pred["positive"] > 0)
        results[label] = metrics_block(pd_pred, m)
    save_json(results, os.path.join(out, "metrics.json"))
    write_model_card(
        os.path.join(out, "model_card.md"),
        architecture=vars(mcfg),
        training={"run_dir": args.run, "ckpt": args.ckpt or "best"},
        metrics=results,
        known_failures="Only the recorded training regime is evaluated. Other holdouts require separately trained checkpoints.",
    )
    print("[eval] wrote", os.path.join(out, "metrics.json"))
    for k in (selected_key, "hardzero_Z45"):
        print(k, {kk: round(vv, 3) for kk, vv in results[k].items()
                  if isinstance(vv, float) and kk.startswith("pos_")})


if __name__ == "__main__":
    main()
