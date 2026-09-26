#!/usr/bin/env python
"""Train a sparse autoencoder on a chosen hook site (PROJECT.md 13.7).

Gathers activations at ``--hook`` over the dataset, trains a ReLU SAE with an L1
penalty (optional dead-feature resampling), and writes the trained weights plus
per-feature statistics (sparsity, mean/max activation, top activating rows).
Falls back to a fresh model + synthetic data when no run / dataset is present.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from schubml.interp._cli import default_report_dir, load_model_and_dataset
from schubml.interp.sae import (SAEConfig, feature_stats, gather_activations,
                                train_sae)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--split", default="product_disjoint:test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hook", default="cross_blocks.1.hook_resid_post",
                    help="hook site to autoencode")
    ap.add_argument("--readout-token", default=None,
                    help="for hook_readout: support|zero|known|count or slot int")
    ap.add_argument("--per-token", action="store_true",
                    help="keep every valid token row instead of pooling per example")
    ap.add_argument("--max-examples", type=int, default=4096)
    ap.add_argument("--expansion", type=int, default=4, help="d_hidden = expansion * d_in")
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--resample-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, mcfg, dataset = load_model_and_dataset(
        args.run, args.data_dir, device=args.device, split=args.split,
        limit=args.max_examples)

    rt = args.readout_token
    if rt is not None and rt.isdigit():
        rt = int(rt)
    acts = gather_activations(model, dataset, args.hook, device=args.device,
                              max_examples=args.max_examples, readout_token=rt,
                              per_token=args.per_token)
    d_in = acts.shape[1]
    cfg = SAEConfig(d_in=d_in, d_hidden=args.expansion * d_in, l1=args.l1,
                    tied=args.tied, steps=args.steps, resample_every=args.resample_every,
                    seed=args.seed,
                    device="cuda" if args.device.startswith("cuda") else "cpu")
    sae = train_sae(acts, cfg)
    final = sae.loss(acts.to("cpu"))
    stats = feature_stats(sae, acts, top_k=16)

    out_dir = args.out or default_report_dir(args.run)
    os.makedirs(out_dir, exist_ok=True)
    tag = args.hook.replace(".", "_")
    torch.save({"state_dict": sae.state_dict(), "cfg": vars(cfg), "hook": args.hook},
               os.path.join(out_dir, f"sae_{tag}.pt"))
    summary = {
        "hook": args.hook, "d_in": d_in, "d_hidden": cfg.d_hidden,
        "n_rows": int(acts.shape[0]),
        "final_recon": float(final["recon"]), "final_l0": float(final["l0"]),
        "n_dead": int(sum(1 for s in stats if s["n_active"] == 0)),
        "features": stats,
    }
    path = os.path.join(out_dir, f"sae_{tag}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[sae] wrote {path}  (d_in={d_in} d_hidden={cfg.d_hidden} "
          f"rows={acts.shape[0]})")
    print(f"  final recon={summary['final_recon']:.4f} L0={summary['final_l0']:.2f} "
          f"dead={summary['n_dead']}/{cfg.d_hidden}")


if __name__ == "__main__":
    main()
