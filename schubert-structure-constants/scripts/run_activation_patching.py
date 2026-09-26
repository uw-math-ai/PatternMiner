#!/usr/bin/env python
"""Activation / reverse patching over matched packs (PROJECT.md 13.3-13.4).

Builds a matched (positive ``w_+`` / hard-zero ``w_-``) batch pair sharing the
factor pair ``(u, v)``, patches each candidate site from the clean run into the
corrupt run, and reports the change in the positivity logit (and ``c_hat``).
``--reverse`` additionally patches corrupt into clean.  Writes a JSON ranking of
sites by effect size and an interpretation-card stub.
"""

from __future__ import annotations

import argparse
import json
import os

from schubml.interp._cli import (default_report_dir, load_model_and_dataset,
                                 matched_clean_corrupt)
from schubml.interp.patching import (activation_patch, default_patch_sites,
                                     reverse_patch)


def _rank(results: dict, key: str = "delta_pos_logit"):
    rows = [(name, abs(d.get(key, 0.0)), d.get(key, 0.0))
            for name, d in results.items()
            if name != "_baseline" and isinstance(d, dict) and key in d]
    rows.sort(key=lambda r: -r[1])
    return [{"site": n, "abs_delta": round(a, 5), "delta": round(d, 5)} for n, a, d in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--split", default="product_disjoint:test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--reverse", action="store_true", help="also run reverse patching")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, mcfg, dataset = load_model_and_dataset(
        args.run, args.data_dir, device=args.device, split=args.split)
    clean, corrupt = matched_clean_corrupt(dataset, device=args.device, n=args.n_pairs)
    sites = default_patch_sites(model)

    forward = activation_patch(model, clean, corrupt, sites, device=args.device)
    report = {
        "baseline": forward.get("_baseline", {}),
        "forward_patch_top": _rank(forward)[:25],
        "forward_patch_full": {k: v for k, v in forward.items() if k != "_baseline"},
    }
    if args.reverse:
        rev = reverse_patch(model, clean, corrupt, sites, device=args.device)
        report["reverse_patch_top"] = _rank(rev)[:25]
        report["reverse_baseline"] = rev.get("_baseline", {})

    out_dir = args.out or default_report_dir(args.run)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "activation_patching.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[patching] wrote {path}")
    print(f"  baseline clean/corrupt pos_logit: "
          f"{report['baseline'].get('clean_pos_logit')}/"
          f"{report['baseline'].get('corrupt_pos_logit')}")
    print("  top forward-patch sites (|delta pos_logit|):")
    for r in report["forward_patch_top"][:8]:
        print(f"    {r['delta']:+.4f}  {r['site']}")


if __name__ == "__main__":
    main()
