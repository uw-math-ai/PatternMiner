#!/usr/bin/env sage-python
"""Generate Schubert structure-constant products with SageMath.

Runs under the ``sage`` conda env.  For each (unordered) product pair ``(u, v)``
of canonical permutations fitting in ``S_n`` it computes

    S_u * S_v = sum_w c_{u,v}^w S_w

via ``SchubertPolynomialRing(ZZ)`` and writes one JSON line per product:

    {"u": [...], "v": [...], "n_compute": int, "support": [[[w...], c], ...]}

Permutations are canonicalised (trailing fixed points trimmed).  Commutativity
is used to store each unordered pair once.  Supports sharding and resume.

Usage:
    sage-python generate_products_sage.py --n 5 --mode exhaustive --out raw/s5.jsonl
    sage-python generate_products_sage.py --n 6 --mode sampled --num-pairs 15000 \
        --out raw/s6_sampled.jsonl --seed 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

from sage.all import ZZ, SchubertPolynomialRing  # type: ignore

X = SchubertPolynomialRing(ZZ)


def canonical(w):
    """Trim trailing fixed points; identity -> (1,)."""
    w = tuple(int(x) for x in w)
    k = len(w)
    while k > 1 and w[k - 1] == k:
        k -= 1
    return w[:k]


def length(w):
    n = len(w)
    return sum(1 for i in range(n) for j in range(i + 1, n) if w[i] > w[j])


def canonical_reps(n):
    """All distinct canonical permutations fitting in S_n (== |S_n| of them)."""
    seen = set()
    reps = []
    for p in itertools.permutations(range(1, n + 1)):
        c = canonical(p)
        if c not in seen:
            seen.add(c)
            reps.append(c)
    # sort by (length, tuple) for stable, curriculum-friendly ordering
    reps.sort(key=lambda w: (length(w), w))
    return reps


def multiply(u, v):
    """Return {canonical_w: int_coeff} for S_u * S_v."""
    prod = X(list(u)) * X(list(v))
    out = {}
    for k, c in prod.monomial_coefficients().items():
        ck = canonical(tuple(k))
        out[ck] = out.get(ck, 0) + int(c)
    # drop any zero coefficients defensively
    return {k: c for k, c in out.items() if c != 0}


def iter_pairs(reps, mode, num_pairs, seed):
    """Yield unordered (u, v) index pairs."""
    m = len(reps)
    if mode == "exhaustive":
        for i in range(m):
            for j in range(i, m):
                yield i, j
    elif mode == "sampled":
        rng = random.Random(seed)
        emitted = set()
        target = num_pairs
        tries = 0
        while len(emitted) < target and tries < target * 20:
            i = rng.randrange(m)
            j = rng.randrange(m)
            tries += 1
            if i > j:
                i, j = j, i
            if (i, j) in emitted:
                continue
            emitted.add((i, j))
            yield i, j
    else:
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mode", choices=["exhaustive", "sampled"], default="exhaustive")
    ap.add_argument("--num-pairs", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    reps = canonical_reps(args.n)

    # resume: count existing lines
    done = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = sum(1 for _ in f)

    all_pairs = list(iter_pairs(reps, args.mode, args.num_pairs, args.seed))
    # shard
    pairs = [p for idx, p in enumerate(all_pairs) if idx % args.num_shards == args.shard]
    total = len(pairs)

    print(f"[gen] S_{args.n} mode={args.mode} reps={len(reps)} pairs(shard {args.shard}/"
          f"{args.num_shards})={total} resume_from={done}", flush=True)

    t0 = time.time()
    with open(args.out, "a") as f:
        for idx, (i, j) in enumerate(pairs):
            if idx < done:
                continue
            u, v = reps[i], reps[j]
            support = multiply(u, v)
            n_compute = max([args.n] + [len(w) for w in support] + [len(u), len(v)])
            rec = {
                "u": list(u),
                "v": list(v),
                "n_compute": n_compute,
                "support": [[list(w), c] for w, c in sorted(support.items())],
            }
            f.write(json.dumps(rec) + "\n")
            if (idx + 1) % args.log_every == 0:
                rate = (idx + 1 - done) / max(1e-9, time.time() - t0)
                print(f"[gen] {idx+1}/{total} rate={rate:.1f}/s "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                f.flush()
    meta = {
        "n": args.n,
        "mode": args.mode,
        "num_pairs_shard": total,
        "seed": args.seed,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "sage_version": _sage_version(),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out + ".meta.json", "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"[gen] DONE S_{args.n} shard {args.shard}: {total} products in "
          f"{time.time()-t0:.0f}s -> {args.out}", flush=True)


def _sage_version():
    try:
        from sage.version import version  # type: ignore
        return version
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
