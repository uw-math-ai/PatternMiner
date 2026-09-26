#!/usr/bin/env python
"""Methodological ablation (EXPERT_RESPONSE.md section 10): what did the
transformer + SAE pipeline add beyond plain predicate mining on the same
engineered features?

We reconstruct the discovery conditions of the three laws found by the
pipeline -- descent-union (v001 era), C1 = essential-tight vanishing
(v002 era), C2 = both-tight essential vanishing (v003 era) -- WITHOUT the
model, and run the identical zero-violation mining plus an exhaustive
low-complexity predicate enumeration.  The expert's four metrics:

    sample complexity, candidate rank, time to conjecture,
    rate of rediscovering hidden known theorems.

Stages (results cached under $ABLATION_CACHE, JSON fragments per stage):

  featurize-v003   featurize the FULL v003 residual-zero pool (328k) + the
                   bit-exact positives sample of the historical discovery run
                   (scripts/residual_mine_v2.py, seed 0) + a large held-out
                   positive pool from files/lines never used in mining.
  featurize-v002   v002-era residual zeros (grading+Bruhat+descent-union;
                   the actual v002 test-split zeros C1 was mined from), a
                   v001-style pool (grading+Bruhat only, for descent-union
                   rediscovery), and the era's positive sample
                   (scripts/residual_necessary_scan.py protocol).
  mine             single-feature zero-violation mining per era; rank of the
                   target law among candidates (by kill count and by
                   simplicity); wall-clock; ML ceilings (tree / boosted trees /
                   sparse logistic) on the same features.
  sample-complexity  detection frequency of each law at
                   n_zeros in {500, 2k, 8k, 32k, full} x 5 subsamples.
  depth2           exhaustive depth<=2 conjunctions over integer-threshold
                   literals; number of surviving zero-violation candidates
                   (triage burden K), rank of each law, false-discovery rate
                   against the held-out positive pool.
  random-control   random projections of the pqt_residual_v1 readout
                   activations vs the SAE features (zero-AUC + correlation
                   with the shadow-infeasibility features that named #653/#857).
  report           assemble reports/conjectures/ablation_baseline.json.

CPU only.  Run under `taskset -c 1,2 nice -n 8`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import get_context

import numpy as np

from schubml.combinatorics import bruhat
from schubml.combinatorics import permutations as P
from schubml.interp.residual_features import (config_feature_names, config_features,
                                              enriched_features, feature_names)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("ABLATION_SCRATCH", os.path.join(os.path.dirname(__file__), "..", "scratch"))
CACHE = os.environ.get("ABLATION_CACHE", os.path.join(SCRATCH, "ablation"))
NAMING_CACHE = os.environ.get("SAE_NAMING_CACHE", os.path.join(SCRATCH, "naming"))
FORENSICS_CACHE = os.environ.get("FORENSICS_CACHE", os.path.join(SCRATCH, "forensics"))

V003_ZEROS = "/mnt/data/processed/v003/residual_zeros.jsonl"
V002_DIR = "/mnt/data/processed/v002"

# battery = the exact feature set of the discovery runs (enriched + config)
# plus n_extra_descents, the set-condition encoding of descent-union used for
# the v001-era rediscovery test (identically 0 downstream of v002 filtering).
BATTERY_NAMES = feature_names() + config_feature_names() + ["n_extra_descents"]

# the three laws, as (feature, direction) regions positives never occupy
TARGETS = {
    "C2_both_tight": ("ess_n_both_tight", ">"),
    "C1_ess_slack0": ("ess_n_slack0", ">"),
    "descent_union": ("n_extra_descents", ">"),
}


def featurize_triple(t):
    u, v, w = t
    d = enriched_features(u, v, w)
    d.update(config_features(u, v, w))
    uc, vc, wc = P.canonical_perm(u), P.canonical_perm(v), P.canonical_perm(w)
    extra = set(P.right_descents(wc)) - (set(P.right_descents(uc)) | set(P.right_descents(vc)))
    d["n_extra_descents"] = float(len(extra))
    return [d[k] for k in BATTERY_NAMES]


def featurize_many(triples, workers=2, tag=""):
    t0 = time.time()
    if workers <= 1:
        rows = [featurize_triple(t) for t in triples]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(workers) as pool:
            rows = list(pool.imap(featurize_triple, triples, chunksize=2048))
    dt = time.time() - t0
    print(f"[featurize{(' ' + tag) if tag else ''}] {len(triples)} triples in {dt:.1f}s "
          f"({1e3 * dt / max(len(triples), 1):.2f} ms/triple)", flush=True)
    return np.asarray(rows, dtype=np.float64), dt


def save_frag(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name + ".json"), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[frag] wrote {name}.json", flush=True)


# ---------------------------------------------------------------------------
# stage: featurize-v003  (bit-exact reconstruction of the discovery inputs)
# ---------------------------------------------------------------------------
def stage_featurize_v003(args):
    os.makedirs(CACHE, exist_ok=True)
    rng = random.Random(0)  # identical to residual_mine_v2.py

    zlines = open(V003_ZEROS).readlines()
    n_all = len(zlines)
    # same state consumption as rng.sample(zlines, 45000) in the discovery run
    disc_idx = rng.sample(range(n_all), 45000) if n_all > 45000 else list(range(n_all))
    zeros = [json.loads(l) for l in zlines]
    mech = np.array([z["mech"] for z in zeros])
    zsrc = np.array([z["source_n"] for z in zeros])
    ztriples = [(tuple(z["u"]), tuple(z["v"]), tuple(z["w"])) for z in zeros]

    # ---- discovery-run positives, replicated with identical rng sequence ----
    pos_files = [("data/raw/s6_shard2.jsonl", 6), ("/mnt/data/raw/s7b_shard1.jsonl", 7),
                 ("/mnt/data/raw/s8.jsonl", 8), ("/mnt/data/raw/s9.jsonl", 9)]
    pos_triples, psrc = [], []
    used_lines = {}  # file -> set(line idx) consumed by the discovery replica
    for fn, src_n in pos_files:
        path = fn if fn.startswith("/") else os.path.join(REPO, fn)
        lines = open(path).readlines()
        order = list(range(len(lines)))
        rng.shuffle(order)  # same permutation as rng.shuffle(lines)
        cnt, used = 0, set()
        for li in order:
            if cnt >= 18000:
                break
            r = json.loads(lines[li])
            u, v = tuple(r["u"]), tuple(r["v"])
            sup = r["support"]
            took = False
            for (w, c) in rng.sample(sup, min(4, len(sup))):
                if cnt >= 18000:
                    break
                pos_triples.append((u, v, tuple(w)))
                psrc.append(src_n); cnt += 1; took = True
            if took:
                used.add(li)
        used_lines[fn] = used
        print(f"[v003] +{cnt} discovery positives from S{src_n}", flush=True)

    # ---- held-out positives: disjoint files (+ unused s9 lines), seed 123 ----
    ho_rng = random.Random(123)
    ho_files = [("data/raw/s6_shard1.jsonl", 6, None), ("/mnt/data/raw/s7b_shard2.jsonl", 7, None),
                ("/mnt/data/raw/s8b.jsonl", 8, None),
                ("/mnt/data/raw/s9.jsonl", 9, used_lines["/mnt/data/raw/s9.jsonl"])]
    ho_triples, hsrc = [], []
    for fn, src_n, excl in ho_files:
        path = fn if fn.startswith("/") else os.path.join(REPO, fn)
        lines = open(path).readlines()
        order = [i for i in range(len(lines)) if not excl or i not in excl]
        ho_rng.shuffle(order)
        cnt = 0
        for li in order:
            if cnt >= 30000:
                break
            r = json.loads(lines[li])
            u, v = tuple(r["u"]), tuple(r["v"])
            for (w, c) in ho_rng.sample(r["support"], min(4, len(r["support"]))):
                if cnt >= 30000:
                    break
                ho_triples.append((u, v, tuple(w)))
                hsrc.append(src_n); cnt += 1
        print(f"[v003] +{cnt} held-out positives from S{src_n} ({os.path.basename(fn)})",
              flush=True)

    Xz, tz = featurize_many(ztriples, args.workers, "v003-zeros")
    Xp, tp = featurize_many(pos_triples, args.workers, "v003-pos")
    Xh, th = featurize_many(ho_triples, args.workers, "v003-heldout")
    np.savez_compressed(
        os.path.join(CACHE, "v003_features.npz"),
        Xz=Xz, mech=mech, zsrc=zsrc, disc_idx=np.array(disc_idx),
        Xp=Xp, psrc=np.array(psrc), Xh=Xh, hsrc=np.array(hsrc),
        names=np.array(BATTERY_NAMES))
    save_frag("featurize_v003", {
        "n_zeros_full": n_all, "n_discovery_zero_sample": len(disc_idx),
        "n_discovery_pos": len(pos_triples), "n_heldout_pos": len(ho_triples),
        "featurize_seconds": {"zeros_full": tz, "discovery_pos": tp, "heldout_pos": th}})


# ---------------------------------------------------------------------------
# stage: featurize-v002  (C1-era and v001-style pools from the real v002 data)
# ---------------------------------------------------------------------------
def stage_featurize_v002(args):
    import pandas as pd
    from schubml.data.datasets import load_triple_arrays

    os.makedirs(CACHE, exist_ok=True)
    pdf = pd.read_parquet(os.path.join(V002_DIR, "permutations.parquet"),
                          columns=["perm_id", "canonical"])
    perms = {int(p): tuple(json.loads(c)) for p, c in
             zip(pdf["perm_id"].to_numpy(), pdf["canonical"])}
    del pdf
    arr = load_triple_arrays(os.path.join(V002_DIR, "triples.parquet"),
                             split="product_disjoint:test", tiers=["Z2", "Z3", "Z5"])
    m = len(arr["w_id"])
    idx = np.random.default_rng(0).choice(m, min(120000, m), replace=False)

    grading_ok, both, tiers, triples = [], [], [], []
    for i in idx:
        u = tuple(perms[int(arr["u_id"][i])]); v = tuple(perms[int(arr["v_id"][i])])
        w = tuple(perms[int(arr["w_id"][i])])
        if P.length(w) != P.length(u) + P.length(v):
            continue
        if not (bruhat.bruhat_leq(u, w) and bruhat.bruhat_leq(v, w)):
            continue
        du = set(P.right_descents(u)) | set(P.right_descents(v))
        desc_ok = set(P.right_descents(w)) <= du
        triples.append((u, v, w)); both.append(desc_ok)
        tiers.append(int(arr["tier_id"][i]))
    both = np.array(both); tiers = np.array(tiers)
    print(f"[v002] zeros passing grading+Bruhat: {len(triples)} "
          f"(also descent-union: {int(both.sum())})", flush=True)

    # era positives: residual_necessary_scan.py protocol (s5 all + 12k s6 lines)
    rng = random.Random(0)
    pos_triples, psrc = [], []
    for fn, cap, src_n in [("data/raw/s5.jsonl", None, 5),
                           ("data/raw/s6_shard0.jsonl", 12000, 6)]:
        lines = open(os.path.join(REPO, fn)).readlines()
        if cap:
            lines = rng.sample(lines, min(cap, len(lines)))
        for line in lines:
            r = json.loads(line)
            u, v = tuple(r["u"]), tuple(r["v"])
            for w, c in r["support"][:20]:
                pos_triples.append((u, v, tuple(w)))
                psrc.append(src_n)
        print(f"[v002] positives so far: {len(pos_triples)} (through S{src_n})", flush=True)

    Xz, tz = featurize_many(triples, args.workers, "v002-zeros")
    Xp, tp = featurize_many(pos_triples, args.workers, "v002-pos")
    np.savez_compressed(
        os.path.join(CACHE, "v002_features.npz"),
        Xz=Xz, desc_ok=both, tiers=tiers, Xp=Xp, psrc=np.array(psrc),
        names=np.array(BATTERY_NAMES))
    save_frag("featurize_v002", {
        "n_zeros_grading_bruhat": len(triples),
        "n_zeros_also_descent_union": int(both.sum()),
        "n_pos": len(pos_triples),
        "featurize_seconds": {"zeros": tz, "pos": tp}})


# ---------------------------------------------------------------------------
# mining primitives
# ---------------------------------------------------------------------------
def zero_violation_candidates(Xp, Xz, names, min_kills=30, groups=None):
    """The exact residual_mine_v2 protocol: per feature/direction, the region
    strictly outside the positives' observed range; candidate if it kills >=
    min_kills zeros.  groups: optional label array over zeros for the
    mechanism-independence flag."""
    cands = []
    for j, nm in enumerate(names):
        pmin, pmax = Xp[:, j].min(), Xp[:, j].max()
        for mask, direc, bound in (((Xz[:, j] < pmin), "<", pmin),
                                   ((Xz[:, j] > pmax), ">", pmax)):
            k = int(mask.sum())
            if k < min_kills:
                continue
            c = {"feature": nm, "direction": direc, "bound": float(bound),
                 "zeros_killed": k, "kill_rate": float(mask.mean())}
            if groups is not None:
                rates = {str(g): float(mask[groups == g].mean())
                         for g in np.unique(groups)}
                lo, hi = min(rates.values()), max(rates.values())
                c["group_kill_rates"] = rates
                c["mechanism_independent"] = bool(lo > 0 and lo / max(hi, 1e-9) > 0.33)
            cands.append(c)
    cands.sort(key=lambda c: -c["zeros_killed"])
    return cands


def rank_in(cands, feature, direction):
    """Competition rank by kill count (1-based); None if absent."""
    kills = None
    for c in cands:
        if c["feature"] == feature and c["direction"] == direction:
            kills = c["zeros_killed"]
            break
    if kills is None:
        return None, None
    r = 1 + sum(1 for c in cands if c["zeros_killed"] > kills)
    return r, kills


def simplicity_key(c):
    b = c["bound"]
    return (0 if float(b).is_integer() else 1, abs(b))


# ---------------------------------------------------------------------------
# stage: mine
# ---------------------------------------------------------------------------
def stage_mine(args):
    out = {}
    names = BATTERY_NAMES

    # ---------------- v003 era (C2) ----------------
    d3 = np.load(os.path.join(CACHE, "v003_features.npz"), allow_pickle=True)
    Xz_full, Xp, mech = d3["Xz"], d3["Xp"], d3["mech"]
    disc = d3["disc_idx"]
    Xz = Xz_full[disc]; mech_d = mech[disc]
    t0 = time.time()
    cands = zero_violation_candidates(Xp, Xz, names, min_kills=30, groups=mech_d)
    mine_dt = time.time() - t0
    r_kill, kills = rank_in(cands, "ess_n_both_tight", ">")
    by_simp = sorted([c for c in cands], key=simplicity_key)
    r_simp = 1 + next(i for i, c in enumerate(by_simp)
                      if c["feature"] == "ess_n_both_tight" and c["direction"] == ">") \
        if any(c["feature"] == "ess_n_both_tight" and c["direction"] == ">" for c in cands) else None
    out["v003_C2"] = {
        "n_zeros": int(len(Xz)), "n_pos": int(len(Xp)),
        "n_candidates_min30": len(cands),
        "candidates": cands,
        "C2_rank_by_kills": r_kill, "C2_kills": kills,
        "C2_rank_by_simplicity": r_simp,
        "mine_seconds": mine_dt,
    }
    print(f"[mine v003] {len(cands)} candidates (>=30 kills); "
          f"C2 rank by kills = {r_kill} ({kills} kills), by simplicity = {r_simp}",
          flush=True)

    # ML ceilings on the same features/sample (expert items 2+3)
    out["v003_ml_ceilings"] = ml_ceilings(Xp, Xz, names)

    # ---------------- v002 era (C1) ----------------
    d2 = np.load(os.path.join(CACHE, "v002_features.npz"), allow_pickle=True)
    Xz2_all, desc_ok, tiers, Xp2 = d2["Xz"], d2["desc_ok"], d2["tiers"], d2["Xp"]
    Xz2 = Xz2_all[desc_ok]  # v002-era residual: grading+Bruhat+descent-union
    t0 = time.time()
    cands2 = zero_violation_candidates(Xp2, Xz2, names, min_kills=30,
                                       groups=tiers[desc_ok])
    dt2 = time.time() - t0
    r1, k1 = rank_in(cands2, "ess_n_slack0", ">")
    r2, k2 = rank_in(cands2, "ess_n_both_tight", ">")
    out["v002_C1"] = {
        "n_zeros": int(len(Xz2)), "n_pos": int(len(Xp2)),
        "n_candidates_min30": len(cands2), "candidates": cands2,
        "C1_rank_by_kills": r1, "C1_kills": k1,
        "C2_rank_by_kills": r2, "C2_kills": k2,
        "mine_seconds": dt2,
    }
    print(f"[mine v002] {len(cands2)} candidates; C1 rank={r1} ({k1} kills); "
          f"C2 also visible at rank={r2} ({k2})", flush=True)

    # ---------------- v001 style (descent-union) ----------------
    Xz1 = Xz2_all  # grading+Bruhat only
    t0 = time.time()
    cands1 = zero_violation_candidates(Xp2, Xz1, names, min_kills=30,
                                       groups=tiers)
    dt1 = time.time() - t0
    rd, kd = rank_in(cands1, "n_extra_descents", ">")
    out["v001_descent_union"] = {
        "n_zeros": int(len(Xz1)), "n_pos": int(len(Xp2)),
        "n_candidates_min30": len(cands1), "candidates": cands1,
        "descent_union_rank_by_kills": rd, "descent_union_kills": kd,
        "mine_seconds": dt1,
    }
    print(f"[mine v001-style] {len(cands1)} candidates; descent-union rank={rd} "
          f"({kd} kills)", flush=True)

    save_frag("mine", out)


def ml_ceilings(Xp, Xz, names):
    """Expert items 2 (boosted trees) and 3 (sparse logistic) + the depth-5
    tree of the discovery run, on the identical features and sample."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xz))])
    X = np.concatenate([Xp, Xz])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    out = {}
    t0 = time.time()
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=60,
                                  class_weight="balanced", random_state=0).fit(Xtr, ytr)
    p = tree.predict(Xte)
    out["tree_d5"] = {"balanced_acc": float(balanced_accuracy_score(yte, p)),
                      "mcc": float(matthews_corrcoef(yte, p)),
                      "seconds": time.time() - t0}
    t0 = time.time()
    gb = HistGradientBoostingClassifier(max_iter=400, random_state=0,
                                        class_weight="balanced").fit(Xtr, ytr)
    p = gb.predict(Xte)
    out["boosted_trees"] = {"balanced_acc": float(balanced_accuracy_score(yte, p)),
                            "mcc": float(matthews_corrcoef(yte, p)),
                            "seconds": time.time() - t0}
    t0 = time.time()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(penalty="l1", solver="saga", C=0.2, max_iter=1000,
                            class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    p = lr.predict((Xte - mu) / sd)
    out["sparse_logistic"] = {"balanced_acc": float(balanced_accuracy_score(yte, p)),
                              "mcc": float(matthews_corrcoef(yte, p)),
                              "n_nonzero_coefs": int((np.abs(lr.coef_[0]) > 1e-6).sum()),
                              "seconds": time.time() - t0}
    for k, v in out.items():
        print(f"[ml] {k}: bal-acc={v['balanced_acc']:.3f} mcc={v['mcc']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# stage: sample-complexity
# ---------------------------------------------------------------------------
def stage_sample_complexity(args):
    names = BATTERY_NAMES
    jix = {n: i for i, n in enumerate(names)}
    d3 = np.load(os.path.join(CACHE, "v003_features.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(CACHE, "v002_features.npz"), allow_pickle=True)

    pools = {
        "C2_both_tight": (d3["Xp"], d3["Xz"]),                       # full 328k zeros
        "C1_ess_slack0": (d2["Xp"], d2["Xz"][d2["desc_ok"]]),
        "descent_union": (d2["Xp"], d2["Xz"]),
    }
    sizes = [500, 2000, 8000, 32000, None]
    out = {}
    for law, (feat, direc) in TARGETS.items():
        Xp, Xz = pools[law]
        j = jix[feat]
        pmax = Xp[:, j].max()
        assert direc == ">"
        hits = Xz[:, j] > pmax  # zeros in the law's region (0 positive viol by range)
        base = {"n_pool_zeros": int(len(Xz)), "pos_bound": float(pmax),
                "pool_kill_rate": float(hits.mean()), "rows": []}
        for n in sizes:
            n_eff = len(Xz) if n is None else min(n, len(Xz))
            det, ranks, killed = [], [], []
            for rep in range(5):
                rng = np.random.default_rng(1000 + rep)
                sub = rng.choice(len(Xz), n_eff, replace=False)
                k = int(hits[sub].sum())
                killed.append(k)
                detected = k >= 10
                det.append(detected)
                if detected:
                    cands = zero_violation_candidates(Xp, Xz[sub], names, min_kills=10)
                    r, _ = rank_in(cands, feat, direc)
                    ranks.append(r)
            base["rows"].append({
                "n_zeros": n_eff, "detection_freq": float(np.mean(det)),
                "kills_per_rep": killed,
                "rank_when_detected": ranks})
            print(f"[sc {law}] n={n_eff}: det={np.mean(det):.1f} kills={killed} "
                  f"ranks={ranks}", flush=True)
        out[law] = base
    save_frag("sample_complexity", out)


# ---------------------------------------------------------------------------
# stage: depth2  (exhaustive low-complexity predicate enumeration)
# ---------------------------------------------------------------------------
def build_literals(X_list, names, thresholds=(0, 1, 2, 3, 4)):
    """Integer-threshold literals  (f >= t) and (f <= t)  that are non-trivial
    on the union of the given matrices.  Returns (literal list, per-matrix
    boolean matrices)."""
    Xall = np.concatenate(X_list)
    lits = []
    for j, nm in enumerate(names):
        lo, hi = Xall[:, j].min(), Xall[:, j].max()
        for t in thresholds:
            if lo < t <= hi:  # f >= t nontrivial (not everything / not nothing)
                lits.append((j, ">=", t))
            if lo <= t < hi:  # f <= t nontrivial
                lits.append((j, "<=", t))
    mats = []
    for X in X_list:
        M = np.zeros((X.shape[0], len(lits)), dtype=np.uint8)
        for k, (j, op, t) in enumerate(lits):
            M[:, k] = (X[:, j] >= t) if op == ">=" else (X[:, j] <= t)
        mats.append(M)
    return lits, mats


def depth2_survey(Xp, Xz, Xh, names, min_kills=10, targets=()):
    """Enumerate all depth-1 and depth-2 conjunctions; return burden metrics."""
    lits, (Lp, Lz) = build_literals([Xp, Xz], names)
    Lh = np.zeros((Xh.shape[0], len(lits)), dtype=np.uint8)
    for k, (j, op, t) in enumerate(lits):
        Lh[:, k] = (Xh[:, j] >= t) if op == ">=" else (Xh[:, j] <= t)
    L = len(lits)
    n_pairs = L * (L - 1) // 2
    t0 = time.time()
    Pp = (Lp.astype(np.float32).T @ Lp.astype(np.float32)).astype(np.int64)
    Zz = (Lz.astype(np.float32).T @ Lz.astype(np.float32)).astype(np.int64)
    Hh = (Lh.astype(np.float32).T @ Lh.astype(np.float32)).astype(np.int64)
    dt_mm = time.time() - t0

    def lit_str(k):
        j, op, t = lits[k]
        return f"{names[j]} {op} {t}"

    survivors = []  # (kills, depth, (k1,k2) or (k1,), heldout_viol)
    for k in range(L):
        if Pp[k, k] == 0 and Zz[k, k] >= min_kills:
            survivors.append({"pred": [lit_str(k)], "lits": (k,), "depth": 1,
                              "kills": int(Zz[k, k]), "heldout_viol": int(Hh[k, k])})
    iu = np.triu_indices(L, 1)
    ok = (Pp[iu] == 0) & (Zz[iu] >= min_kills)
    for a, b in zip(iu[0][ok], iu[1][ok]):
        # skip pairs where one literal alone already has zero positive support
        # AND the same kill count (pure padding of a depth-1 survivor)
        survivors.append({"pred": [lit_str(a), lit_str(b)], "lits": (int(a), int(b)),
                          "depth": 2, "kills": int(Zz[a, b]),
                          "heldout_viol": int(Hh[a, b])})
    survivors.sort(key=lambda s: -s["kills"])

    # kill-signature equivalence classes + subset analysis vs targets
    sigs = {}
    for s in survivors:
        ks = s["lits"]
        col = Lz[:, ks[0]].astype(bool)
        for k in ks[1:]:
            col &= Lz[:, k].astype(bool)
        h = col.tobytes()
        sigs.setdefault(h, []).append(s)
        s["_col"] = col
    n_classes = len(sigs)

    res = {"n_literals": L, "n_pairs_enumerated": n_pairs,
           "n_predicates_enumerated": L + n_pairs,
           "matmul_seconds": dt_mm,
           "n_survivors": len(survivors),
           "n_distinct_kill_signatures": n_classes,
           "n_clean_survivors": sum(1 for s in survivors if s["heldout_viol"] == 0),
           "junk_rate_heldout": (float(np.mean([s["heldout_viol"] > 0 for s in survivors]))
                                 if survivors else None),
           "targets": {}}

    for tname, feat, direc in targets:
        # the law as a literal: feat >= 1 (direction '>' with bound 0)
        tk = next((k for k, (j, op, t) in enumerate(lits)
                   if names[j] == feat and op == ">=" and t == 1), None)
        entry = {"present_as_literal": tk is not None}
        if tk is not None:
            tcol = Lz[:, tk].astype(bool)
            tkills = int(tcol.sum())
            in_surv = Pp[tk, tk] == 0 and tkills >= min_kills
            entry.update({"kills": tkills, "is_survivor": bool(in_surv)})
            if in_surv:
                entry["rank_by_kills"] = 1 + sum(1 for s in survivors
                                                 if s["kills"] > tkills)
                # simplicity ordering: depth asc, then sum|t| asc, then kills desc
                simp = sorted(survivors,
                              key=lambda s: (s["depth"],
                                             sum(abs(lits[k][2]) for k in s["lits"]),
                                             -s["kills"]))
                entry["rank_by_simplicity"] = 1 + next(
                    i for i, s in enumerate(simp)
                    if s["lits"] == (tk,) or set(s["lits"]) == {tk})
                clean = [s for s in survivors if s["heldout_viol"] == 0]
                entry["rank_by_kills_among_clean"] = 1 + sum(
                    1 for s in clean if s["kills"] > tkills)
                # how many survivors are NOT killed-subsets of the law's region
                nsub = sum(1 for s in survivors if not (s["_col"] & ~tcol).any())
                entry["n_survivors_subset_of_law"] = nsub
        res["targets"][tname] = entry
    for s in survivors:
        s.pop("_col", None); s.pop("lits", None)
    res["top_survivors"] = survivors[:40]
    return res


def stage_depth2(args):
    d3 = np.load(os.path.join(CACHE, "v003_features.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(CACHE, "v002_features.npz"), allow_pickle=True)
    names = BATTERY_NAMES
    out = {}

    Xz3 = d3["Xz"][d3["disc_idx"]]
    t0 = time.time()
    out["v003"] = depth2_survey(d3["Xp"], Xz3, d3["Xh"], names,
                                targets=[("C2_both_tight", "ess_n_both_tight", ">"),
                                         ("C1_ess_slack0", "ess_n_slack0", ">")])
    out["v003"]["total_seconds"] = time.time() - t0
    print(f"[depth2 v003] survivors={out['v003']['n_survivors']} "
          f"classes={out['v003']['n_distinct_kill_signatures']} "
          f"junk={out['v003']['junk_rate_heldout']:.2%} "
          f"C2={out['v003']['targets']['C2_both_tight']}", flush=True)

    Xz2 = d2["Xz"][d2["desc_ok"]]
    t0 = time.time()
    out["v002"] = depth2_survey(d2["Xp"], Xz2, d3["Xh"], names,
                                targets=[("C1_ess_slack0", "ess_n_slack0", ">"),
                                         ("C2_both_tight", "ess_n_both_tight", ">")])
    out["v002"]["total_seconds"] = time.time() - t0
    print(f"[depth2 v002] survivors={out['v002']['n_survivors']} "
          f"junk={out['v002']['junk_rate_heldout']:.2%} "
          f"C1={out['v002']['targets']['C1_ess_slack0']}", flush=True)

    t0 = time.time()
    out["v001"] = depth2_survey(d2["Xp"], d2["Xz"], d3["Xh"], names,
                                targets=[("descent_union", "n_extra_descents", ">")])
    out["v001"]["total_seconds"] = time.time() - t0
    print(f"[depth2 v001] survivors={out['v001']['n_survivors']} "
          f"du={out['v001']['targets']['descent_union']}", flush=True)
    save_frag("depth2", out)


# ---------------------------------------------------------------------------
# stage: random-control
# ---------------------------------------------------------------------------
def stage_random_control(args):
    import pandas as pd
    from scipy.stats import rankdata, spearmanr
    from sklearn.metrics import roc_auc_score

    cache = np.load(os.path.join(FORENSICS_CACHE, "matched_cache.npz"))
    acts = cache["acts"].astype(np.float64)
    pos = cache["positive"]; pid = cache["product_id"]
    Z = np.load(os.path.join(NAMING_CACHE, "sae_codes.npy")).astype(np.float64)
    sh_path = os.path.join(NAMING_CACHE, "shadow_features_repo.parquet")
    shadow = pd.read_parquet(sh_path) if os.path.exists(sh_path) else None
    y0 = (~pos).astype(int)

    def centered(x):
        s = pd.Series(x)
        return (s - s.groupby(pd.Series(pid)).transform("mean")).to_numpy()

    def folded_aucs(M):
        # AUC via rank statistics, vectorized over columns
        n1 = y0.sum(); n0 = len(y0) - n1
        aucs = np.empty(M.shape[1])
        for j in range(M.shape[1]):
            r = rankdata(M[:, j])
            aucs[j] = (r[y0 == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return np.maximum(aucs, 1 - aucs)

    out = {}
    t0 = time.time()
    sae_auc = folded_aucs(Z)
    out["sae"] = {"n_features": int(Z.shape[1]),
                  "n_dead": int((Z.std(0) <= 1e-9).sum()),
                  "max_zero_auc": float(sae_auc.max()),
                  "top5": [[int(j), float(sae_auc[j])] for j in np.argsort(-sae_auc)[:5]],
                  "n_auc_gt_0.9": int((sae_auc > 0.9).sum()),
                  "n_auc_gt_0.95": int((sae_auc > 0.95).sum()),
                  "n_auc_gt_0.98": int((sae_auc > 0.98).sum())}
    print(f"[rc] SAE: max AUC={sae_auc.max():.4f} top={np.argsort(-sae_auc)[:3]}",
          flush=True)

    rng = np.random.default_rng(0)
    R = rng.standard_normal((acts.shape[1], Z.shape[1]))
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    lin = acts @ R
    relu = np.maximum((acts - acts.mean(0)) @ R, 0.0)
    for nm, M in (("random_linear", lin), ("random_relu", relu)):
        a = folded_aucs(M)
        out[nm] = {"n_features": int(M.shape[1]), "max_zero_auc": float(a.max()),
                   "n_auc_gt_0.9": int((a > 0.9).sum()),
                   "n_auc_gt_0.95": int((a > 0.95).sum()),
                   "n_auc_gt_0.98": int((a > 0.98).sum()),
                   "median_auc": float(np.median(a))}
        print(f"[rc] {nm}: max AUC={a.max():.4f} "
              f">0.9:{out[nm]['n_auc_gt_0.9']} >0.95:{out[nm]['n_auc_gt_0.95']}",
              flush=True)

    # raw activation coordinates for reference
    a = folded_aucs(acts)
    out["raw_coords"] = {"n_features": int(acts.shape[1]),
                         "max_zero_auc": float(a.max()),
                         "n_auc_gt_0.95": int((a > 0.95).sum())}

    if shadow is not None:
        mp = shadow["shadow_mp_infeasible"].to_numpy()
        mdb = shadow["shadow_min_delta_both"].to_numpy()
        cmp_, cmdb = centered(mp), centered(mdb)

        def best_rho(M, cols):
            best = 0.0
            for j in cols:
                r = spearmanr(centered(M[:, j]), cmp_).statistic
                if abs(r) > abs(best):
                    best = r
            return float(best)

        top_sae = np.argsort(-sae_auc)[:8]
        out["shadow_naming"] = {
            "sae_653_rho_mp": float(spearmanr(centered(Z[:, 653]), cmp_).statistic),
            "sae_653_rho_min_delta": float(spearmanr(centered(Z[:, 653]), cmdb).statistic),
            "best_rho_mp_top8_sae": best_rho(Z, top_sae),
            "best_rho_mp_top8_random_linear": best_rho(lin, np.argsort(-folded_aucs(lin))[:8]),
            "best_rho_mp_top8_random_relu": best_rho(relu, np.argsort(-folded_aucs(relu))[:8]),
        }
        print(f"[rc] shadow naming: {out['shadow_naming']}", flush=True)
    out["seconds"] = time.time() - t0
    save_frag("random_control", out)


# ---------------------------------------------------------------------------
# stage: report  (assemble the consolidated JSON)
# ---------------------------------------------------------------------------
def stage_report(args):
    frags = {}
    for name in ("featurize_v003", "featurize_v002", "mine", "mine_heldout",
                 "sample_complexity", "depth2", "random_control"):
        p = os.path.join(CACHE, name + ".json")
        if os.path.exists(p):
            frags[name] = json.load(open(p))
        else:
            frags[name] = {"skipped": True}
    # trim: drop full candidate lists to top 25 in the consolidated file
    for era in ("v003_C2", "v002_C1", "v001_descent_union"):
        if era in frags.get("mine", {}):
            frags["mine"][era]["candidates"] = frags["mine"][era]["candidates"][:25]
    out_path = os.path.join(REPO, "reports/conjectures/ablation_baseline.json")
    with open(out_path, "w") as f:
        json.dump(frags, f, indent=2)
    print(f"[report] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["featurize-v003", "featurize-v002", "mine",
                             "sample-complexity", "depth2", "random-control",
                             "report"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    {"featurize-v003": stage_featurize_v003,
     "featurize-v002": stage_featurize_v002,
     "mine": stage_mine,
     "sample-complexity": stage_sample_complexity,
     "depth2": stage_depth2,
     "random-control": stage_random_control,
     "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
