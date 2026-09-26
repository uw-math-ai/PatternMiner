"""Shared helpers for the ``scripts/`` interpretability CLIs.

Loads a trained run + processed dataset when available, and otherwise falls back
to a *fresh* (random-weight) model on a tiny **synthetic** dataset so every CLI
has a runnable minimal path even before training / data-building completes
(PROJECT.md says ``data/processed/v001`` may still be building).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..combinatorics import permutations as P
from ..data.datasets import (SchubertTripleDataset, build_perms_by_id,
                             load_triple_arrays)
from ..data.tokenizers import Tokenizer, TokenizerConfig
from ..models.product_query_transformer import ModelConfig, ProductQueryTransformer

# a small but non-trivial set of permutations for the synthetic fallback
_SYNTH_PERMS: List[Tuple[int, ...]] = [
    (1, 3, 2), (2, 1), (2, 3, 1), (3, 1, 2), (1, 2), (3, 2, 1),
    (2, 1, 3), (1, 4, 3, 2), (2, 4, 1, 3), (4, 1, 3, 2), (3, 1, 4, 2),
    (1, 3, 4, 2), (2, 3, 4, 1), (4, 3, 2, 1),
]


def synthetic_dataset(
    n_slots: int = 8,
    n_ctx: int = 14,
    n_triples: int = 64,
    seed: int = 0,
) -> Tuple[SchubertTripleDataset, Tokenizer]:
    """Build a tiny in-memory dataset of synthetic triples for smoke paths."""
    rng = np.random.default_rng(seed)
    perms = _SYNTH_PERMS
    perms_by_id = {i: p for i, p in enumerate(perms)}
    m = len(perms)
    u_id = rng.integers(0, m, n_triples)
    v_id = rng.integers(0, m, n_triples)
    w_id = rng.integers(0, m, n_triples)
    # cheap pseudo-labels: positive iff length defect is zero
    coeff = np.zeros(n_triples, dtype=np.int64)
    pos = np.zeros(n_triples, dtype=np.int64)
    for k in range(n_triples):
        lu, lv, lw = (P.length(perms[u_id[k]]), P.length(perms[v_id[k]]),
                      P.length(perms[w_id[k]]))
        if lw == lu + lv and lw > 0:
            coeff[k] = int(rng.integers(1, 4))
            pos[k] = 1
    arrays = {
        "u_id": u_id, "v_id": v_id, "w_id": w_id,
        "coefficient": coeff, "positive": pos,
        "tier_id": rng.integers(0, 4, n_triples).astype(np.int64),
        "product_id": (u_id * m + v_id).astype(np.int64),
    }
    tok = Tokenizer(TokenizerConfig(n_ctx=n_ctx))
    ds = SchubertTripleDataset(arrays, perms_by_id, tok, n_count_queries=n_slots,
                               with_aux=False)
    return ds, tok


def fresh_model(n_slots: int = 8, n_ctx: int = 14) -> ProductQueryTransformer:
    """A small random-weight model for smoke runs."""
    cfg = ModelConfig(d_model=32, n_encoder_layers=2, n_cross_layers=2, n_heads=2,
                      d_mlp=64, n_ctx=n_ctx, n_slots=n_slots)
    return ProductQueryTransformer(cfg).eval()


def load_model_and_dataset(
    run: Optional[str],
    data_dir: Optional[str],
    device: str = "cpu",
    *,
    split: str = "product_disjoint:test",
    limit: int = 4096,
):
    """Return ``(model, mcfg, dataset)``.

    Uses the trained run + processed parquet when both are present; otherwise a
    fresh model and/or synthetic dataset, so the CLI always has a runnable path.
    """
    model = mcfg = None
    if run and os.path.isdir(run) and os.path.exists(os.path.join(run, "run_config.json")):
        try:
            from ..train.inference import load_run
            model, mcfg, _ = load_run(run, device=device)
        except Exception:
            model = mcfg = None
    if model is None:
        model = fresh_model()
        mcfg = model.cfg
        model.to(device)

    n_slots = mcfg.n_slots
    dataset = None
    trip = os.path.join(data_dir or "", "triples.parquet")
    perms = os.path.join(data_dir or "", "permutations.parquet")
    if data_dir and os.path.exists(trip) and os.path.exists(perms):
        try:
            from ..data.schemas import PERM_JSON_COLUMNS, load_table
            arr = load_triple_arrays(trip, split=split)
            if len(arr["w_id"]) > limit:
                idx = np.random.default_rng(0).choice(len(arr["w_id"]), limit, replace=False)
                arr = {k: v[idx] for k, v in arr.items()}
            perm_df = load_table(perms, PERM_JSON_COLUMNS)
            perms_by_id = build_perms_by_id(perm_df)
            tok = Tokenizer(TokenizerConfig(n_ctx=mcfg.n_ctx))
            dataset = SchubertTripleDataset(arr, perms_by_id, tok,
                                            n_count_queries=n_slots, with_aux=False)
        except Exception:
            dataset = None
    if dataset is None:
        dataset, _ = synthetic_dataset(n_slots=n_slots, n_ctx=mcfg.n_ctx)
    return model, mcfg, dataset


def matched_clean_corrupt(dataset, device: str = "cpu", n: int = 8):
    """Build a (clean=positive, corrupt=negative) batch pair sharing factors.

    Greedily pairs a positive triple with a negative triple having the *same*
    ``(u, v)`` product when possible (true matched pack); otherwise pairs by
    product id, and finally falls back to arbitrary pos/neg rows.  Returns
    ``(clean_batch, corrupt_batch)`` on ``device``.
    """
    from ..data.collators import collate, to_device

    a = dataset.a
    pos_idx = [i for i in range(len(dataset)) if int(a["positive"][i]) == 1]
    neg_idx = [i for i in range(len(dataset)) if int(a["positive"][i]) == 0]
    if not pos_idx or not neg_idx:
        # degenerate: just take the first rows for both sides
        pos_idx = neg_idx = list(range(min(n, len(dataset))))

    neg_by_prod: Dict[int, List[int]] = {}
    for j in neg_idx:
        neg_by_prod.setdefault(int(a["product_id"][j]), []).append(j)

    clean_rows, corrupt_rows = [], []
    used_neg = set()
    for i in pos_idx:
        pid = int(a["product_id"][i])
        cand = [j for j in neg_by_prod.get(pid, []) if j not in used_neg]
        if not cand:
            cand = [j for j in neg_idx if j not in used_neg]
        if not cand:
            break
        j = cand[0]
        used_neg.add(j)
        clean_rows.append(i)
        corrupt_rows.append(j)
        if len(clean_rows) >= n:
            break
    if not clean_rows:  # ultimate fallback
        clean_rows = pos_idx[:n]
        corrupt_rows = neg_idx[:n]
    m = min(len(clean_rows), len(corrupt_rows))
    clean = to_device(collate([dataset[i] for i in clean_rows[:m]]), device)
    corrupt = to_device(collate([dataset[i] for i in corrupt_rows[:m]]), device)
    return clean, corrupt


def triples_from_dataset(dataset, n: int = 512):
    """Extract ``(triples, positive, tier_id)`` for symbolic rule learning."""
    a = dataset.a
    pb = dataset.perms_by_id
    n = min(n, len(dataset))
    triples, pos, tier = [], [], []
    for i in range(n):
        triples.append((pb[int(a["u_id"][i])], pb[int(a["v_id"][i])], pb[int(a["w_id"][i])]))
        pos.append(int(a["positive"][i]))
        tier.append(int(a["tier_id"][i]))
    return triples, np.array(pos), np.array(tier)


def default_report_dir(run: Optional[str], fallback: str = "reports/interp_cards") -> str:
    """Prefer ``<run>/interp`` when a run dir exists, else the shared reports dir."""
    if run and os.path.isdir(run):
        return os.path.join(run, "interp")
    return fallback
