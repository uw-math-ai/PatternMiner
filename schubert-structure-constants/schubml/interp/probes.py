"""Linear / logistic probing of model activations (PROJECT.md 13.2).

For every chosen hook *site* we extract a single pooled vector per example and
fit cheap sklearn probes (Ridge for real-valued / multilabel targets, logistic
regression for boolean targets) predicting **known** combinatorial features of
``(u, v, w)``.  A successful model should expose

* early layers  -> raw permutation / Rothe information,
* middle layers -> ``u, v, w`` compatibility features,
* late layers   -> positivity / support / counting features.

This is Level-1 evidence (a linear probe *detects* a feature); it is hypothesis
generation, not a causal claim.

Public API
----------
``collect_activations(model, batch, names)``
    -> ``{name: pooled [B, d] tensor}`` for the requested hook sites.
``probe_targets(perms_by_id, u_ids, v_ids, w_ids)``
    -> ``{feature_name: (y[N], kind)}`` of known-feature regression / class targets.
``run_probes(model, dataset, sites, device, ...)``
    -> nested ``{site: {feature: {metric: value}}}`` score dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..combinatorics import known_families as KF
from ..combinatorics import permutations as P
from ..combinatorics import rothe
from ..data.collators import collate, to_device
from ..data.labels import FAMILY_TAGS, family_multihot
from .hooks import HookedModule

# ---------------------------------------------------------------------------
# Site pooling: turn a (possibly token-stream) activation into one [B, d] vector
# ---------------------------------------------------------------------------
# The readout token layout (first ``Tg = n_slots + 3`` tokens of ``hook_readout``):
#   index 0          -> SUPPORT
#   index 1 .. K     -> the K COUNT slots
#   index K + 1      -> ZERO
#   index K + 2      -> KNOWN
READOUT_SUPPORT = "support"
READOUT_ZERO = "zero"
READOUT_KNOWN = "known"


@dataclass
class Site:
    """How to extract a fixed ``[B, d]`` vector from a hook activation.

    Parameters
    ----------
    name:
        Hook-point name (see ``model.hook_points()``).
    pool:
        ``"mean"``   masked-mean over valid tokens of a ``[B, T, d]`` stream,
        ``"token"``  take a single token index of a ``[B, T, d]`` stream,
        ``"flat"``   already a per-example vector (e.g. ``slots.hook_slot_values``
                      ``[B, K]``) — returned unchanged,
        ``"readout"`` a named readout token (SUPPORT/ZERO/KNOWN) or a COUNT slot.
    mask_key:
        Which batch mask gates ``"mean"`` pooling.  For ``hook_factor`` use
        ``factor_mask``; for ``hook_query`` use ``query_mask``.  Readout/slot
        streams need no mask.
    token_index / readout:
        Selector for ``"token"`` / ``"readout"`` pooling.
    """

    name: str
    pool: str = "mean"
    mask_key: Optional[str] = None
    token_index: Optional[int] = None
    readout: Optional[str] = None


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def pool_site(x: torch.Tensor, site: Site, batch: Dict[str, torch.Tensor],
              n_slots: int) -> torch.Tensor:
    """Reduce a cached activation ``x`` to a per-example ``[B, d]`` vector."""
    if site.pool == "flat":
        return x.reshape(x.shape[0], -1)
    if site.pool == "token":
        idx = 0 if site.token_index is None else site.token_index
        return x[:, idx]
    if site.pool == "readout":
        sel = site.readout
        if sel == READOUT_SUPPORT:
            return x[:, 0]
        if sel == READOUT_ZERO:
            return x[:, n_slots + 1]
        if sel == READOUT_KNOWN:
            return x[:, n_slots + 2]
        if isinstance(sel, int):  # a COUNT slot 0..K-1 -> token 1..K
            return x[:, 1 + sel]
        # default: pooled mean over the K COUNT tokens
        return x[:, 1:1 + n_slots].mean(dim=1)
    # "mean" (default)
    if site.mask_key is not None and site.mask_key in batch:
        return _masked_mean(x, batch[site.mask_key].bool())
    return x.mean(dim=1)


def default_sites(n_encoder_layers: int, n_cross_layers: int) -> List[Site]:
    """A reasonable default probing menu spanning encoder -> cross -> readout."""
    sites: List[Site] = []
    for li in range(n_encoder_layers):
        sites.append(Site(f"encoder.blocks.{li}.hook_resid_post", "mean", "query_mask"))
    sites.append(Site("hook_factor", "mean", "factor_mask"))
    sites.append(Site("hook_query", "mean", "query_mask"))
    for li in range(n_cross_layers):
        sites.append(Site(f"cross_blocks.{li}.hook_resid_post", "mean"))
    sites.append(Site("hook_readout", "readout", readout=READOUT_SUPPORT))
    sites.append(Site("hook_readout", "readout", readout=READOUT_ZERO))
    sites.append(Site("hook_readout", "readout", readout=READOUT_KNOWN))
    sites.append(Site("slots.hook_slot_values", "flat"))
    return sites


def _site_key(site: Site) -> str:
    if site.pool == "readout":
        return f"{site.name}::{site.readout}"
    if site.pool == "token":
        return f"{site.name}::tok{site.token_index}"
    return site.name


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_activations(
    model: HookedModule,
    batch: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Run ``model`` once and return the raw cached activation for each name.

    ``names`` are bare hook-point names; pooling is the caller's job (see
    :func:`pool_site`).  Tensors are detached and moved to CPU.
    """
    names = list(dict.fromkeys(names))  # dedupe, preserve order
    _, cache = model.run_with_cache(batch, names=names)
    return {n: cache[n].detach().cpu() for n in names if n in cache}


@torch.no_grad()
def collect_pooled(
    model: HookedModule,
    batch: Dict[str, torch.Tensor],
    sites: Sequence[Site],
) -> Dict[str, torch.Tensor]:
    """Run once and return ``{site_key: pooled [B, d] cpu tensor}``."""
    names = list({s.name for s in sites})
    _, cache = model.run_with_cache(batch, names=names)
    n_slots = int(getattr(model, "n_slots", model.cfg.n_slots))
    out: Dict[str, torch.Tensor] = {}
    for s in sites:
        if s.name not in cache:
            continue
        out[_site_key(s)] = pool_site(cache[s.name], s, batch, n_slots).detach().cpu()
    return out


# ---------------------------------------------------------------------------
# Known-feature targets for ``(u, v, w)``
# ---------------------------------------------------------------------------
def _len_defect(u, v, w) -> int:
    return P.length(w) - P.length(u) - P.length(v)


def _multiset_summary(values: Sequence[int], k: int = 4) -> List[int]:
    """Sum + ``k`` largest entries — an order-invariant multiset fingerprint."""
    arr = sorted((int(x) for x in values), reverse=True)
    head = arr[:k] + [0] * (k - len(arr))
    return [int(sum(arr))] + head


def triple_feature_dict(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> Dict[str, object]:
    """All known regression/classification targets for one triple."""
    feats: Dict[str, object] = {}
    feats["ell_u"] = P.length(u)
    feats["ell_v"] = P.length(v)
    feats["ell_w"] = P.length(w)
    feats["len_defect"] = _len_defect(u, v, w)
    feats["n_right_desc_w"] = P.n_right_descents(w)
    feats["n_left_desc_w"] = P.n_left_descents(w)
    feats["n_right_desc_u"] = P.n_right_descents(u)
    feats["n_right_desc_v"] = P.n_right_descents(v)
    # rothe row / column length multisets (order-invariant summary)
    feats["rothe_row_w"] = _multiset_summary(rothe.rothe_row_lengths(w))
    feats["rothe_col_w"] = _multiset_summary(rothe.rothe_col_lengths(w))
    # family tags (multilabel) of w
    feats["family_w"] = family_multihot(KF.family_tags(w))
    return feats


# regression (Ridge) vs. boolean (logistic) target kinds
SCALAR_FEATURES = [
    "ell_u", "ell_v", "ell_w", "len_defect",
    "n_right_desc_w", "n_left_desc_w", "n_right_desc_u", "n_right_desc_v",
]
VECTOR_FEATURES = ["rothe_row_w", "rothe_col_w"]
MULTILABEL_FEATURES = {"family_w": FAMILY_TAGS}


def probe_targets(
    perms_by_id: Dict[int, Sequence[int]],
    u_ids: Sequence[int],
    v_ids: Sequence[int],
    w_ids: Sequence[int],
    *,
    positive: Optional[Sequence[int]] = None,
    coeff_bin: Optional[Sequence[int]] = None,
) -> Dict[str, Tuple[np.ndarray, str]]:
    """Build ``{feature_name: (Y, kind)}`` for a batch of triples.

    ``kind`` is one of ``"reg"`` (Ridge, R^2), ``"multi"`` (per-label logistic,
    macro accuracy) or ``"clf"`` (logistic, accuracy).
    """
    rows = [triple_feature_dict(perms_by_id[int(a)], perms_by_id[int(b)], perms_by_id[int(c)])
            for a, b, c in zip(u_ids, v_ids, w_ids)]
    targets: Dict[str, Tuple[np.ndarray, str]] = {}
    for name in SCALAR_FEATURES:
        targets[name] = (np.array([r[name] for r in rows], dtype=np.float64), "reg")
    for name in VECTOR_FEATURES:
        targets[name] = (np.array([r[name] for r in rows], dtype=np.float64), "reg")
    for name in MULTILABEL_FEATURES:
        targets[name] = (np.array([r[name] for r in rows], dtype=np.int64), "multi")
    if positive is not None:
        targets["positivity"] = (np.asarray(positive, dtype=np.int64), "clf")
    if coeff_bin is not None:
        targets["coeff_bin"] = (np.asarray(coeff_bin, dtype=np.int64), "clf")
    return targets


# ---------------------------------------------------------------------------
# Probe fitting
# ---------------------------------------------------------------------------
def _fit_score(X: np.ndarray, y: np.ndarray, kind: str, *,
               test_frac: float = 0.25, seed: int = 0, max_iter: int = 500) -> Dict[str, float]:
    """Fit a probe on a train split, report the held-out score."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    te, tr = perm[:n_test], perm[n_test:]
    if len(tr) < 2:
        tr = perm  # degenerate: tiny smoke batches just fit-on-all
        te = perm
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    if kind == "reg":
        from sklearn.metrics import r2_score
        Y = y.reshape(n, -1)
        model = Ridge(alpha=1.0).fit(Xtr, Y[tr])
        pred = model.predict(Xte)
        try:
            r2 = float(r2_score(Y[te], pred))
        except Exception:
            r2 = float("nan")
        return {"r2": r2, "n_train": int(len(tr)), "dim": int(Y.shape[1])}

    if kind == "multi":
        from sklearn.metrics import accuracy_score
        accs = []
        for c in range(y.shape[1]):
            yc = y[:, c]
            if len(np.unique(yc[tr])) < 2:
                accs.append(float((yc[te] == yc[tr][0]).mean()) if len(tr) else float("nan"))
                continue
            clf = LogisticRegression(max_iter=max_iter).fit(Xtr, yc[tr])
            accs.append(float(accuracy_score(yc[te], clf.predict(Xte))))
        return {"macro_acc": float(np.nanmean(accs)), "per_label_acc": [round(a, 4) for a in accs],
                "n_train": int(len(tr))}

    # "clf"
    from sklearn.metrics import accuracy_score
    if len(np.unique(y[tr])) < 2:
        acc = float((y[te] == y[tr][0]).mean()) if len(tr) else float("nan")
        return {"acc": acc, "n_train": int(len(tr)), "note": "single-class-train"}
    clf = LogisticRegression(max_iter=max_iter).fit(Xtr, y[tr])
    return {"acc": float(accuracy_score(y[te], clf.predict(Xte))), "n_train": int(len(tr))}


def fit_probes(
    activations: Dict[str, np.ndarray],
    targets: Dict[str, Tuple[np.ndarray, str]],
    *,
    seed: int = 0,
    test_frac: float = 0.25,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Fit one probe per (site, feature) pair; return nested score dict."""
    scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for site_key, X in activations.items():
        Xa = np.asarray(X, dtype=np.float64)
        site_scores: Dict[str, Dict[str, float]] = {}
        for fname, (y, kind) in targets.items():
            try:
                site_scores[fname] = _fit_score(Xa, y, kind, seed=seed, test_frac=test_frac)
            except Exception as e:  # keep the run alive; record the failure
                site_scores[fname] = {"error": str(e)}
        scores[site_key] = site_scores
    return scores


# ---------------------------------------------------------------------------
# End-to-end driver over a dataset
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_probes(
    model: HookedModule,
    dataset,
    sites: Sequence[Site],
    device: str = "cuda",
    *,
    max_examples: int = 4096,
    batch_size: int = 256,
    seed: int = 0,
) -> Dict[str, object]:
    """Collect pooled activations + known targets over ``dataset`` and fit probes.

    ``dataset`` is a :class:`~schubml.data.datasets.SchubertTripleDataset`
    (it exposes ``perms_by_id`` and the underlying id arrays ``a``).  Returns
    ``{"scores": {...}, "sites": [...], "n_examples": int}``.
    """
    model.eval().to(device)
    n = min(len(dataset), max_examples)
    a = dataset.a
    perms_by_id = dataset.perms_by_id

    pooled: Dict[str, List[np.ndarray]] = {}
    u_ids, v_ids, w_ids, pos, cbin = [], [], [], [], []
    from ..models.heads import coeff_to_bin

    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        exs = [dataset[i] for i in idxs]
        batch = to_device(collate(exs), device)
        bp = collect_pooled(model, batch, sites)
        for k, v in bp.items():
            pooled.setdefault(k, []).append(v.numpy())
        for i in idxs:
            u_ids.append(int(a["u_id"][i]))
            v_ids.append(int(a["v_id"][i]))
            w_ids.append(int(a["w_id"][i]))
            pos.append(int(a["positive"][i]))
            cbin.append(coeff_to_bin(int(a["coefficient"][i])))

    activations = {k: np.concatenate(v, axis=0) for k, v in pooled.items()}
    targets = probe_targets(perms_by_id, u_ids, v_ids, w_ids, positive=pos, coeff_bin=cbin)
    scores = fit_probes(activations, targets, seed=seed)
    return {
        "scores": scores,
        "sites": [_site_key(s) for s in sites],
        "features": list(targets.keys()),
        "n_examples": int(n),
    }
