"""Feature mining for counting slots and SAE features (PROJECT.md 13.7, 13.8).

For each counting slot (or SAE feature) we:

1. collect the top / bottom activating triples (indices into a dataset);
2. correlate the activation with **known** combinatorial features
   (lengths, descents, Rothe sizes, Bruhat indicators, family tags);
3. emit a *feature card* matching the PROJECT.md 13.7 schema.

The cards are observational (Level-2) — they flag candidate meanings; the causal
``causal_ablation_effect`` / ``patching_effect`` fields are left ``None`` here and
filled in by the patching tools when a circuit is confirmed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..combinatorics import bruhat
from ..combinatorics import known_families as KF
from ..combinatorics import permutations as P
from ..combinatorics import rothe
from ..data.collators import collate, to_device
from ..data.labels import FAMILY_TAGS, family_multihot
from .hooks import HookedModule

# ---------------------------------------------------------------------------
# Known scalar features of a triple, used for correlation
# ---------------------------------------------------------------------------
def known_feature_vector(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> Dict[str, float]:
    """Named scalar features of ``(u, v, w)`` (PROJECT.md 13.7 correlation set)."""
    feats: Dict[str, float] = {
        "ell_u": P.length(u),
        "ell_v": P.length(v),
        "ell_w": P.length(w),
        "len_defect": P.length(w) - P.length(u) - P.length(v),
        "n_right_desc_w": P.n_right_descents(w),
        "n_left_desc_w": P.n_left_descents(w),
        "n_right_desc_u": P.n_right_descents(u),
        "n_right_desc_v": P.n_right_descents(v),
        "rothe_size_w": len(rothe.rothe_cells(w)),
        "max_rothe_row_w": max(rothe.rothe_row_lengths(w) or [0]),
        "max_rothe_col_w": max(rothe.rothe_col_lengths(w) or [0]),
        # Bruhat comparability indicators (necessary positivity conditions)
        "bruhat_u_leq_w": float(bruhat.bruhat_leq(u, w)),
        "bruhat_v_leq_w": float(bruhat.bruhat_leq(v, w)),
    }
    fam = family_multihot(KF.family_tags(w))
    for tag, val in zip(FAMILY_TAGS, fam):
        feats[f"fam_{tag}"] = float(val)
    return feats


def _feature_matrix(perms_by_id, u_ids, v_ids, w_ids):
    """Stack known-feature vectors -> ``(names, X[N, F])``."""
    rows = [known_feature_vector(perms_by_id[int(a)], perms_by_id[int(b)], perms_by_id[int(c)])
            for a, b, c in zip(u_ids, v_ids, w_ids)]
    names = list(rows[0].keys())
    X = np.array([[r[k] for k in names] for r in rows], dtype=np.float64)
    return names, X


def _correlations(activation: np.ndarray, names: List[str], X: np.ndarray,
                  *, top: int = 8) -> List[Dict[str, object]]:
    """Pearson correlation of one activation vector with each known feature."""
    a = activation.astype(np.float64)
    a = a - a.mean()
    sa = a.std()
    corrs = []
    for j, nm in enumerate(names):
        col = X[:, j] - X[:, j].mean()
        sc = col.std()
        if sa < 1e-9 or sc < 1e-9:
            r = 0.0
        else:
            r = float((a * col).mean() / (sa * sc))
        corrs.append((nm, r))
    corrs.sort(key=lambda t: -abs(t[1]))
    return [{"feature": nm, "corr": round(r, 4)} for nm, r in corrs[:top]]


# ---------------------------------------------------------------------------
# Gather slot / SAE activations together with triple ids
# ---------------------------------------------------------------------------
@torch.no_grad()
def _gather_slot_activations(model: HookedModule, dataset, device: str,
                             max_examples: int, batch_size: int, hook: str):
    """Return ``(acts[N, K], u_ids, v_ids, w_ids, coeff, positive)``."""
    model.eval().to(device)
    n = min(len(dataset), max_examples)
    a = dataset.a
    acts: List[np.ndarray] = []
    u_ids, v_ids, w_ids, coeff, pos = [], [], [], [], []
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        exs = [dataset[i] for i in idxs]
        batch = to_device(collate(exs), device)
        _, cache = model.run_with_cache(batch, names=[hook])
        acts.append(cache[hook].detach().cpu().numpy())
        for i in idxs:
            u_ids.append(int(a["u_id"][i])); v_ids.append(int(a["v_id"][i]))
            w_ids.append(int(a["w_id"][i]))
            coeff.append(int(a["coefficient"][i])); pos.append(int(a["positive"][i]))
    return (np.concatenate(acts, axis=0), u_ids, v_ids, w_ids,
            np.array(coeff), np.array(pos))


def _triple_label(perms_by_id, uid, vid, wid) -> Dict[str, object]:
    return {
        "u": list(perms_by_id[int(uid)]),
        "v": list(perms_by_id[int(vid)]),
        "w": list(perms_by_id[int(wid)]),
        "u_id": int(uid), "v_id": int(vid), "w_id": int(wid),
    }


# ---------------------------------------------------------------------------
# Slot mining (PROJECT.md 13.8)
# ---------------------------------------------------------------------------
def mine_slot_features(
    model: HookedModule,
    dataset,
    device: str = "cuda",
    *,
    hook: str = "slots.hook_slot_values",
    max_examples: int = 4096,
    batch_size: int = 256,
    top_k: int = 10,
) -> List[Dict[str, object]]:
    """Feature cards for each counting slot.

    For slot ``a``: collect examples with large / zero ``z_a``, correlate with
    known features, and record whether the slot value tracks the coefficient.
    """
    acts, u_ids, v_ids, w_ids, coeff, pos = _gather_slot_activations(
        model, dataset, device, max_examples, batch_size, hook)
    perms_by_id = dataset.perms_by_id
    names, X = _feature_matrix(perms_by_id, u_ids, v_ids, w_ids)
    K = acts.shape[1]
    cards: List[Dict[str, object]] = []
    for slot in range(K):
        col = acts[:, slot]
        order = np.argsort(-col)
        top_idx = [int(i) for i in order[:top_k] if col[i] > 0]
        bot_idx = [int(i) for i in order[::-1][:top_k]]
        # correlation of slot value with the (exact) coefficient
        if col.std() > 1e-9 and coeff.std() > 1e-9:
            coeff_corr = float(np.corrcoef(col, coeff.astype(float))[0, 1])
        else:
            coeff_corr = 0.0
        cards.append({
            "feature_id": f"slot_{slot}",
            "layer_site": hook,
            "sparsity": float((col > 1e-6).mean()),
            "mean_activation": float(col[col > 1e-6].mean()) if (col > 1e-6).any() else 0.0,
            "max_activation": float(col.max()),
            "top_activating_triples": [_triple_label(perms_by_id, u_ids[i], v_ids[i], w_ids[i])
                                       for i in top_idx],
            "bottom_activating_triples": [_triple_label(perms_by_id, u_ids[i], v_ids[i], w_ids[i])
                                          for i in bot_idx],
            "correlated_known_features": _correlations(col, names, X),
            "coefficient_correlation": round(coeff_corr, 4),
            "causal_ablation_effect": None,
            "patching_effect": None,
            "candidate_rule": None,
            "status": "observational",
        })
    return cards


# ---------------------------------------------------------------------------
# SAE feature mining (PROJECT.md 13.7)
# ---------------------------------------------------------------------------
def mine_sae_features(
    sae,
    activations: torch.Tensor,
    perms_by_id: Dict[int, Sequence[int]],
    u_ids: Sequence[int],
    v_ids: Sequence[int],
    w_ids: Sequence[int],
    *,
    coeff: Optional[Sequence[int]] = None,
    layer_site: str = "sae",
    top_k: int = 10,
    max_features: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Feature cards for SAE features given a *pooled* (per-example) activation set.

    ``activations`` is the ``[N, d_in]`` matrix the SAE was trained on, aligned
    row-for-row with ``u_ids/v_ids/w_ids``.  Encodes, then for each hidden
    feature records top activating triples + known-feature correlations.
    """
    from .sae import encode

    z = encode(sae, activations).numpy()  # [N, H]
    names, X = _feature_matrix(perms_by_id, u_ids, v_ids, w_ids)
    H = z.shape[1] if max_features is None else min(z.shape[1], max_features)
    coeff_arr = None if coeff is None else np.asarray(coeff, dtype=float)
    cards: List[Dict[str, object]] = []
    for f in range(H):
        col = z[:, f]
        active = col > 0
        order = np.argsort(-col)
        top_idx = [int(i) for i in order[:top_k] if col[i] > 0]
        bot_idx = [int(i) for i in order[::-1][:top_k]]
        if coeff_arr is not None and col.std() > 1e-9 and coeff_arr.std() > 1e-9:
            coeff_corr = round(float(np.corrcoef(col, coeff_arr)[0, 1]), 4)
        else:
            coeff_corr = None
        cards.append({
            "feature_id": f"sae_{f}",
            "layer_site": layer_site,
            "sparsity": float(active.mean()),
            "mean_activation": float(col[active].mean()) if active.any() else 0.0,
            "max_activation": float(col.max()) if len(col) else 0.0,
            "top_activating_triples": [_triple_label(perms_by_id, u_ids[i], v_ids[i], w_ids[i])
                                       for i in top_idx],
            "bottom_activating_triples": [_triple_label(perms_by_id, u_ids[i], v_ids[i], w_ids[i])
                                          for i in bot_idx],
            "correlated_known_features": _correlations(col, names, X),
            "coefficient_correlation": coeff_corr,
            "causal_ablation_effect": None,
            "patching_effect": None,
            "candidate_rule": None,
            "status": "observational",
        })
    return cards
