"""Held-out evaluation table assembly + exact known-rule baselines.

Two responsibilities, both from PROJECT.md:

* **Required evaluation table (12.2).**  Given a flat ``predictions`` dict (the
  arrays a model / baseline produces over a set of triples), stratify it by
  negative tier, by family tag and by split, and emit the rows of the required
  evaluation table -- positivity, coefficient and support-ranking metrics, with
  easy and hard zeroes kept *separate* (12.1).

* **Known-rule baseline (11.4).**  For products where one factor is a simple
  reflection ``s_r``, Monk's rule gives the exact support.  ``known_rule_baseline_monk``
  predicts that support with ``KF.monk_product`` and reports precision / recall /
  F1 against the stored support -- an exact, training-free reference point.

Metrics come from :mod:`schubml.train.metrics`; nothing here trains a model.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..combinatorics import known_families as KF
from ..combinatorics import permutations as P
from ..data.labels import TIERS
from ..data.schemas import (PERM_JSON_COLUMNS, PRODUCT_JSON_COLUMNS, load_table)
from ..train.metrics import (coefficient_metrics, positivity_metrics,
                             support_ranking_metrics)

Perm = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Stratified metric rows from a predictions dict (PROJECT.md 12.1, 12.2)
# ---------------------------------------------------------------------------
# A ``predictions`` dict is expected to carry parallel numpy arrays:
#   pos_logit, positive, c_hat, coefficient, tier_id, product_id
# plus optionally support_score (defaults to pos_logit) and family_tags
# (a list-of-lists, one tag list per triple) for family stratification.
HARD_TIER_IDS = (3, 4, 5, 6)   # Z2..Z5  (TIERS index)
EASY_TIER_IDS = (1, 2)         # Z0, Z1


def _block(pred: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, float]:
    """All applicable metrics for the subset selected by ``mask``."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    out: Dict[str, float] = {"n": n}
    y = pred["positive"][mask].astype(int)
    out.update({f"pos_{k}": v for k, v in positivity_metrics(pred["pos_logit"][mask], y).items()})
    if "c_hat" in pred and "coefficient" in pred:
        out.update({f"coeff_{k}": v for k, v in coefficient_metrics(
            pred["c_hat"][mask], pred["coefficient"][mask].astype(int)).items()})
    if "product_id" in pred:
        score = pred.get("support_score", pred["pos_logit"])
        out.update({f"support_{k}": v for k, v in support_ranking_metrics(
            score[mask], pred["positive"][mask], pred["coefficient"][mask],
            pred["product_id"][mask]).items()})
    return out


def stratify_by_tier(pred: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Metrics per negative tier (positives always included for a 2-class split).

    Also emits the aggregate ``easy`` (Z0/Z1) and ``hard`` (Z2--Z5) slices that
    PROJECT.md 12.1 insists on keeping separate, plus ``all``.
    """
    tier_id = pred["tier_id"].astype(int)
    pos = pred["positive"] > 0
    rows: Dict[str, Dict[str, float]] = {}
    rows["all"] = _block(pred, np.ones(len(tier_id), dtype=bool))
    for tid, tname in enumerate(TIERS):
        if tname == "positive":
            continue
        rows[tname] = _block(pred, pos | (tier_id == tid))
    rows["easy"] = _block(pred, pos | np.isin(tier_id, EASY_TIER_IDS))
    rows["hard"] = _block(pred, pos | np.isin(tier_id, HARD_TIER_IDS))
    return rows


def stratify_by_family(pred: Dict[str, object],
                       families: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Metrics restricted to triples whose ``w`` carries each family tag.

    Requires ``pred['family_tags']`` (a list of tag-lists, one per triple).  If
    ``families`` is ``None`` every tag observed in the data is used.
    """
    fam_lists = pred.get("family_tags")
    if fam_lists is None:
        return {}
    n = len(fam_lists)
    if families is None:
        families = sorted({t for tags in fam_lists for t in tags})
    rows: Dict[str, Dict[str, float]] = {}
    for fam in families:
        mask = np.array([fam in fam_lists[i] for i in range(n)], dtype=bool)
        rows[fam] = _block(pred, mask)
    return rows


def stratify_by_split(pred: Dict[str, object],
                      split_tags: Sequence[Sequence[str]],
                      splits: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Metrics restricted to each named split tag.

    ``split_tags`` is the per-triple list of split-tag strings (as stored in
    ``triples.parquet``); ``splits`` is the set of tags to break out (e.g.
    ``["ambient_ood:test", "coeff_mag:test"]``).
    """
    n = len(split_tags)
    rows: Dict[str, Dict[str, float]] = {}
    for split in splits:
        mask = np.array([split in split_tags[i] for i in range(n)], dtype=bool)
        rows[split] = _block(pred, mask)
    return rows


def evaluation_table(pred: Dict[str, object]) -> Dict[str, object]:
    """Assemble the PROJECT.md 12.2 required evaluation table from ``pred``.

    Produces the canonical rows -- product-disjoint, hard-zero Z2/Z3/Z4-Z5,
    family-holdout and support-ranking -- from a single predictions dict over the
    relevant triples.  Easy and hard zeroes are reported separately (12.1).
    """
    by_tier = stratify_by_tier(pred)
    table: Dict[str, object] = {
        "overall": by_tier["all"],
        "hard_zero_Z2": by_tier.get("Z2", {"n": 0}),
        "hard_zero_Z3": by_tier.get("Z3", {"n": 0}),
        "hard_zero_Z45": _merge_blocks(pred, ("Z4", "Z5")),
        "easy_zero": by_tier["easy"],
        "hard_zero": by_tier["hard"],
        "by_tier": by_tier,
    }
    fam = stratify_by_family(pred)
    if fam:
        table["by_family"] = fam
    return table


def _merge_blocks(pred: Dict[str, np.ndarray], tnames: Sequence[str]) -> Dict[str, float]:
    """Metrics over the union of several tiers (e.g. the Z4/Z5 adversarial slice)."""
    tids = [TIERS.index(t) for t in tnames if t in TIERS]
    tier_id = pred["tier_id"].astype(int)
    pos = pred["positive"] > 0
    return _block(pred, pos | np.isin(tier_id, tids))


# ---------------------------------------------------------------------------
# Known-rule baseline: Monk / Chevalley (PROJECT.md 11.4)
# ---------------------------------------------------------------------------
def _simple_reflection_index(w: Sequence[int]) -> Optional[int]:
    """If ``w`` is a simple reflection ``s_r`` return ``r``, else ``None``.

    ``s_r`` is the canonical permutation ``(1, .., r-1, r+1, r, ..)`` whose only
    descent is at position ``r``; canonically this is ``(2, 1)`` shifted, i.e. its
    single right descent is at ``r``.
    """
    if P.length(w) != 1:
        return None
    desc = P.right_descents(w)
    return desc[0] if len(desc) == 1 else None


def known_rule_baseline_monk(data_dir: str, *, max_products: Optional[int] = None) -> Dict[str, object]:
    """Exact Monk-rule support prediction vs the stored support (PROJECT.md 11.4).

    Scans ``products.parquet`` for products where one factor is a simple
    reflection ``s_r``.  For each, predicts the support of ``S_{s_r} * S_w`` with
    :func:`schubml.combinatorics.known_families.monk_product` (an independent,
    training-free solver) and compares it to the stored support, accumulating
    micro precision / recall / F1 over the predicted ``w`` set.

    ``monk_product`` enumerates Bruhat covers *within the ambient size of* ``w``,
    so the candidate ``w`` is padded to the product's ``ambient_n_compute`` (at
    least ``r + 1``) before the call -- otherwise covers that move a trailing
    fixed point of the canonical ``w`` are missed and recall is spuriously low.

    Also checks the predicted coefficients (Monk's rule always yields ``1``)
    against the stored coefficients, reporting exact-coefficient agreement.

    Returns a dict with ``n_products`` matched, micro/macro precision-recall-F1
    and per-coefficient accuracy.
    """
    perm_df = load_table(os.path.join(data_dir, "permutations.parquet"), PERM_JSON_COLUMNS)
    prod_df = load_table(os.path.join(data_dir, "products.parquet"), PRODUCT_JSON_COLUMNS)
    canon_by_id: Dict[int, Perm] = {
        int(pid): tuple(c) for pid, c in zip(perm_df["perm_id"], perm_df["canonical"])
    }

    tp = fp = fn = 0
    coeff_correct = coeff_total = 0
    per_product_f1: List[float] = []
    n_matched = 0
    n_coeff_one = 0  # stored coefficients that equal 1 (Monk's rule predicts all-ones)

    for row in prod_df.itertuples(index=False):
        if max_products is not None and n_matched >= max_products:
            break
        u = canon_by_id[int(row.u_id)]
        v = canon_by_id[int(row.v_id)]
        # identify which factor (if any) is a simple reflection
        ru = _simple_reflection_index(u)
        rv = _simple_reflection_index(v)
        if ru is not None:
            r, w = ru, v
        elif rv is not None:
            r, w = rv, u
        else:
            continue
        n_matched += 1

        # pad w to the product's ambient compute size (>= r+1) so monk_product
        # sees every relevant Bruhat cover (see the note in the docstring).
        n_compute = int(getattr(row, "ambient_n_compute", 0)) or 0
        n_pad = max(n_compute, r + 1, len(w))
        w_padded = P.pad(P.canonical_perm(w), n_pad)
        predicted = KF.monk_product(r, w_padded)  # {canonical_w': coeff}
        pred_support = {P.canonical_perm(k): c for k, c in predicted.items()}

        stored = {
            P.canonical_perm(canon_by_id[int(wid)]): int(c)
            for wid, c in zip(row.support_w_ids, row.support_coeffs)
        }
        pred_keys = set(pred_support)
        stored_keys = set(stored)
        tp += len(pred_keys & stored_keys)
        fp += len(pred_keys - stored_keys)
        fn += len(stored_keys - pred_keys)

        # per-product F1
        ptp = len(pred_keys & stored_keys)
        pp = ptp / max(1, len(pred_keys))
        pr = ptp / max(1, len(stored_keys))
        per_product_f1.append(2 * pp * pr / (pp + pr) if (pp + pr) else 0.0)

        # coefficient agreement on the intersection
        for k in pred_keys & stored_keys:
            coeff_total += 1
            if pred_support[k] == stored[k]:
                coeff_correct += 1
        n_coeff_one += sum(1 for c in stored.values() if c == 1)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "rule": "monk",
        "description": "S_{s_r} * S_w support via Monk's rule (one factor a simple reflection).",
        "n_products": n_matched,
        "support_precision": round(precision, 6),
        "support_recall": round(recall, 6),
        "support_f1": round(f1, 6),
        "support_macro_f1": round(float(np.mean(per_product_f1)) if per_product_f1 else 0.0, 6),
        "coeff_exact_acc": round(coeff_correct / max(1, coeff_total), 6),
        "n_support_cells": tp + fn,
        "n_stored_coeff_eq_1": n_coeff_one,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
