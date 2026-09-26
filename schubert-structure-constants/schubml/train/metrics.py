"""Evaluation metrics (PROJECT.md 12.1).

Pure functions over numpy arrays; the evaluation script stratifies the inputs by
split / negative tier / family before calling these.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _safe(fn, *a, default=float("nan"), **k):
    try:
        return float(fn(*a, **k))
    except Exception:
        return default


def positivity_metrics(pos_logit: np.ndarray, positive: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )
    prob = 1.0 / (1.0 + np.exp(-pos_logit))
    pred = (pos_logit > 0).astype(int)
    y = positive.astype(int)
    out = {
        "accuracy": float((pred == y).mean()),
        "precision": _safe(precision_score, y, pred, zero_division=0),
        "recall": _safe(recall_score, y, pred, zero_division=0),
        "f1": _safe(f1_score, y, pred, zero_division=0),
        "mcc": _safe(matthews_corrcoef, y, pred),
        "auroc": _safe(roc_auc_score, y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": _safe(average_precision_score, y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "n": int(len(y)),
        "base_rate": float(y.mean()),
    }
    return out


def coefficient_metrics(c_hat: np.ndarray, coeff: np.ndarray,
                        bin_pred: Optional[np.ndarray] = None,
                        bin_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    c_round = np.rint(np.clip(c_hat, 0, None)).astype(int)
    out = {
        "exact_acc": float((c_round == coeff).mean()),
        "exact_small_acc": float((np.minimum(c_round, 10) == np.minimum(coeff, 10)).mean()),
        "mae": float(np.abs(c_hat - coeff).mean()),
        "mae_log": float(np.abs(np.log1p(np.clip(c_hat, 0, None)) - np.log1p(coeff)).mean()),
    }
    # restrict exact accuracy to positives (more informative)
    posmask = coeff > 0
    if posmask.any():
        out["exact_acc_pos"] = float((c_round[posmask] == coeff[posmask]).mean())
    if bin_pred is not None and bin_true is not None:
        out["bin_acc"] = float((bin_pred == bin_true).mean())
    return out


def mod_accuracy(mod_pred: Dict[int, np.ndarray], coeff: np.ndarray) -> Dict[str, float]:
    out = {}
    for p, pred in mod_pred.items():
        out[f"mod{p}_acc"] = float((pred == (coeff % p)).mean())
    return out


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (prob >= lo) & (prob < hi if b < n_bins - 1 else prob <= hi)
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def dcg(rels: np.ndarray) -> float:
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(scores: np.ndarray, rels: np.ndarray, k: Optional[int] = None) -> float:
    order = np.argsort(-scores)
    rels_sorted = rels[order]
    ideal = np.sort(rels)[::-1]
    if k is not None:
        rels_sorted = rels_sorted[:k]
        ideal = ideal[:k]
    idcg = dcg(ideal)
    return float(dcg(rels_sorted) / idcg) if idcg > 0 else float("nan")


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-scores)
    p = positive[order].astype(float)
    if p.sum() == 0:
        return float("nan")
    cum = np.cumsum(p)
    prec_at = cum / (np.arange(len(p)) + 1)
    return float((prec_at * p).sum() / p.sum())


def support_ranking_metrics(scores: np.ndarray, positive: np.ndarray,
                            coeff: np.ndarray, product_ids: np.ndarray) -> Dict[str, float]:
    """Per-product NDCG (relevance = coefficient) and MAP (relevance = positive)."""
    ndcgs, maps = [], []
    for pid in np.unique(product_ids):
        m = product_ids == pid
        if m.sum() < 2:
            continue
        s, pos, c = scores[m], positive[m], coeff[m].astype(float)
        if pos.sum() == 0 or pos.sum() == m.sum():
            continue
        ndcgs.append(ndcg_at_k(s, c))
        maps.append(average_precision(s, pos))
    return {
        "ndcg": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
        "map": float(np.nanmean(maps)) if maps else float("nan"),
        "n_products": len(ndcgs),
    }
