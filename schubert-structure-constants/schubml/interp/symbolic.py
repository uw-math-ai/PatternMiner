"""Symbolic distillation and conjecture mining (PROJECT.md section 14).

Converts neural features into human-readable rules.  Three pieces:

1. **Predicate DSL** — :func:`predicate_features` extracts a *named* boolean / int
   feature vector from a triple ``(u, v, w)``.  Features cover the PROJECT.md 14.2
   predicate language: descent compatibility, Bruhat comparability ``u <= w`` and
   ``v <= w``, rank-matrix inequality counts ``r_w(i,j)`` vs ``r_u(i,j)+r_v(i,j)``,
   Rothe row/column interlacing counts, the length defect, pattern (family) tags,
   and support-interval features.

2. **Rule learners** (PROJECT.md 14.3) — sparse logistic regression, a shallow
   decision tree, and a simple monotone rule list, each predicting positivity
   ``c > 0`` from the predicates.  All return a callable ``rule_fn(features)->bool``
   plus a human-readable description.

3. **Validation harness** :func:`validate_rule` (PROJECT.md 14.4) reporting
   accuracy on held-out data and stratified by tier; and a conjecture-card writer
   :func:`write_conjecture_card` following the PROJECT.md 14.5 markdown template.

Do not trust a rule discovered on random triples until it survives the hard-zero
and product-disjoint tests (PROJECT.md 14.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..combinatorics import bruhat
from ..combinatorics import known_families as KF
from ..combinatorics import permutations as P
from ..combinatorics import rank_matrix as RM
from ..combinatorics import rothe
from ..data.labels import FAMILY_TAGS

Triple = Tuple[Sequence[int], Sequence[int], Sequence[int]]

# ---------------------------------------------------------------------------
# 14.2 Predicate language
# ---------------------------------------------------------------------------
def predicate_features(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> Dict[str, int]:
    """Named boolean / integer predicate features for one triple.

    All values are ints (booleans as 0/1) so they can feed sklearn directly and
    be quoted verbatim in a conjecture card.
    """
    n = max(len(u), len(v), len(w))
    up, vp, wp = P.pad(u, n), P.pad(v, n), P.pad(w, n)

    lu, lv, lw = P.length(up), P.length(vp), P.length(wp)
    feats: Dict[str, int] = {}

    # --- lengths / length defect ---
    feats["ell_u"] = lu
    feats["ell_v"] = lv
    feats["ell_w"] = lw
    feats["len_defect"] = lw - lu - lv               # = 0 is necessary for c>0
    feats["len_defect_zero"] = int(lw == lu + lv)
    feats["len_defect_nonneg"] = int(lw >= lu + lv)

    # --- Bruhat comparability (necessary positivity conditions) ---
    u_leq_w = bruhat.bruhat_leq(up, wp)
    v_leq_w = bruhat.bruhat_leq(vp, wp)
    feats["bruhat_u_leq_w"] = int(u_leq_w)
    feats["bruhat_v_leq_w"] = int(v_leq_w)
    feats["bruhat_both_leq_w"] = int(u_leq_w and v_leq_w)

    # --- descent compatibility (D_R(u) ∪ D_R(v) vs D_R(w)) ---
    du, dv, dw = set(P.right_descents(up)), set(P.right_descents(vp)), set(P.right_descents(wp))
    feats["n_right_desc_u"] = len(du)
    feats["n_right_desc_v"] = len(dv)
    feats["n_right_desc_w"] = len(dw)
    feats["desc_u_subset_w"] = int(du <= dw)
    feats["desc_v_subset_w"] = int(dv <= dw)
    feats["desc_union_subset_w"] = int((du | dv) <= dw)
    feats["desc_w_subset_union"] = int(dw <= (du | dv))
    feats["n_desc_w_not_in_union"] = len(dw - (du | dv))
    # left descents (value-side)
    ldu, ldv, ldw = set(P.left_descents(up)), set(P.left_descents(vp)), set(P.left_descents(wp))
    feats["left_desc_union_subset_w"] = int((ldu | ldv) <= ldw)

    # --- rank-matrix inequality  r_w(i,j) <= r_u(i,j) + r_v(i,j) ---
    Ru, Rv, Rw = RM.rank_matrix(up), RM.rank_matrix(vp), RM.rank_matrix(wp)
    diff = (Ru + Rv) - Rw                              # (n+1)x(n+1); ignore borders
    inner = diff[1:n, 1:n] if n >= 2 else diff[:0, :0]
    feats["rank_ineq_violations"] = int((inner < 0).sum())     # cells where r_w > r_u+r_v
    feats["rank_ineq_holds_all"] = int((inner >= 0).all()) if inner.size else 1
    feats["rank_ineq_min_slack"] = int(inner.min()) if inner.size else 0
    feats["rank_ineq_total_slack"] = int(inner.sum()) if inner.size else 0

    # --- Rothe row/column interlacing counts ---
    ru_rows, rv_rows, rw_rows = (rothe.rothe_row_lengths(up), rothe.rothe_row_lengths(vp),
                                 rothe.rothe_row_lengths(wp))
    ru_cols, rv_cols, rw_cols = (rothe.rothe_col_lengths(up), rothe.rothe_col_lengths(vp),
                                 rothe.rothe_col_lengths(wp))
    feats["rothe_size_u"] = sum(ru_rows)
    feats["rothe_size_v"] = sum(rv_rows)
    feats["rothe_size_w"] = sum(rw_rows)
    feats["rothe_row_sum_geq"] = int(_rowwise_geq(rw_rows, ru_rows, rv_rows))
    feats["rothe_col_sum_geq"] = int(_rowwise_geq(rw_cols, ru_cols, rv_cols))
    feats["rothe_row_interlace_viol"] = _rowwise_violations(rw_rows, ru_rows, rv_rows)
    feats["rothe_col_interlace_viol"] = _rowwise_violations(rw_cols, ru_cols, rv_cols)

    # --- pattern / family tags of each factor and w ---
    feats["w_grassmannian"] = int(KF.is_grassmannian(wp))
    feats["w_vexillary"] = int(KF.is_vexillary(wp))
    feats["w_dominant"] = int(KF.is_dominant(wp))
    feats["w_boolean"] = int(KF.is_boolean(wp))
    feats["w_321_avoiding"] = int(KF.is_321_avoiding(wp))
    feats["uv_both_grassmannian"] = int(KF.is_grassmannian(up) and KF.is_grassmannian(vp))
    feats["uv_same_grass_descent"] = int(_same_grass_descent(up, vp))
    feats["u_or_v_simple"] = int(KF.is_simple_reflection(up) or KF.is_simple_reflection(vp))

    # --- support-interval features (Bruhat interval [max(u,v)-ish, w]) ---
    # w must dominate both factors; the "interval slack" is ell(w) - max(ell(u),ell(v)).
    feats["interval_slack"] = lw - max(lu, lv)
    feats["w_geq_u_and_v_len"] = int(lw >= lu and lw >= lv)
    # monk-style cover indicator: is w a single Bruhat cover above u or v?
    feats["w_covers_u"] = int(bruhat.bruhat_covers(up, wp))
    feats["w_covers_v"] = int(bruhat.bruhat_covers(vp, wp))

    return feats


def _rowwise_geq(target: Sequence[int], a: Sequence[int], b: Sequence[int]) -> bool:
    """True iff ``target[i] >= a[i] + b[i]`` for every row (padding short lists)."""
    m = max(len(target), len(a), len(b))
    t = list(target) + [0] * (m - len(target))
    aa = list(a) + [0] * (m - len(a))
    bb = list(b) + [0] * (m - len(b))
    return all(t[i] >= aa[i] + bb[i] for i in range(m))


def _rowwise_violations(target: Sequence[int], a: Sequence[int], b: Sequence[int]) -> int:
    m = max(len(target), len(a), len(b))
    t = list(target) + [0] * (m - len(target))
    aa = list(a) + [0] * (m - len(a))
    bb = list(b) + [0] * (m - len(b))
    return int(sum(1 for i in range(m) if t[i] < aa[i] + bb[i]))


def _same_grass_descent(u: Sequence[int], v: Sequence[int]) -> bool:
    if not (KF.is_grassmannian(u) and KF.is_grassmannian(v)):
        return False
    du, dv = KF.grassmannian_descent(u), KF.grassmannian_descent(v)
    return du is not None and dv is not None and du == dv


# canonical feature ordering (stable column layout for the learners)
def feature_names() -> List[str]:
    e = P.identity(3)
    return list(predicate_features(e, e, e).keys())


def feature_matrix(triples: Sequence[Triple]) -> Tuple[List[str], np.ndarray]:
    """Stack predicate vectors -> ``(names, X[N, F])`` (float for sklearn)."""
    names = feature_names()
    X = np.array([[predicate_features(u, v, w)[k] for k in names] for (u, v, w) in triples],
                 dtype=np.float64)
    return names, X


# ---------------------------------------------------------------------------
# 14.3 Rule learners
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    """A learned symbolic rule predicting positivity.

    ``predict`` maps a predicate-feature *dict* to a bool.  ``description`` is the
    human-readable / DSL form for the conjecture card.  ``metrics`` holds the
    fit-time scores.
    """

    kind: str
    description: str
    predict: Callable[[Dict[str, int]], bool]
    feature_names: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)
    model: object = None

    def __call__(self, features: Dict[str, int]) -> bool:
        return bool(self.predict(features))


def _split(n: int, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    te, tr = perm[:n_test], perm[n_test:]
    if len(tr) < 2:
        tr = perm
    return tr, te


def _l1_logistic(C: float, seed: int = 0):
    """An L1-penalised logistic regressor, robust across sklearn versions.

    sklearn >= 1.8 deprecates ``penalty='l1'`` in favour of ``l1_ratio``; older
    versions lack ``l1_ratio`` on the saga path.  Try the new API, fall back.
    """
    from sklearn.linear_model import LogisticRegression
    try:  # new API (sklearn >= 1.8): pure L1 via l1_ratio=1.0 on saga
        return LogisticRegression(solver="saga", l1_ratio=1.0, C=C, max_iter=2000,
                                  random_state=seed)
    except TypeError:  # pragma: no cover - older sklearn
        return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=500,
                                  random_state=seed)


def fit_sparse_logistic(
    triples: Sequence[Triple],
    y: Sequence[int],
    *,
    C: float = 0.2,
    seed: int = 0,
    test_frac: float = 0.25,
) -> Rule:
    """Sparse (L1) logistic regression over predicates (PROJECT.md 14.3.1)."""
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    names, X = feature_matrix(triples)
    y = np.asarray(y, dtype=int)
    tr, te = _split(len(y), test_frac, seed)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    if len(np.unique(y[tr])) < 2:
        clf = None
        acc = float((y[te] == y[tr][0]).mean())
        coef = np.zeros(len(names))
        desc = f"const({int(y[tr][0])})"
    else:
        clf = _l1_logistic(C, seed)
        clf.fit(Xtr, y[tr])
        acc = float(accuracy_score(y[te], clf.predict(Xte)))
        coef = clf.coef_[0]
        nz = [(names[i], round(float(coef[i]), 3)) for i in np.argsort(-np.abs(coef))
              if abs(coef[i]) > 1e-6]
        desc = "sigmoid(" + " + ".join(f"{c}*z[{nm}]" for nm, c in nz[:8]) + ") > 0.5"

    mean = scaler.mean_
    scale = scaler.scale_

    def predict(feats: Dict[str, int]) -> bool:
        if clf is None:
            return bool(y[tr][0])
        x = np.array([[feats.get(k, 0) for k in names]], dtype=np.float64)
        x = (x - mean) / scale
        return bool(clf.predict(x)[0])

    return Rule("sparse_logistic", desc, predict, names,
                {"held_out_acc": round(acc, 4),
                 "nonzero_coefs": int(np.sum(np.abs(coef) > 1e-6))}, clf)


def fit_decision_tree(
    triples: Sequence[Triple],
    y: Sequence[int],
    *,
    max_depth: int = 4,
    seed: int = 0,
    test_frac: float = 0.25,
) -> Rule:
    """Shallow decision tree over predicates (PROJECT.md 14.3.2)."""
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    names, X = feature_matrix(triples)
    y = np.asarray(y, dtype=int)
    tr, te = _split(len(y), test_frac, seed)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X[tr], y[tr])
    acc = float(accuracy_score(y[te], clf.predict(X[te])))
    try:
        text = export_text(clf, feature_names=list(names))
    except Exception:
        text = f"DecisionTree(depth={max_depth})"

    def predict(feats: Dict[str, int]) -> bool:
        x = np.array([[feats.get(k, 0) for k in names]], dtype=np.float64)
        return bool(clf.predict(x)[0])

    return Rule("decision_tree", text.strip(), predict, names,
                {"held_out_acc": round(acc, 4), "max_depth": max_depth,
                 "n_leaves": int(clf.get_n_leaves())}, clf)


def fit_monotone_rule_list(
    triples: Sequence[Triple],
    y: Sequence[int],
    *,
    max_rules: int = 6,
    seed: int = 0,
    test_frac: float = 0.25,
    candidate_features: Optional[Sequence[str]] = None,
) -> Rule:
    """A simple greedy *monotone* rule list (PROJECT.md 14.3.3).

    Each rule is a single-feature threshold of the form ``z[feat] <= thr -> 0``
    (a sufficient condition for a hard zero) chosen greedily to peel off the
    purest negative slab; the default (fall-through) prediction is positive.
    Threshold candidates come from the feature's observed quantiles.  "Monotone"
    here means every rule fires in the same (``-> 0``) direction, giving a
    transparent obstruction list.
    """
    names, X = feature_matrix(triples)
    name_idx = {nm: i for i, nm in enumerate(names)}
    y = np.asarray(y, dtype=int)
    tr, te = _split(len(y), test_frac, seed)
    Xtr, ytr = X[tr], y[tr]

    feats_to_try = list(candidate_features) if candidate_features else names
    rules: List[Tuple[str, float]] = []
    remaining = np.ones(len(ytr), dtype=bool)

    for _ in range(max_rules):
        best = None  # (impurity_removed, feat, thr)
        for nm in feats_to_try:
            j = name_idx[nm]
            col = Xtr[remaining, j]
            if col.size == 0:
                continue
            for thr in np.unique(np.quantile(col, [0.1, 0.25, 0.5, 0.75])):
                fire = remaining & (X[tr][:, j] <= thr)
                if fire.sum() == 0:
                    continue
                neg_caught = int((ytr[fire] == 0).sum())
                pos_caught = int((ytr[fire] == 1).sum())
                # reward catching negatives, penalise catching positives
                score = neg_caught - 2 * pos_caught
                if score > 0 and (best is None or score > best[0]):
                    best = (score, nm, float(thr))
        if best is None:
            break
        _, nm, thr = best
        rules.append((nm, thr))
        j = name_idx[nm]
        remaining = remaining & ~(X[tr][:, j] <= thr)
        if remaining.sum() == 0:
            break

    def predict_arr(Xa: np.ndarray) -> np.ndarray:
        pred = np.ones(Xa.shape[0], dtype=int)  # default positive
        fired = np.zeros(Xa.shape[0], dtype=bool)
        for nm, thr in rules:
            j = name_idx[nm]
            cond = (~fired) & (Xa[:, j] <= thr)
            pred[cond] = 0
            fired |= cond
        return pred

    from sklearn.metrics import accuracy_score
    acc = float(accuracy_score(y[te], predict_arr(X[te]))) if len(te) else float("nan")
    if rules:
        desc = "IF " + " OR ".join(f"z[{nm}] <= {thr:g}" for nm, thr in rules) + \
               " THEN c=0 ELSE c>0"
    else:
        desc = "ALWAYS c>0"

    def predict(feats: Dict[str, int]) -> bool:
        for nm, thr in rules:
            if feats.get(nm, 0) <= thr:
                return False
        return True

    return Rule("monotone_rule_list", desc, predict, names,
                {"held_out_acc": round(acc, 4), "n_rules": len(rules),
                 "rules": [(nm, round(thr, 3)) for nm, thr in rules]}, rules)


def learn_positivity_rules(
    triples: Sequence[Triple],
    positive: Sequence[int],
    *,
    seed: int = 0,
    max_depth: int = 4,
) -> Dict[str, Rule]:
    """Fit all three learners; return ``{kind: Rule}`` for positivity ``c>0``."""
    return {
        "sparse_logistic": fit_sparse_logistic(triples, positive, seed=seed),
        "decision_tree": fit_decision_tree(triples, positive, max_depth=max_depth, seed=seed),
        "monotone_rule_list": fit_monotone_rule_list(triples, positive, seed=seed),
    }


# ---------------------------------------------------------------------------
# 14.4 Validation protocol
# ---------------------------------------------------------------------------
def validate_rule(
    rule_fn: Callable[[Dict[str, int]], bool],
    triples: Sequence[Triple],
    positive: Sequence[int],
    *,
    tiers: Optional[Sequence[object]] = None,
) -> Dict[str, object]:
    """Evaluate a rule's accuracy overall and stratified by negative tier.

    ``rule_fn(feature_dict) -> bool`` (True == predicts ``c>0``).  ``tiers`` is an
    optional per-example tier label (e.g. ``tier_id`` from the dataset) used to
    break out accuracy / false-positive / false-negative rates by tier.
    """
    y = np.asarray(positive, dtype=int)
    preds = np.array([1 if rule_fn(predicate_features(u, v, w)) else 0
                      for (u, v, w) in triples], dtype=int)
    out: Dict[str, object] = {
        "n": int(len(y)),
        "accuracy": float((preds == y).mean()) if len(y) else float("nan"),
        "precision": _safe_div(int(((preds == 1) & (y == 1)).sum()), int((preds == 1).sum())),
        "recall": _safe_div(int(((preds == 1) & (y == 1)).sum()), int((y == 1).sum())),
        "false_positive_rate": _safe_div(int(((preds == 1) & (y == 0)).sum()), int((y == 0).sum())),
    }
    if tiers is not None:
        tiers = np.asarray(list(tiers))
        by_tier: Dict[str, object] = {}
        for t in sorted(set(tiers.tolist())):
            m = tiers == t
            by_tier[str(t)] = {
                "n": int(m.sum()),
                "accuracy": float((preds[m] == y[m]).mean()) if m.any() else float("nan"),
            }
        out["by_tier"] = by_tier
    return out


def _safe_div(a: int, b: int) -> float:
    return float(a) / b if b else float("nan")


# ---------------------------------------------------------------------------
# 14.5 Conjecture card template
# ---------------------------------------------------------------------------
def write_conjecture_card(path: str, card: Dict[str, object]) -> str:
    """Render a conjecture card to markdown (PROJECT.md 14.5 template).

    Expected ``card`` keys (all optional, sensible defaults filled):
    ``id, statement, neural_origin, symbolic_form, empirical_support,
    relation_to_known_theory, minimal_examples, proof_strategy, status``.
    Returns the path written.
    """
    import os
    from datetime import datetime, timezone

    cid = card.get("id") or f"C-{datetime.now(timezone.utc):%Y%m%d}-01"

    def block(val) -> str:
        if isinstance(val, (dict, list)):
            import json
            return "```json\n" + json.dumps(val, indent=2, default=str) + "\n```"
        return str(val) if val is not None else "_TODO_"

    lines = [
        f"# Conjecture {cid}",
        "",
        "## Statement",
        block(card.get("statement")),
        "",
        "## Neural origin",
        block(card.get("neural_origin")),
        "",
        "## Symbolic form",
        block(card.get("symbolic_form")),
        "",
        "## Empirical support",
        block(card.get("empirical_support")),
        "",
        "## Relation to known theory",
        block(card.get("relation_to_known_theory")),
        "",
        "## Minimal examples",
        block(card.get("minimal_examples")),
        "",
        "## Proof strategy",
        block(card.get("proof_strategy")),
        "",
        "## Status",
        str(card.get("status", "observed")),
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
