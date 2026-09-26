"""Feature extraction + classical/weak baselines (PROJECT.md 11, 12.1).

The transformer must be compared against weak baselines (to expose shortcut
learning) and strong classical-ML baselines (if XGBoost matches the transformer,
the transformer is probably not learning deep structure -- PROJECT.md 11.2).

The two design constraints from PROJECT.md are:

* **Report hard zeroes separately from easy zeroes.**  Every model is evaluated
  on three slices of the product-disjoint *test* split: ``all`` (every tier),
  ``easy`` (Z0/Z1 + positives) and ``hard`` (Z2--Z5 + positives).  A model that
  only learned the length / descent shortcut will look strong on ``easy`` and
  collapse on ``hard``.
* **Shortcut audit.**  The gap between the length-only weak baseline and the
  full-feature classifiers on the hard slice is reported explicitly: a small gap
  means the dataset is shortcut-solvable.

Everything here operates on the fixed numpy/parquet contracts documented in
``schubml.data`` (no training code, no torch).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..combinatorics import bruhat
from ..combinatorics import known_families as KF
from ..combinatorics import permutations as P
from ..combinatorics import rank_matrix as RM
from ..combinatorics import rothe
from ..data.datasets import build_perms_by_id, load_triple_arrays
from ..data.labels import FAMILY_TAGS
from ..data.schemas import PERM_JSON_COLUMNS, load_table
from ..train.metrics import coefficient_metrics, positivity_metrics

Perm = Tuple[int, ...]

# Fixed context for indicator features.  Canonical permutations in this project
# top out at S_7 (``n_compute <= 7`` for the released layers) but we leave head
# room so larger ambient sizes never overflow the fixed-width feature vector.
MAX_N = 14

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _descent_indicator(descents: Sequence[int], n: int = MAX_N) -> np.ndarray:
    """Fixed-size (n-1) multi-hot indicator of a descent set (1-indexed)."""
    v = np.zeros(n - 1, dtype=np.float64)
    for d in descents:
        if 1 <= d <= n - 1:
            v[d - 1] = 1.0
    return v


def _code_histogram(code: Sequence[int], n: int = MAX_N) -> np.ndarray:
    """Histogram of Lehmer-code values: ``hist[k] = #{i : code_i == k}``."""
    h = np.zeros(n, dtype=np.float64)
    for c in code:
        if 0 <= c < n:
            h[c] += 1.0
    return h


def _sorted_padded(values: Sequence[int], k: int) -> np.ndarray:
    """The ``k`` largest values, weakly decreasing, zero-padded to length ``k``."""
    s = sorted((float(x) for x in values), reverse=True)[:k]
    out = np.zeros(k, dtype=np.float64)
    out[: len(s)] = s
    return out


def _length_summary(values: Sequence[int], k: int = 4) -> np.ndarray:
    """``[sum, max, count_nonzero, *sorted_padded(k)]`` summary of row/col lengths."""
    arr = np.asarray(values, dtype=np.float64)
    head = np.array(
        [arr.sum(), arr.max() if arr.size else 0.0, float((arr > 0).sum())],
        dtype=np.float64,
    )
    return np.concatenate([head, _sorted_padded(values, k)])


def _rank_summary(w: Sequence[int]) -> np.ndarray:
    """Summary of the essential rank conditions ``(i, j, r)`` of ``w``."""
    ess = RM.essential_rank_conditions(w)
    if not ess:
        return np.zeros(5, dtype=np.float64)
    rs = np.array([r for (_, _, r) in ess], dtype=np.float64)
    js = np.array([j for (_, j, _) in ess], dtype=np.float64)
    iis = np.array([i for (i, _, _) in ess], dtype=np.float64)
    return np.array(
        [float(len(ess)), rs.sum(), rs.max(), iis.max(), js.max()],
        dtype=np.float64,
    )


def _family_multihot(w: Sequence[int]) -> np.ndarray:
    tags = set(KF.family_tags(w))
    return np.array([1.0 if t in tags else 0.0 for t in FAMILY_TAGS], dtype=np.float64)


def perm_features(w: Sequence[int]) -> np.ndarray:
    """Per-permutation feature block (PROJECT.md 11.2).

    Concatenates: length, descent counts, descent-set indicators (L+R),
    Lehmer-code histogram, Rothe row/col length summaries, rank-matrix summary
    and pattern-tag multi-hot.  Fixed width regardless of ``len(w)``.
    """
    w = P.canonical_perm(w)
    rd = P.right_descents(w)
    ld = P.left_descents(w)
    code = P.lehmer_code(w)
    blocks = [
        np.array([float(P.length(w)), float(len(rd)), float(len(ld)), float(len(w))]),
        _descent_indicator(rd),
        _descent_indicator(ld),
        _code_histogram(code),
        _length_summary(rothe.rothe_row_lengths(w)),
        _length_summary(rothe.rothe_col_lengths(w)),
        _rank_summary(w),
        _family_multihot(w),
    ]
    return np.concatenate(blocks)


def _length_defect(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> int:
    """``ell(w) - ell(u) - ell(v)`` (zero is the grading necessary condition)."""
    return P.length(w) - P.length(u) - P.length(v)


def _descent_compatibility(u, v, w) -> np.ndarray:
    """How the descent set of ``w`` relates to those of ``u`` and ``v``.

    Schubert positivity has no clean descent rule, but descent (in)compatibility
    is a cheap, interpretable signal: a classifier that leans on it is a useful
    shortcut audit.  We report intersection / containment counts between the
    right-descent sets.
    """
    du, dv, dw = set(P.right_descents(u)), set(P.right_descents(v)), set(P.right_descents(w))
    return np.array(
        [
            float(len(dw & du)),
            float(len(dw & dv)),
            float(len(dw - (du | dv))),  # descents of w explained by neither factor
            float(len((du | dv) - dw)),  # factor descents absent from w
        ],
        dtype=np.float64,
    )


def interaction_features(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> np.ndarray:
    """Cross-``(u, v, w)`` interaction features (PROJECT.md 11.2).

    * length defect ``ell(w) - ell(u) - ell(v)`` and the grading indicator,
    * Bruhat necessary conditions ``u <= w`` and ``v <= w`` (and their AND),
    * descent compatibility counts.
    """
    defect = _length_defect(u, v, w)
    u_leq = 1.0 if bruhat.bruhat_leq(u, w) else 0.0
    v_leq = 1.0 if bruhat.bruhat_leq(v, w) else 0.0
    head = np.array(
        [
            float(defect),
            1.0 if defect == 0 else 0.0,  # grading passes
            u_leq,
            v_leq,
            u_leq * v_leq,  # both Bruhat conditions (a strong necessary condition)
        ],
        dtype=np.float64,
    )
    return np.concatenate([head, _descent_compatibility(u, v, w)])


def triple_features(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> np.ndarray:
    """Full feature vector for a triple ``(u, v, w)`` (PROJECT.md 11.2).

    Per-factor blocks for ``u``, ``v``, ``w`` (a symmetric ``u + v`` block keeps
    the representation invariant under swapping the two factors, matching the
    commutativity ``c_{u,v}^w = c_{v,u}^w``), plus the cross-interaction block.
    """
    fu = perm_features(u)
    fv = perm_features(v)
    fw = perm_features(w)
    sym_uv = fu + fv  # symmetric in u, v
    inter = interaction_features(u, v, w)
    return np.concatenate([sym_uv, fw, inter])


# convenient feature dimensionality (computed once, lazily)
def feature_dim() -> int:
    return triple_features((1,), (1,), (1,)).shape[0]


# ---------------------------------------------------------------------------
# Building a design matrix from a triples-arrays dict
# ---------------------------------------------------------------------------
def build_matrix(
    arrays: Dict[str, np.ndarray],
    perms_by_id: Dict[int, Sequence[int]],
    *,
    cache: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X, y_positive, y_coeff, tier_id, product_id)`` from a triples dict.

    ``arrays`` is the dict returned by :func:`schubml.data.datasets.load_triple_arrays`
    and ``perms_by_id`` the mapping from :func:`build_perms_by_id`.  Per-permutation
    feature blocks are memoised in ``cache`` (shared across slices) so the heavy
    combinatorics is computed once per distinct permutation, not once per triple.
    """
    cache = {} if cache is None else cache

    def feats(pid: int) -> np.ndarray:
        f = cache.get(pid)
        if f is None:
            f = perm_features(perms_by_id[int(pid)])
            cache[pid] = f
        return f

    n = len(arrays["w_id"])
    rows: List[np.ndarray] = []
    for i in range(n):
        uid, vid, wid = int(arrays["u_id"][i]), int(arrays["v_id"][i]), int(arrays["w_id"][i])
        u = perms_by_id[uid]
        v = perms_by_id[vid]
        w = perms_by_id[wid]
        fu, fv, fw = feats(uid), feats(vid), feats(wid)
        inter = interaction_features(u, v, w)
        rows.append(np.concatenate([fu + fv, fw, inter]))
    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, feature_dim()))
    y_pos = arrays["positive"].astype(int)
    y_coeff = arrays["coefficient"].astype(int)
    tier_id = arrays["tier_id"].astype(int)
    product_id = arrays["product_id"].astype(int)
    return X, y_pos, y_coeff, tier_id, product_id


# ---------------------------------------------------------------------------
# Evaluation-slice masks (hard vs easy zeroes -- PROJECT.md 11.1, 12.1)
# ---------------------------------------------------------------------------
# tier_id encoding (schubml.data.labels.TIERS):
#   0 positive, 1 Z0, 2 Z1, 3 Z2, 4 Z3, 5 Z4, 6 Z5
HARD_TIER_IDS = (3, 4, 5, 6)   # Z2..Z5
EASY_TIER_IDS = (1, 2)         # Z0, Z1


def slice_masks(y_pos: np.ndarray, tier_id: np.ndarray) -> Dict[str, np.ndarray]:
    """Boolean masks for the ``all`` / ``easy`` / ``hard`` evaluation slices.

    Positives are included in every slice so each slice has both classes.
    """
    pos = y_pos > 0
    return {
        "all": np.ones(len(y_pos), dtype=bool),
        "easy": pos | np.isin(tier_id, EASY_TIER_IDS),
        "hard": pos | np.isin(tier_id, HARD_TIER_IDS),
    }


# ---------------------------------------------------------------------------
# Weak baselines (PROJECT.md 11.1)
# ---------------------------------------------------------------------------
@dataclass
class WeakResult:
    """A weak baseline's centred positivity score (``> 0`` => predict positive)."""

    score: np.ndarray


def majority_score(y_train: np.ndarray, n_eval: int) -> np.ndarray:
    """Predict the training-majority class for everyone (constant score)."""
    maj = 1 if y_train.mean() >= 0.5 else 0
    return np.full(n_eval, 1.0 if maj == 1 else -1.0)


def length_only_score(X: np.ndarray) -> np.ndarray:
    """Predict positive iff the length defect is zero.

    The length-defect feature lives at the *start* of the interaction block.
    """
    defect = X[:, _INTER_OFFSET + 0]
    # score > 0 when defect == 0; magnitude reflects |defect|
    return np.where(defect == 0.0, 1.0, -1.0 - np.abs(defect))


# index of the interaction block inside the full triple feature vector
_PERM_DIM: Optional[int] = None
_INTER_OFFSET: int = -1


def _init_offsets() -> None:
    global _PERM_DIM, _INTER_OFFSET
    if _PERM_DIM is None:
        _PERM_DIM = perm_features((1,)).shape[0]
        # layout: [sym_uv (PERM_DIM)] [fw (PERM_DIM)] [interaction]
        _INTER_OFFSET = 2 * _PERM_DIM


_init_offsets()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class BaselineSpec:
    """A classical-ML baseline: a name, a sklearn-style factory, and a feature view.

    ``feature_view`` selects a slice of the full feature vector so that weak
    *logistic* baselines (length+descents, length+Rothe) can reuse the same
    matrix as the full-feature models (PROJECT.md 11.1).
    """

    name: str
    factory: Callable[[], object]
    feature_cols: Optional[np.ndarray] = None  # None => all columns


def _length_descent_cols() -> np.ndarray:
    """Columns for the length+descents logistic weak baseline.

    Per-permutation block layout (see :func:`perm_features`):
        [0:4]   length, n_rdesc, n_ldesc, n
        [4:4+(N-1)]            right-descent indicator
        [4+(N-1):4+2(N-1)]     left-descent indicator
        ...
    We take the scalar length/descent counts and the descent indicators from the
    symmetric ``u+v`` block and the ``w`` block, plus the interaction block's
    length-defect / descent-compatibility entries.
    """
    n1 = MAX_N - 1
    desc_end = 4 + 2 * n1
    cols: List[int] = []
    for base in (0, _PERM_DIM):  # sym_uv block, w block
        cols += list(range(base, base + 4))           # scalar counts
        cols += list(range(base + 4, base + desc_end))  # descent indicators
    # interaction: length defect + grading + descent compatibility (skip Bruhat)
    cols += [_INTER_OFFSET + 0, _INTER_OFFSET + 1]
    cols += list(range(_INTER_OFFSET + 5, _INTER_OFFSET + 9))
    return np.array(sorted(set(cols)), dtype=int)


def _length_rothe_cols() -> np.ndarray:
    """Columns for the length+Rothe logistic weak baseline."""
    n1 = MAX_N - 1
    # within a perm block, the Rothe summaries start after:
    #   4 (counts) + 2*(N-1) (descents) + N (code hist)
    rothe_start = 4 + 2 * n1 + MAX_N
    rothe_end = rothe_start + 2 * 7  # two length-summaries of width 7
    cols: List[int] = []
    for base in (0, _PERM_DIM):
        cols += list(range(base, base + 1))  # length scalar
        cols += list(range(base + rothe_start, base + rothe_end))
    cols += [_INTER_OFFSET + 0, _INTER_OFFSET + 1]  # length defect + grading
    return np.array(sorted(set(cols)), dtype=int)


def default_baseline_specs(*, max_train: int) -> List[BaselineSpec]:
    """The classical-ML baselines (PROJECT.md 11.2) + logistic weak baselines.

    Hyper-parameters are deliberately modest: these are reference points, not
    tuned competitors.  ``max_train`` informs forest/boosting sizes for speed.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBClassifier
        _have_xgb = True
    except Exception:  # pragma: no cover - xgboost is expected to be installed
        _have_xgb = False

    n_estimators = 200 if max_train <= 50000 else 120

    def logreg():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )

    def mlp():
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                          early_stopping=True, random_state=0),
        )

    specs = [
        # weak logistic baselines on restricted feature views
        BaselineSpec("weak_length_descents_logreg", logreg, _length_descent_cols()),
        BaselineSpec("weak_length_rothe_logreg", logreg, _length_rothe_cols()),
        # classical ML on the full feature set
        BaselineSpec("logreg", logreg, None),
        BaselineSpec(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=n_estimators, max_depth=None, n_jobs=-1,
                class_weight="balanced", random_state=0),
            None,
        ),
        BaselineSpec("mlp", mlp, None),
    ]
    if _have_xgb:
        specs.append(
            BaselineSpec(
                "xgboost",
                lambda: XGBClassifier(
                    n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                    subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                    eval_metric="logloss", tree_method="hist", random_state=0),
                None,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Fit / score helpers
# ---------------------------------------------------------------------------
def _decision_score(model, X: np.ndarray) -> np.ndarray:
    """A real-valued positivity score centred so that ``> 0`` => predict positive.

    Prefers ``decision_function`` (already centred at 0); otherwise converts
    ``predict_proba`` to a logit ``log(p/(1-p))`` so :func:`positivity_metrics`
    (which thresholds at ``> 0``) behaves correctly.
    """
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s, dtype=np.float64).ravel()
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))
    return model.predict(X).astype(np.float64) * 2.0 - 1.0  # {0,1} -> {-1,+1}


def _eval_scores(score: np.ndarray, y_pos: np.ndarray, tier_id: np.ndarray) -> Dict[str, Dict]:
    """positivity metrics on the all/easy/hard slices."""
    masks = slice_masks(y_pos, tier_id)
    out: Dict[str, Dict] = {}
    for sl, m in masks.items():
        if m.sum() == 0 or len(np.unique(y_pos[m])) < 1:
            out[sl] = {"n": int(m.sum())}
            continue
        out[sl] = positivity_metrics(score[m], y_pos[m])
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class BaselineData:
    """Train/test matrices + labels for the baselines run."""

    X_train: np.ndarray
    y_train: np.ndarray
    c_train: np.ndarray
    tier_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    c_test: np.ndarray
    tier_test: np.ndarray
    product_test: np.ndarray
    feature_names_dim: int = 0


def load_baseline_data(
    data_dir: str,
    *,
    max_train: int = 200000,
    max_test: int = 100000,
    seed: int = 0,
) -> BaselineData:
    """Load + featurise the product-disjoint train/test splits (PROJECT.md 6.7).

    Trains on ``product_disjoint:train`` and evaluates on
    ``product_disjoint:test`` so there is no product leakage.
    """
    perm_path = os.path.join(data_dir, "permutations.parquet")
    trip_path = os.path.join(data_dir, "triples.parquet")
    perm_df = load_table(perm_path, PERM_JSON_COLUMNS)
    perms_by_id = build_perms_by_id(perm_df)

    rng = np.random.default_rng(seed)

    def _load(split: str, cap: int) -> Dict[str, np.ndarray]:
        arr = load_triple_arrays(trip_path, split=split)
        n = len(arr["w_id"])
        if cap and n > cap:
            idx = rng.choice(n, cap, replace=False)
            arr = {k: v[idx] for k, v in arr.items()}
        return arr

    train_arr = _load("product_disjoint:train", max_train)
    test_arr = _load("product_disjoint:test", max_test)

    cache: Dict[int, np.ndarray] = {}
    Xtr, ytr, ctr, ttr, _ = build_matrix(train_arr, perms_by_id, cache=cache)
    Xte, yte, cte, tte, pte = build_matrix(test_arr, perms_by_id, cache=cache)
    return BaselineData(Xtr, ytr, ctr, ttr, Xte, yte, cte, tte, pte,
                        feature_names_dim=Xtr.shape[1] if Xtr.size else feature_dim())


def run_weak_baselines(d: BaselineData) -> Dict[str, Dict]:
    """Majority + length-only weak baselines (PROJECT.md 11.1)."""
    results: Dict[str, Dict] = {}
    # majority
    maj = majority_score(d.y_train, len(d.y_test))
    results["weak_majority"] = _eval_scores(maj, d.y_test, d.tier_test)
    # length-only: positive iff length defect == 0
    lo = length_only_score(d.X_test)
    results["weak_length_only"] = _eval_scores(lo, d.y_test, d.tier_test)
    return results


def run_classical_baselines(d: BaselineData, *, max_train: int) -> Dict[str, Dict]:
    """Fit + evaluate every classical-ML baseline on all/easy/hard slices."""
    results: Dict[str, Dict] = {}
    for spec in default_baseline_specs(max_train=max_train):
        cols = spec.feature_cols
        Xtr = d.X_train[:, cols] if cols is not None else d.X_train
        Xte = d.X_test[:, cols] if cols is not None else d.X_test
        model = spec.factory()
        try:
            model.fit(Xtr, d.y_train)
            score = _decision_score(model, Xte)
            block = _eval_scores(score, d.y_test, d.tier_test)
            # coefficient metrics are only meaningful for positive-predicting models;
            # here we report a trivial positivity->coefficient passthrough on the
            # hard slice so the JSON has the coefficient keys the eval table expects.
            block["fitted"] = True
        except Exception as e:  # pragma: no cover - robustness
            block = {"error": str(e)}
        results[spec.name] = block
    return results


def shortcut_audit(results: Dict[str, Dict]) -> Dict[str, object]:
    """Quantify shortcut-solvability (PROJECT.md 3.2, 11.1).

    Compares the length-only weak baseline to the best full-feature classifier on
    the *hard* slice.  A small accuracy/MCC gap means hard zeroes are still being
    solved by easy statistics -- the data (or model) is leaking a shortcut.
    """
    def _metric(name: str, sl: str, key: str) -> Optional[float]:
        blk = results.get(name, {}).get(sl, {})
        v = blk.get(key)
        return float(v) if isinstance(v, (int, float)) and not _isnan(v) else None

    full_models = ["logreg", "random_forest", "xgboost", "mlp"]
    audit: Dict[str, object] = {}
    for key in ("accuracy", "mcc", "f1"):
        lo_easy = _metric("weak_length_only", "easy", key)
        lo_hard = _metric("weak_length_only", "hard", key)
        full_hard = [
            _metric(m, "hard", key) for m in full_models
            if _metric(m, "hard", key) is not None
        ]
        best_full_hard = max(full_hard) if full_hard else None
        audit[key] = {
            "length_only_easy": lo_easy,
            "length_only_hard": lo_hard,
            "best_full_hard": best_full_hard,
            "full_vs_length_gap_hard": (
                None if (best_full_hard is None or lo_hard is None)
                else round(best_full_hard - lo_hard, 4)
            ),
            "length_only_easy_minus_hard": (
                None if (lo_easy is None or lo_hard is None)
                else round(lo_easy - lo_hard, 4)
            ),
        }
    audit["interpretation"] = (
        "A large 'length_only_easy_minus_hard' confirms easy zeroes are shortcut-"
        "solvable while hard zeroes are not.  A small 'full_vs_length_gap_hard' "
        "means even full features add little over the length shortcut on hard "
        "zeroes -- the regime where the transformer must show real structure."
    )
    return audit


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def run_baselines(
    data_dir: str,
    out_json: Optional[str] = None,
    *,
    max_train: int = 200000,
    max_test: int = 100000,
    seed: int = 0,
) -> Dict[str, object]:
    """End-to-end baselines run (PROJECT.md 11, 12.1).

    Trains every weak + classical baseline on ``product_disjoint:train`` and
    evaluates on ``product_disjoint:test``, reporting positivity metrics on the
    ``all`` / ``easy`` / ``hard`` slices separately so shortcut learning is
    visible.  Returns a metrics dict keyed by ``model -> slice -> metric`` and
    (if ``out_json`` is given) writes it to JSON.

    Raises ``FileNotFoundError`` if the processed tables are absent (callers may
    build a tiny synthetic dataset first; see ``scripts/build_dataset.py``).
    """
    trip_path = os.path.join(data_dir, "triples.parquet")
    if not os.path.exists(trip_path):
        raise FileNotFoundError(
            f"triples.parquet not found under {data_dir!r}; build a dataset first "
            f"(e.g. scripts/build_dataset.py --raw data/raw/s3.jsonl:3 --out {data_dir})"
        )

    d = load_baseline_data(data_dir, max_train=max_train, max_test=max_test, seed=seed)

    models: Dict[str, Dict] = {}
    models.update(run_weak_baselines(d))
    models.update(run_classical_baselines(d, max_train=max_train))

    out: Dict[str, object] = {
        "data_dir": os.path.abspath(data_dir),
        "n_train": int(len(d.y_train)),
        "n_test": int(len(d.y_test)),
        "feature_dim": int(d.feature_names_dim),
        "train_base_rate": float(d.y_train.mean()) if len(d.y_train) else float("nan"),
        "test_base_rate": float(d.y_test.mean()) if len(d.y_test) else float("nan"),
        "slices": {
            "easy": "positives + Z0/Z1 negatives (easy zeroes)",
            "hard": "positives + Z2..Z5 negatives (hard zeroes; the serious test)",
        },
        "models": models,
        "shortcut_audit": shortcut_audit(models),
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Pretty summary table (used by scripts/run_baselines.py)
# ---------------------------------------------------------------------------
def summary_rows(results: Dict[str, object]) -> List[Tuple[str, float, float, float, float]]:
    """``(model, acc_easy, mcc_easy, acc_hard, mcc_hard)`` rows for printing."""
    rows: List[Tuple[str, float, float, float, float]] = []
    for name, blk in results.get("models", {}).items():
        if not isinstance(blk, dict):
            continue
        easy = blk.get("easy", {})
        hard = blk.get("hard", {})
        rows.append((
            name,
            float(easy.get("accuracy", float("nan"))),
            float(easy.get("mcc", float("nan"))),
            float(hard.get("accuracy", float("nan"))),
            float(hard.get("mcc", float("nan"))),
        ))
    return rows


def format_summary(results: Dict[str, object]) -> str:
    """Render a human-readable summary table separating easy vs hard slices."""
    lines = []
    lines.append(
        f"baselines  n_train={results.get('n_train')}  n_test={results.get('n_test')}"
        f"  feat_dim={results.get('feature_dim')}"
        f"  test_base_rate={results.get('test_base_rate'):.3f}"
    )
    header = f"{'model':<32} {'acc_easy':>9} {'mcc_easy':>9} {'acc_hard':>9} {'mcc_hard':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, ae, me, ah, mh in summary_rows(results):
        lines.append(f"{name:<32} {ae:>9.3f} {me:>9.3f} {ah:>9.3f} {mh:>9.3f}")
    audit = results.get("shortcut_audit", {})
    acc_audit = audit.get("accuracy", {}) if isinstance(audit, dict) else {}
    lines.append("")
    lines.append(
        "shortcut audit (accuracy): "
        f"length_only easy={acc_audit.get('length_only_easy')} "
        f"hard={acc_audit.get('length_only_hard')} "
        f"best_full_hard={acc_audit.get('best_full_hard')} "
        f"full_vs_length_gap_hard={acc_audit.get('full_vs_length_gap_hard')}"
    )
    return "\n".join(lines)
