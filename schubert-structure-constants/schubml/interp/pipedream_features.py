"""Pipe-dream / Newton-polytope / Bruhat-chain feature battery for the residual gap.

Rank-matrix features cap at balanced-acc 0.74 on the v003 residual zeros while the
trained model reaches 0.96; the missing structure is conjectured to be monomial /
chain theoretic.  This module supplies:

1. **Pipe dream (RC-graph) enumeration** for a permutation ``w`` via ladder-move
   BFS from the bottom pipe dream (Bergeron--Billey: every RC-graph of ``w`` is
   reachable from ``D_bot(w)`` by ladder moves), with a hard cap (``capped`` flag).
   The x-weight multiset of the dreams is exactly the monomial expansion of the
   Schubert polynomial ``S_w`` (Billey--Jockusch--Stanley), so the *distinct*
   weights are ``supp(S_w)`` and — by the Saturated Newton Polytope theorem of
   Fink--Meszaros--St.Dizier (2018) — the lattice points of ``Newton(S_w)``.

2. **Triple features** for ``(u, v, w)``:

   * ``frac_w_monomials_covered``: fraction of (checked) monomials of ``S_w``
     lying in the Minkowski sum ``supp(S_u) + supp(S_v)``.  Because all Schubert
     coefficients and all ``c_{u,v}^{w'}`` are nonnegative,
     ``supp(S_u S_v) = supp(S_u) + supp(S_v)`` (no cancellation) and
     ``c_{u,v}^w > 0  =>  supp(S_w) \\subseteq supp(S_u) + supp(S_v)``.
     Hence ``frac < 1`` is a *theorem-grade certificate* that ``c_{u,v}^w = 0``;
     observing ``frac < 1`` on a ground-truth positive means the enumeration or
     Minkowski logic is buggy (the acceptance test).
     Soundness requires ``supp(S_u)`` and ``supp(S_v)`` to be complete (uncapped);
     ``supp(S_w)`` may be partially enumerated / sampled (checking a subset of a
     necessary condition stays sound).
   * newton-gap features: number of Minkowski decompositions of the bottom
     monomial ``x^{code(w)}`` and of the lex-max monomial of ``S_w``.
   * ``pd_count_ratio``: ``log(n_pd(u) * n_pd(v) / n_pd(w))`` (capped counts).
   * Bruhat-chain features: cover counts of ``u`` (and ``v``) staying inside
     ``[., w]``, depth-2 interval widths, and co-cover counts of ``w`` above
     ``u`` / ``v``.
   * Lascoux--Schuetzenberger transition-tree stats of ``w`` minus those of
     ``u, v`` (budget-guarded, globally memoized).

All permutations are 1-indexed one-line tuples (project convention); pipe dreams
are enumerated at the canonical (trimmed) size — Schubert polynomials are stable
under the inclusion ``S_n -> S_{n+1}``, so this loses nothing — and exponent
vectors are zero-padded to the common triple size ``N``.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import numpy as np

from ..combinatorics import permutations as P
from ..combinatorics import rank_matrix as RM
from ..combinatorics.bruhat import cover_transpositions_up
from ..combinatorics.permutations import (
    Perm,
    canonical_perm,
    compose,
    inverse,
    lehmer_code,
    length,
    longest_element,
    pad,
    right_mult_t,
)
from ..combinatorics.reduced_words import canonical_reduced_word, word_to_perm
from ..combinatorics.transition import transition_children

Cell = Tuple[int, int]
Dream = FrozenSet[Cell]

DEFAULT_PD_CAP = 5000


# ---------------------------------------------------------------------------
# Pipe dream enumeration (RC-graphs)
# ---------------------------------------------------------------------------
def bottom_pipe_dream(w: Sequence[int]) -> Dream:
    """Bottom RC-graph ``D_bot(w)``: row ``i`` has crosses in columns ``1..c_i``
    where ``c = lehmer_code(w)`` (left-justified rows)."""
    code = lehmer_code(w)
    return frozenset((i + 1, j) for i, c in enumerate(code) for j in range(1, c + 1))


def _ladder_moves(D: Dream) -> List[Dream]:
    """All single ladder moves applicable to ``D`` (Bergeron--Billey).

    A ladder move takes a cross at ``(i, j)`` with ``(i, j+1)`` empty, walks up
    rows ``r = i-1, i-2, ...`` while both ``(r, j)`` and ``(r, j+1)`` are crosses
    ("rungs"), and — if the first non-rung row ``k`` has *both* ``(k, j)`` and
    ``(k, j+1)`` empty — moves the cross from ``(i, j)`` to ``(k, j+1)``.
    Each cross admits at most one ladder move.
    """
    out: List[Dream] = []
    for (i, j) in D:
        if (i, j + 1) in D:
            continue
        k = i - 1
        while k >= 1 and (k, j) in D and (k, j + 1) in D:
            k -= 1
        if k < 1:
            continue
        if (k, j) in D or (k, j + 1) in D:
            continue
        out.append((D - {(i, j)}) | {(k, j + 1)})
    return out


def pipe_dreams(w: Sequence[int], cap: int = DEFAULT_PD_CAP) -> Tuple[List[Dream], bool]:
    """Reduced pipe dreams of ``w`` via ladder-move BFS from ``D_bot(w)``.

    Returns ``(dreams, capped)``.  If ``capped`` the list is a *strict subset*
    of RC(w) with exactly ``cap`` elements (BFS order from the bottom dream).
    """
    w = canonical_perm(w)
    bot = bottom_pipe_dream(w)
    seen: Set[Dream] = {bot}
    frontier: List[Dream] = [bot]
    while frontier:
        nxt: List[Dream] = []
        for D in frontier:
            for E in _ladder_moves(D):
                if E not in seen:
                    seen.add(E)
                    if len(seen) >= cap:
                        return list(seen)[:cap], True
                    nxt.append(E)
        frontier = nxt
    return list(seen), False


def dream_weight(D: Dream, n: int) -> Tuple[int, ...]:
    """x-weight of a pipe dream: ``weight[i-1]`` = number of crosses in row ``i``."""
    wt = [0] * n
    for (i, _j) in D:
        wt[i - 1] += 1
    return tuple(wt)


def n_pipe_dreams(w: Sequence[int], cap: int = DEFAULT_PD_CAP) -> Tuple[int, bool]:
    """Number of reduced pipe dreams of ``w`` (= ``S_w(1,...,1)``), capped."""
    dreams, capped = pipe_dreams(w, cap)
    return len(dreams), capped


def monomial_support(w: Sequence[int], cap: int = DEFAULT_PD_CAP
                     ) -> Tuple[np.ndarray, int, bool]:
    """``(supp, n_dreams, capped)`` for ``S_w`` at canonical size.

    ``supp`` is an ``(m, n)`` int array of distinct exponent vectors (the lattice
    points of ``Newton(S_w)`` by SNP when uncapped); ``n_dreams`` counts dreams
    with multiplicity.
    """
    w = canonical_perm(w)
    n = len(w)
    dreams, capped = pipe_dreams(w, cap)
    wts = {dream_weight(D, n) for D in dreams}
    supp = np.array(sorted(wts), dtype=np.int64).reshape(len(wts), n)
    return supp, len(dreams), capped


# ---------------------------------------------------------------------------
# Brute-force + divided-difference cross-checks (validation only)
# ---------------------------------------------------------------------------
def dream_word(D: Dream) -> List[int]:
    """Reading word of a dream: rows top->bottom, within a row right->left,
    a cross at ``(i, j)`` contributing the letter ``i + j - 1``."""
    word: List[int] = []
    for (i, j) in sorted(D, key=lambda c: (c[0], -c[1])):
        word.append(i + j - 1)
    return word


def pipe_dreams_brute(w: Sequence[int]) -> Set[Dream]:
    """All pipe dreams of ``w`` by exhaustive search over staircase subsets of
    size ``length(w)`` whose reading word is a reduced word for ``w``.
    Exponential; validation of :func:`pipe_dreams` on tiny cases only."""
    from itertools import combinations

    w = canonical_perm(w)
    n = len(w)
    l = length(w)
    cells = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i + j <= n]
    out: Set[Dream] = set()
    for sub in combinations(cells, l):
        D = frozenset(sub)
        if word_to_perm(dream_word(D), n) == w:
            out.add(D)
    return out


def _divided_difference(poly: Dict[Tuple[int, ...], int], i: int
                        ) -> Dict[Tuple[int, ...], int]:
    """Exact ``partial_i`` on an integer exponent-dict polynomial (1-indexed i)."""
    out: Dict[Tuple[int, ...], int] = {}
    for exp, c in poly.items():
        p, q = exp[i - 1], exp[i]
        if p == q:
            continue
        sgn, lo, hi = (1, q, p) if p > q else (-1, p, q)
        for t in range(lo, hi):
            e = list(exp)
            e[i - 1], e[i] = t, lo + hi - 1 - t
            e = tuple(e)
            out[e] = out.get(e, 0) + sgn * c
    return {e: c for e, c in out.items() if c != 0}


def schubert_poly_dd(w: Sequence[int]) -> Dict[Tuple[int, ...], int]:
    """Exact ``S_w`` via divided differences: ``S_w = \\partial_{w^{-1} w_0} x^delta``.

    Pure integer dicts (no sympy).  Cross-check oracle for small cases.
    """
    w = canonical_perm(w)
    n = len(w)
    poly: Dict[Tuple[int, ...], int] = {tuple(range(n - 1, -1, -1)): 1}
    word = canonical_reduced_word(compose(inverse(w), longest_element(n)))
    # word gives w^{-1} w_0 = s_{a_1} ... s_{a_k}; apply partials from the last
    # letter to the first (each step: S_{x s_i} = partial_i S_x for descents i).
    for a in reversed(word):
        poly = _divided_difference(poly, a)
    return poly


def schubert_poly_from_dreams(w: Sequence[int]) -> Dict[Tuple[int, ...], int]:
    """``S_w`` as an exponent dict from (uncapped) pipe dream enumeration."""
    w = canonical_perm(w)
    n = len(w)
    dreams, capped = pipe_dreams(w, cap=10 ** 9)
    assert not capped
    out: Dict[Tuple[int, ...], int] = {}
    for D in dreams:
        e = dream_weight(D, n)
        out[e] = out.get(e, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Support memoization
# ---------------------------------------------------------------------------
_SUPP_MEMO: Dict[Tuple[Perm, int], Tuple[np.ndarray, int, bool]] = {}
_SUPP_MEMO_MAX = 200_000


def support_cached(w: Sequence[int], cap: int = DEFAULT_PD_CAP
                   ) -> Tuple[np.ndarray, int, bool]:
    """Memoized :func:`monomial_support` keyed by (canonical perm, cap)."""
    key = (canonical_perm(w), cap)
    hit = _SUPP_MEMO.get(key)
    if hit is None:
        if len(_SUPP_MEMO) > _SUPP_MEMO_MAX:
            _SUPP_MEMO.clear()
        hit = monomial_support(key[0], cap)
        _SUPP_MEMO[key] = hit
    return hit


def _pad_cols(supp: np.ndarray, N: int) -> np.ndarray:
    n = supp.shape[1]
    if n == N:
        return supp
    if n > N:
        assert not supp[:, N:].any(), "support exceeds padded size"
        return supp[:, :N]
    out = np.zeros((supp.shape[0], N), dtype=supp.dtype)
    out[:, :n] = supp
    return out


# ---------------------------------------------------------------------------
# Minkowski coverage (meet-in-middle set lookup)
# ---------------------------------------------------------------------------
def _n_decompositions(m: np.ndarray, small: np.ndarray, big_set: Set[Tuple[int, ...]],
                      count_all: bool) -> int:
    """#{(a, b) : a in small, b in big, a + b = m} (0/1 if ``count_all`` False).

    Iterates the smaller support and looks the complement up in the bigger
    support's hash set.
    """
    if small.shape[0] == 0:
        return 0
    cand = small[(small <= m).all(axis=1)]
    cnt = 0
    for a in cand:
        if tuple((m - a).tolist()) in big_set:
            cnt += 1
            if not count_all:
                return 1
    return cnt


def coverage_features(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                      cap: int = DEFAULT_PD_CAP, max_w_monomials: int = 160,
                      seed: int = 0) -> Dict[str, float]:
    """Newton-polytope / Minkowski-sum features of the triple.

    ``cover_sound = 1`` iff supp(u) and supp(v) are complete, in which case any
    uncovered monomial of ``S_w`` is a *certificate* of ``c_{u,v}^w = 0``.
    """
    uc, vc, wc = canonical_perm(u), canonical_perm(v), canonical_perm(w)
    N = max(len(uc), len(vc), len(wc))
    supp_u, npd_u, cap_u = support_cached(uc, cap)
    supp_v, npd_v, cap_v = support_cached(vc, cap)
    supp_w, npd_w, cap_w = support_cached(wc, cap)
    Su, Sv, Sw = (_pad_cols(supp_u, N), _pad_cols(supp_v, N), _pad_cols(supp_w, N))

    small, big = (Su, Sv) if Su.shape[0] <= Sv.shape[0] else (Sv, Su)
    big_set = {tuple(r.tolist()) for r in big}

    # which monomials of S_w to check: bottom (code(w)), lex-max, lex-min + sample
    code_w = np.zeros(N, dtype=np.int64)
    cw = lehmer_code(wc)
    code_w[: len(cw)] = cw
    order = np.lexsort(Sw.T[::-1])          # lex order on rows
    lexmin, lexmax = Sw[order[0]], Sw[order[-1]]
    rng = np.random.default_rng(seed)
    if Sw.shape[0] > max_w_monomials:
        idx = rng.choice(Sw.shape[0], size=max_w_monomials, replace=False)
        checked = Sw[idx]
        full_check = False
    else:
        checked = Sw
        full_check = True

    n_cov = 0
    for m in checked:
        if _n_decompositions(m, small, big_set, count_all=False):
            n_cov += 1
    bottom_n = _n_decompositions(code_w, small, big_set, count_all=True)
    lexmax_n = _n_decompositions(lexmax, small, big_set, count_all=True)
    lexmin_n = _n_decompositions(lexmin, small, big_set, count_all=True)

    f: Dict[str, float] = {}
    f["n_pd_u"] = float(npd_u)
    f["n_pd_v"] = float(npd_v)
    f["n_pd_w"] = float(npd_w)
    f["pd_capped_any"] = float(cap_u or cap_v or cap_w)
    f["pd_count_ratio"] = math.log(max(npd_u, 1)) + math.log(max(npd_v, 1)) \
        - math.log(max(npd_w, 1))
    f["n_supp_u"] = float(Su.shape[0])
    f["n_supp_v"] = float(Sv.shape[0])
    f["n_supp_w"] = float(Sw.shape[0])
    f["supp_log_ratio"] = math.log(Su.shape[0]) + math.log(Sv.shape[0]) \
        - math.log(Sw.shape[0])
    f["supp_density_w"] = Sw.shape[0] / max(npd_w, 1)
    f["frac_w_monomials_covered"] = n_cov / max(len(checked), 1)
    f["n_w_checked"] = float(len(checked))
    f["n_w_uncovered"] = float(len(checked) - n_cov)
    f["w_check_full"] = float(full_check)
    f["bottom_n_decomp"] = float(bottom_n)
    f["bottom_covered"] = float(bottom_n > 0)
    f["log1p_bottom_n_decomp"] = math.log1p(bottom_n)
    f["lexmax_n_decomp"] = float(lexmax_n)
    f["lexmax_covered"] = float(lexmax_n > 0)
    f["lexmin_n_decomp"] = float(lexmin_n)
    f["cover_sound"] = float(not (cap_u or cap_v))
    return f


# ---------------------------------------------------------------------------
# Bruhat chain features
# ---------------------------------------------------------------------------
def _rank_matrix_fast(x: Sequence[int]) -> np.ndarray:
    return RM.rank_matrix(x)


def chain_features(u: Sequence[int], v: Sequence[int], w: Sequence[int]
                   ) -> Dict[str, float]:
    """Cover / interval-width features of ``u`` and ``v`` inside ``[., w]``.

    ``covers_X_in_wint`` = #{x covering X with x <= w}  (interval width at
    level l(X)+1); ``width2_X`` = #{y : X <= y <= w, l(y) = l(X)+2}.
    ``cocovers_w_geq_X`` counts co-covers of ``w`` that stay above ``X``.
    """
    N = max(len(canonical_perm(u)), len(canonical_perm(v)), len(canonical_perm(w)), 1)
    up, vp, wp = pad(u, N), pad(v, N), pad(w, N)
    Rw = _rank_matrix_fast(wp)

    def leq_w(x: Sequence[int]) -> bool:
        return bool((_rank_matrix_fast(x) >= Rw).all())

    f: Dict[str, float] = {}
    for tag, xp in (("u", up), ("v", vp)):
        covers = cover_transpositions_up(xp)
        in_int = [c for c in covers if leq_w(c)]
        lvl2: Set[Perm] = set()
        for c in in_int:
            for y in cover_transpositions_up(c):
                if y not in lvl2 and leq_w(y):
                    lvl2.add(y)
        f[f"n_covers_{tag}"] = float(len(covers))
        f[f"covers_{tag}_in_wint"] = float(len(in_int))
        f[f"frac_covers_{tag}_in_wint"] = len(in_int) / max(len(covers), 1)
        f[f"width2_{tag}"] = float(len(lvl2))

    # co-covers of w (all y = w t_{ab} with l(y) = l(w) - 1)
    lw = length(wp)
    cocovers = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            y = right_mult_t(wp, a, b)
            if length(y) == lw - 1:
                cocovers.append(y)
    Ru = _rank_matrix_fast(up)
    Rv = _rank_matrix_fast(vp)
    geq_u = sum(1 for y in cocovers if bool((Ru >= _rank_matrix_fast(y)).all()))
    geq_v = sum(1 for y in cocovers if bool((Rv >= _rank_matrix_fast(y)).all()))
    f["n_cocovers_w"] = float(len(cocovers))
    f["cocovers_w_geq_u"] = float(geq_u)
    f["cocovers_w_geq_v"] = float(geq_v)
    f["frac_cocovers_geq_u"] = geq_u / max(len(cocovers), 1)
    f["frac_cocovers_geq_v"] = geq_v / max(len(cocovers), 1)
    return f


# ---------------------------------------------------------------------------
# Transition-tree deltas (budget-guarded, globally memoized)
# ---------------------------------------------------------------------------
class _TransitionBudgetExceeded(Exception):
    pass


_TRANS_MEMO: Dict[Perm, Tuple[int, int, int]] = {}  # perm -> (leaves, depth, max_branch)


def _transition_stats(w: Perm, budget: List[int], depth: int = 0,
                      max_depth: int = 64) -> Tuple[int, int, int]:
    cw = canonical_perm(w)
    hit = _TRANS_MEMO.get(cw)
    if hit is not None:
        return hit
    budget[0] -= 1
    if budget[0] <= 0 or depth >= max_depth:
        raise _TransitionBudgetExceeded
    children = transition_children(cw)
    if not children:
        res = (1, 0, 0)
    else:
        leaves, maxd, maxb = 0, 0, len(children)
        for c in children:
            l, d, b = _transition_stats(c, budget, depth + 1, max_depth)
            leaves += l
            maxd = max(maxd, d)
            maxb = max(maxb, b)
        res = (leaves, maxd + 1, maxb)
    _TRANS_MEMO[cw] = res  # only completed nodes are memoized -> entries valid
    return res


def transition_features(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                        node_budget: int = 30_000) -> Dict[str, float]:
    """LS transition-tree stats for w minus those for u, v (NaN on budget blowout)."""
    f: Dict[str, float] = {}
    stats = {}
    ok = True
    for tag, x in (("u", u), ("v", v), ("w", w)):
        try:
            stats[tag] = _transition_stats(canonical_perm(x), [node_budget])
        except (_TransitionBudgetExceeded, RecursionError):
            stats[tag] = None
            ok = False
    for tag in ("u", "v", "w"):
        s = stats[tag]
        f[f"tr_leaves_{tag}"] = float(s[0]) if s else float("nan")
        f[f"tr_depth_{tag}"] = float(s[1]) if s else float("nan")
    if ok:
        lu, lv, lw = stats["u"][0], stats["v"][0], stats["w"][0]
        f["tr_log_leaf_gap"] = math.log(max(lu, 1)) + math.log(max(lv, 1)) \
            - math.log(max(lw, 1))
        f["tr_depth_gap"] = stats["w"][1] - stats["u"][1] - stats["v"][1]
        f["tr_branch_max_w"] = float(stats["w"][2])
    else:
        f["tr_log_leaf_gap"] = float("nan")
        f["tr_depth_gap"] = float("nan")
        f["tr_branch_max_w"] = float("nan")
    f["tr_valid"] = float(ok)
    return f


# ---------------------------------------------------------------------------
# Full battery
# ---------------------------------------------------------------------------
def triple_features(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                    cap: int = DEFAULT_PD_CAP, max_w_monomials: int = 160,
                    seed: int = 0) -> Dict[str, float]:
    """The complete pipe-dream battery for a triple (see module docstring)."""
    uc, vc, wc = canonical_perm(u), canonical_perm(v), canonical_perm(w)
    N = max(len(uc), len(vc), len(wc))
    f: Dict[str, float] = {
        "N": float(N),
        "ell_u": float(length(uc)),
        "ell_v": float(length(vc)),
        "ell_w": float(length(wc)),
    }
    f.update(coverage_features(uc, vc, wc, cap=cap, max_w_monomials=max_w_monomials,
                               seed=seed))
    f.update(chain_features(uc, vc, wc))
    f.update(transition_features(uc, vc, wc))
    return f


def triple_feature_names() -> List[str]:
    return list(triple_features((2, 1), (2, 1), (3, 1, 2)).keys())
