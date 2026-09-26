"""Widened per-triple feature battery for survivor-frontier forensics.

The pqt_residual_v1 model separates survivor zeros from positives within a
product at pairwise win-rate 0.9924 (0.9825 on pairs where n_pd(w) points the
wrong way).  This module supplies a *nameable* feature DSL, wider than
``pipedream_features`` / ``residual_features``, focused on w-side and joint
quantities that vary within a product (u, v fixed):

Families (feature-name prefix = family):

* ``qspec_``     principal-specialization q-analogs ``S_x(1,q,q^2,...)`` at
                 q in {1/2, 2} (n_pd is q=1) + comaj-distribution moments.
                 For a pipe dream D, the monomial x^wt(D) specializes to
                 ``q^{comaj(D)}`` with ``comaj(D) = sum_{(i,j) in D} (i-1)``,
                 so ``S_x(1,q,...) = sum_D q^{comaj(D)}`` (log-sum-exp'd).
                 If ``c_{u,v}^w > 0`` then ``S_u S_v - S_w`` is coeffwise
                 nonnegative, hence ``S_u(q)S_v(q) >= S_w(q)`` for all q>0:
                 the log-ratios are graded margins of the coeffwise
                 certificate, and comaj min/max margins are end-point margins.
* ``chute_``     ladder-move / dream-poset statistics of w (BFS from the
                 bottom dream: #moves at bottom, eccentricity, level widths).
* ``codeconv_``  Lehmer-code convolution counts: # elementwise decompositions
                 ``code(w) = a + b`` (all / graded ``|a| = l(u)`` / Bruhat-
                 checked ``perm(a) <= u, perm(b) <= v``, budget-guarded).
* ``interval_``  Bruhat interval data of [u,w], [v,w]: atoms (covers of u
                 under w), coatoms (cocovers of w above u/v), level-2 widths
                 (reuses :func:`pipedream_features.chain_features`).
* ``descent_``   descent/recoil profiles of w vs u, v (right AND left).
* ``slack_``     rank-matrix slack spectrum ``r_u + r_v - r_w`` at Fulton
                 essential cells of w and on the interior: first/second
                 minima, counts, quantiles, one-sided (u/v) minima.
* ``dominance_`` dominance-order (majorization) margins of sorted code(w)
                 vs code(u)+code(v) (elementwise-sum-sorted and sum-of-sorted).
* ``newton_``    continuous refinements of the Newton/coeffwise certificates:
                 exact coefficient margins ``[S_u S_v]_m - [S_w]_m`` at the
                 bottom monomial x^{code(w)}, the lex-max monomial and a
                 sample of supp(S_w); support-size log-ratio.
* ``npd_``       the pipe-dream-count baseline (q=1), kept as its own family
                 so attribution can isolate "beyond count" signal.

All quantities are exact for uncapped enumerations (cap default 20000; v004
survivor data has l(u), l(v) <= 8, N <= 10, where caps essentially never
bind); ``*_valid`` / ``*_capped`` flags mark the exceptions.  Budget:
< 100 ms / triple at this scale, with per-permutation memoization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..combinatorics import rank_matrix as RM
from ..combinatorics.permutations import (
    Perm,
    canonical_perm,
    code_to_perm,
    inverse,
    lehmer_code,
    left_descents,
    length,
    pad,
    right_descents,
)
from .pipedream_features import _ladder_moves, bottom_pipe_dream, chain_features

DEFAULT_CAP = 20000


# ---------------------------------------------------------------------------
# Per-permutation record (memoized): dream poset + comaj distribution + support
# ---------------------------------------------------------------------------
class PermRecord:
    __slots__ = ("perm", "n", "code", "n_pd", "capped", "comaj_min", "comaj_max",
                 "comaj_mean", "comaj_var", "logS_qhalf", "logS_q2", "log_npd",
                 "depth", "max_width", "top_width", "n_ladder_bottom",
                 "wt_mult", "supp_size")

    def __init__(self, w: Sequence[int], cap: int = DEFAULT_CAP):
        wc = canonical_perm(w)
        self.perm = wc
        n = len(wc)
        self.n = n
        self.code = tuple(lehmer_code(wc))

        bot = bottom_pipe_dream(wc)
        self.n_ladder_bottom = len(_ladder_moves(bot))
        seen = {bot}
        frontier = [bot]
        level_sizes = [1]
        comaj: List[int] = [sum(i - 1 for (i, _j) in bot)]
        wt_mult: Dict[Tuple[int, ...], int] = {}

        def _wt(D):
            wt = [0] * n
            for (i, _j) in D:
                wt[i - 1] += 1
            return tuple(wt)

        wt_mult[_wt(bot)] = 1
        capped = False
        while frontier and not capped:
            nxt = []
            for D in frontier:
                for E in _ladder_moves(D):
                    if E not in seen:
                        seen.add(E)
                        nxt.append(E)
                        comaj.append(sum(i - 1 for (i, _j) in E))
                        wtE = _wt(E)
                        wt_mult[wtE] = wt_mult.get(wtE, 0) + 1
                        if len(seen) >= cap:
                            capped = True
                            break
                if capped:
                    break
            if nxt:
                level_sizes.append(len(nxt))
            frontier = nxt

        self.capped = capped
        self.n_pd = len(seen)
        cm = np.asarray(comaj, dtype=np.float64)
        self.comaj_min = float(cm.min())
        self.comaj_max = float(cm.max())
        self.comaj_mean = float(cm.mean())
        self.comaj_var = float(cm.var())
        self.log_npd = math.log(self.n_pd)
        # log S_w(1,q,q^2,...) = logsumexp(comaj * log q), exact & stable
        for attr, q in (("logS_qhalf", 0.5), ("logS_q2", 2.0)):
            e = cm * math.log(q)
            m = e.max()
            setattr(self, attr, float(m + math.log(np.exp(e - m).sum())))
        self.depth = len(level_sizes) - 1
        self.max_width = max(level_sizes)
        self.top_width = level_sizes[-1]
        self.wt_mult = wt_mult
        self.supp_size = len(wt_mult)


_PERM_MEMO: Dict[Tuple[Perm, int], PermRecord] = {}
_PERM_MEMO_MAX = 100_000


def perm_record(w: Sequence[int], cap: int = DEFAULT_CAP) -> PermRecord:
    key = (canonical_perm(w), cap)
    rec = _PERM_MEMO.get(key)
    if rec is None:
        if len(_PERM_MEMO) > _PERM_MEMO_MAX:
            _PERM_MEMO.clear()
        rec = PermRecord(key[0], cap)
        _PERM_MEMO[key] = rec
    return rec


def _padded_wt_items(rec: PermRecord, N: int) -> List[Tuple[Tuple[int, ...], int]]:
    """Weight-multiplicity items zero-padded to length N."""
    padz = (0,) * (N - rec.n)
    return [(wt + padz, m) for wt, m in rec.wt_mult.items()]


# ---------------------------------------------------------------------------
# Family: qspec (+ chute, npd — all read off the PermRecords)
# ---------------------------------------------------------------------------
def _qspec_features(ru: PermRecord, rv: PermRecord, rw: PermRecord) -> Dict[str, float]:
    f: Dict[str, float] = {}
    f["qspec_logSw_qhalf"] = rw.logS_qhalf
    f["qspec_logSw_q2"] = rw.logS_q2
    f["qspec_logratio_qhalf"] = ru.logS_qhalf + rv.logS_qhalf - rw.logS_qhalf
    f["qspec_logratio_q2"] = ru.logS_q2 + rv.logS_q2 - rw.logS_q2
    # spread of the comaj distribution of w beyond its count (2nd difference in q)
    f["qspec_curv_w"] = rw.logS_qhalf + rw.logS_q2 - 2.0 * rw.log_npd
    f["qspec_comaj_mean_w"] = rw.comaj_mean
    f["qspec_comaj_var_w"] = rw.comaj_var
    f["qspec_comaj_span_w"] = rw.comaj_max - rw.comaj_min
    # moment-matching of S_w against the convolution S_u * S_v
    f["qspec_comaj_mean_gap"] = rw.comaj_mean - ru.comaj_mean - rv.comaj_mean
    f["qspec_comaj_var_gap"] = rw.comaj_var - ru.comaj_var - rv.comaj_var
    # end-point margins (>= 0 required when supp(S_w) subset supp(S_u S_v))
    f["qspec_comaj_min_gap"] = rw.comaj_min - ru.comaj_min - rv.comaj_min
    f["qspec_comaj_max_gap"] = ru.comaj_max + rv.comaj_max - rw.comaj_max
    return f


def _chute_features(ru: PermRecord, rv: PermRecord, rw: PermRecord) -> Dict[str, float]:
    f: Dict[str, float] = {}
    f["chute_ladder_bottom_w"] = float(rw.n_ladder_bottom)
    f["chute_depth_w"] = float(rw.depth)
    f["chute_max_width_w"] = float(rw.max_width)
    f["chute_top_width_w"] = float(rw.top_width)
    f["chute_log_max_width_w"] = math.log(max(rw.max_width, 1))
    f["chute_depth_gap"] = float(rw.depth - ru.depth - rv.depth)
    f["chute_capped_w"] = float(rw.capped)
    return f


def _npd_features(ru: PermRecord, rv: PermRecord, rw: PermRecord) -> Dict[str, float]:
    return {
        "npd_log_w": rw.log_npd,
        "npd_logratio": ru.log_npd + rv.log_npd - rw.log_npd,
        "npd_capped_any": float(ru.capped or rv.capped or rw.capped),
        "npd_supp_w": float(rw.supp_size),
        "npd_density_w": rw.supp_size / max(rw.n_pd, 1),
    }


# ---------------------------------------------------------------------------
# Family: codeconv — Lehmer-code convolution counts
# ---------------------------------------------------------------------------
def _graded_decomp_count(code_w: Sequence[int], lu: int) -> int:
    """#{a : 0 <= a_i <= c_i, sum a_i = lu} — coefficient of x^lu in
    prod_i (1 + x + ... + x^{c_i}).  Any such a (and its complement) is a
    valid Lehmer code since a_i <= c_i <= N - i."""
    lw = sum(code_w)
    dp = np.zeros(lw + 1, dtype=np.float64)
    dp[0] = 1.0
    for c in code_w:
        if c == 0:
            continue
        # dp <- dp * (1 + x + ... + x^c) via prefix sums
        cs = np.concatenate([[0.0], np.cumsum(dp)])
        new = cs[1:] - np.concatenate([np.zeros(min(c + 1, lw + 1)),
                                       cs[1:lw + 1 - c] if c + 1 <= lw else np.zeros(0)])
        dp = new
    return int(round(dp[lu])) if 0 <= lu <= lw else 0


def _bruhat_decomp_count(code_w: Sequence[int], u: Perm, v: Perm, N: int,
                         lu: int, budget: int) -> Optional[Tuple[int, int, int]]:
    """Exact #{(a,b): a+b=code(w), |a|=l(u), perm(a) <= u, perm(b) <= v} plus
    one-sided counts, or None if the graded enumeration exceeds ``budget``."""
    Ru = RM.rank_matrix(pad(u, N))
    Rv = RM.rank_matrix(pad(v, N))
    cw = list(code_w) + [0] * (N - len(code_w))
    suffix_max = np.concatenate([np.cumsum(cw[::-1])[::-1], [0]])
    out = [0, 0, 0]  # (both, u-side, v-side)
    a = [0] * N
    count = [0]

    def rec(i: int, rem: int) -> bool:
        if rem > suffix_max[i]:
            return True
        if i == N:
            count[0] += 1
            if count[0] > budget:
                return False
            ua = code_to_perm(a)
            uok = bool((RM.rank_matrix(ua) >= Ru).all())
            b = [cw[k] - a[k] for k in range(N)]
            vb = code_to_perm(b)
            vok = bool((RM.rank_matrix(vb) >= Rv).all())
            out[1] += uok
            out[2] += vok
            out[0] += uok and vok
            return True
        for ai in range(min(cw[i], rem) + 1):
            a[i] = ai
            if not rec(i + 1, rem - ai):
                return False
        a[i] = 0
        return True

    if not rec(0, lu):
        return None
    return tuple(out)  # type: ignore[return-value]


def _codeconv_features(u: Perm, v: Perm, w: Perm, N: int,
                       budget: int = 2000) -> Dict[str, float]:
    cw = lehmer_code(canonical_perm(w))
    lu, lv = length(u), length(v)
    f: Dict[str, float] = {}
    f["codeconv_log_total"] = float(sum(math.log(c + 1) for c in cw))
    n_graded = _graded_decomp_count(cw, lu)
    f["codeconv_log_graded"] = math.log1p(n_graded)
    f["codeconv_frac_graded"] = n_graded / math.exp(f["codeconv_log_total"])
    if 0 < n_graded <= budget:
        res = _bruhat_decomp_count(cw, canonical_perm(u), canonical_perm(v),
                                   N, lu, budget)
    else:
        res = None
    if res is None:
        f["codeconv_bruhat_valid"] = 0.0
        f["codeconv_log_bruhat"] = float("nan")
        f["codeconv_frac_bruhat"] = float("nan")
        f["codeconv_log_bruhat_u"] = float("nan")
        f["codeconv_log_bruhat_v"] = float("nan")
    else:
        both, uside, vside = res
        f["codeconv_bruhat_valid"] = 1.0
        f["codeconv_log_bruhat"] = math.log1p(both)
        f["codeconv_frac_bruhat"] = both / max(n_graded, 1)
        f["codeconv_log_bruhat_u"] = math.log1p(uside)
        f["codeconv_log_bruhat_v"] = math.log1p(vside)
    return f


# ---------------------------------------------------------------------------
# Family: descent / recoil profiles
# ---------------------------------------------------------------------------
def _descent_features(u: Perm, v: Perm, w: Perm) -> Dict[str, float]:
    DRu, DRv, DRw = set(right_descents(u)), set(right_descents(v)), set(right_descents(w))
    DLu, DLv, DLw = set(left_descents(u)), set(left_descents(v)), set(left_descents(w))
    f: Dict[str, float] = {}
    f["descent_nR_w"] = float(len(DRw))
    f["descent_nL_w"] = float(len(DLw))
    f["descent_R_union_slack"] = float(len(DRu | DRv) - len(DRw & (DRu | DRv)))
    f["descent_R_viol"] = float(len(DRw - (DRu | DRv)))   # 0 by certificate
    f["descent_R_int_u"] = float(len(DRw & DRu))
    f["descent_R_int_v"] = float(len(DRw & DRv))
    f["descent_R_int_both"] = float(len(DRw & DRu & DRv))
    f["descent_L_union_slack"] = float(len(DLu | DLv) - len(DLw & (DLu | DLv)))
    f["descent_L_viol"] = float(len(DLw - (DLu | DLv)))   # >0 would be a NEW certificate hit
    f["descent_L_int_u"] = float(len(DLw & DLu))
    f["descent_L_int_v"] = float(len(DLw & DLv))
    f["descent_maxR_w"] = float(max(DRw) if DRw else 0)
    f["descent_maxL_w"] = float(max(DLw) if DLw else 0)
    f["descent_minR_w"] = float(min(DRw) if DRw else 0)
    return f


# ---------------------------------------------------------------------------
# Family: slack spectrum
# ---------------------------------------------------------------------------
def _slack_features(u: Perm, v: Perm, w: Perm, N: int) -> Dict[str, float]:
    up, vp, wp = pad(u, N), pad(v, N), pad(w, N)
    Ru = RM.rank_matrix(up).astype(np.int64)
    Rv = RM.rank_matrix(vp).astype(np.int64)
    Rw = RM.rank_matrix(wp).astype(np.int64)
    s = Ru + Rv - Rw
    eu = Ru - Rw
    ev = Rv - Rw
    f: Dict[str, float] = {}

    ess = RM.essential_set(wp)
    if ess:
        vals = np.sort(np.array([s[i, j] for (i, j) in ess], dtype=np.int64))
        f["slack_ess_min1"] = float(vals[0])
        f["slack_ess_min2"] = float(vals[1] if len(vals) > 1 else vals[0])
        f["slack_ess_mean"] = float(vals.mean())
        f["slack_ess_max"] = float(vals[-1])
        f["slack_ess_n1"] = float((vals == 1).sum())
        f["slack_ess_n2"] = float((vals == 2).sum())
        uvals = np.sort(np.array([min(int(eu[i, j]), int(ev[i, j])) for (i, j) in ess]))
        f["slack_ess_1side_min1"] = float(uvals[0])
        f["slack_ess_1side_min2"] = float(uvals[1] if len(uvals) > 1 else uvals[0])
        f["slack_ess_umin"] = float(min(int(eu[i, j]) for (i, j) in ess))
        f["slack_ess_vmin"] = float(min(int(ev[i, j]) for (i, j) in ess))
        f["slack_ess_size"] = float(len(ess))
    else:
        for k in ("slack_ess_min1", "slack_ess_min2", "slack_ess_mean", "slack_ess_max",
                  "slack_ess_n1", "slack_ess_n2", "slack_ess_1side_min1",
                  "slack_ess_1side_min2", "slack_ess_umin", "slack_ess_vmin",
                  "slack_ess_size"):
            f[k] = 0.0

    si = np.sort(s[1:N, 1:N].ravel())
    f["slack_int_min1"] = float(si[0]) if si.size else 0.0
    f["slack_int_min2"] = float(si[1] if si.size > 1 else f["slack_int_min1"])
    f["slack_int_p25"] = float(np.percentile(si, 25)) if si.size else 0.0
    f["slack_int_p50"] = float(np.percentile(si, 50)) if si.size else 0.0
    f["slack_int_mean"] = float(si.mean()) if si.size else 0.0
    f["slack_int_n0"] = float((si == 0).sum())
    f["slack_int_n1"] = float((si == 1).sum())
    both0 = (eu[1:N, 1:N] == 0) & (ev[1:N, 1:N] == 0)
    f["slack_int_nboth0"] = float(both0.sum())
    return f


# ---------------------------------------------------------------------------
# Family: dominance margins
# ---------------------------------------------------------------------------
def _dominance_features(u: Perm, v: Perm, w: Perm, N: int) -> Dict[str, float]:
    cu = np.array(lehmer_code(pad(u, N)), dtype=np.int64)
    cv = np.array(lehmer_code(pad(v, N)), dtype=np.int64)
    cw = np.array(lehmer_code(pad(w, N)), dtype=np.int64)
    lw = np.sort(cw)[::-1]
    mu1 = np.sort(cu + cv)[::-1]                       # elementwise sum, sorted
    mu2 = np.sort(cu)[::-1] + np.sort(cv)[::-1]        # sum of sorted (majorizes mu1)
    m1 = np.cumsum(mu1) - np.cumsum(lw)
    m2 = np.cumsum(mu2) - np.cumsum(lw)
    f: Dict[str, float] = {}
    f["dominance_min_margin1"] = float(m1.min())
    f["dominance_n_tight1"] = float((m1 == 0).sum())
    f["dominance_sum_margin1"] = float(m1.sum())
    f["dominance_neg1"] = float(m1.min() < 0)
    f["dominance_min_margin2"] = float(m2.min())
    f["dominance_n_tight2"] = float((m2 == 0).sum())
    f["dominance_top_gap"] = float(mu2[0] - lw[0])
    f["dominance_max_code_w"] = float(lw[0])
    return f


# ---------------------------------------------------------------------------
# Family: newton — exact coefficient margins of S_u S_v vs S_w
# ---------------------------------------------------------------------------
def _newton_features(ru: PermRecord, rv: PermRecord, rw: PermRecord, N: int,
                     n_sample: int = 48, seed: int = 0) -> Dict[str, float]:
    f: Dict[str, float] = {}
    f["newton_supp_logratio"] = (math.log(ru.supp_size) + math.log(rv.supp_size)
                                 - math.log(rw.supp_size))
    valid = not (ru.capped or rv.capped or rw.capped)
    f["newton_valid"] = float(valid)
    keys = ("newton_margin_bottom", "newton_margin_lexmax", "newton_margin_min",
            "newton_margin_mean", "newton_frac_tight", "newton_log_prod_bottom",
            "newton_bottom_ndecomp")
    if not valid:
        for k in keys:
            f[k] = float("nan")
        return f

    items_u = _padded_wt_items(ru, N)
    dv = dict(_padded_wt_items(rv, N))
    dw = dict(_padded_wt_items(rw, N))
    if len(dv) < len(items_u):   # iterate the smaller support
        items_u, dv = list(dv.items()), dict(items_u)

    def prod_coeff(m: Tuple[int, ...]) -> Tuple[int, int]:
        tot, ndec = 0, 0
        for a, ma in items_u:
            b = tuple(x - y for x, y in zip(m, a))
            if min(b) < 0:
                continue
            mb = dv.get(b)
            if mb:
                tot += ma * mb
                ndec += 1
        return tot, ndec

    supp_w = sorted(dw.keys())
    bottom = rw.code + (0,) * (N - len(rw.code))
    lexmax = supp_w[-1]
    checked = {bottom, lexmax}
    if len(supp_w) > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(supp_w), size=n_sample, replace=False)
        checked.update(supp_w[i] for i in idx)
    else:
        checked.update(supp_w)

    margins = []
    for m in checked:
        pc, _ = prod_coeff(m)
        margins.append(pc - dw[m])
    margins = np.array(margins, dtype=np.float64)
    pb, ndec_b = prod_coeff(bottom)
    f["newton_margin_bottom"] = float(pb - dw[bottom])
    pl, _ = prod_coeff(lexmax)
    f["newton_margin_lexmax"] = float(pl - dw[lexmax])
    f["newton_margin_min"] = float(margins.min())
    f["newton_margin_mean"] = float(margins.mean())
    f["newton_frac_tight"] = float((margins == 0).mean())
    f["newton_log_prod_bottom"] = math.log1p(pb)
    f["newton_bottom_ndecomp"] = float(ndec_b)
    return f


# ---------------------------------------------------------------------------
# Full battery
# ---------------------------------------------------------------------------
def deep_features(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                  cap: int = DEFAULT_CAP, codeconv_budget: int = 2000,
                  newton_sample: int = 48, include_interval: bool = True
                  ) -> Dict[str, float]:
    """The widened battery.  Returns {feature_name: value}; family = prefix."""
    uc, vc, wc = canonical_perm(u), canonical_perm(v), canonical_perm(w)
    N = max(len(uc), len(vc), len(wc))
    ru, rv, rw = perm_record(uc, cap), perm_record(vc, cap), perm_record(wc, cap)
    f: Dict[str, float] = {}
    f.update(_npd_features(ru, rv, rw))
    f.update(_qspec_features(ru, rv, rw))
    f.update(_chute_features(ru, rv, rw))
    f.update(_codeconv_features(uc, vc, wc, N, budget=codeconv_budget))
    f.update(_descent_features(pad(uc, N), pad(vc, N), pad(wc, N)))
    f.update(_slack_features(uc, vc, wc, N))
    f.update(_dominance_features(uc, vc, wc, N))
    f.update(_newton_features(ru, rv, rw, N, n_sample=newton_sample))
    if include_interval:
        f.update({f"interval_{k}": val for k, val in chain_features(uc, vc, wc).items()})
    return f


def family_of(name: str) -> str:
    return name.split("_", 1)[0]


def deep_feature_names(include_interval: bool = True) -> List[str]:
    return list(deep_features((2, 1), (2, 1), (3, 1, 2),
                              include_interval=include_interval).keys())
