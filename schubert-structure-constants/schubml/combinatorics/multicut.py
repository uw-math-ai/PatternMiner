"""Multi-cut compatibility machinery for Schubert vanishing (v005 arc).

Background (reports/EXPERT_RESPONSE.md SS12): for every cut i, a positive
c_{u,v}^w forces a Grassmannian LR witness

    rho_i in R_i := { rho subseteq Lambda_i(w) : c_{Lambda_i(u),Lambda_i(v)}^rho > 0 },

(all rho in R_i automatically have |rho| = |Lambda_i(u)| + |Lambda_i(v)| by
grading).  Separate existence of the rho_i (= the MP_i test of shadows.py) is
not sufficient; the witnesses at neighboring cuts must satisfy compatibility
conditions because they arise from a single witness flag V_bullet.

Geometric derivation implemented here
-------------------------------------
Let E, F, G be generic flags and V_bullet a witness flag (exists iff
c_{u,v}^w > 0, with w-side conditions relative to G via w0 w).  Write

    h_i(a, b) = dim(V_i cap E_a cap F_b).

*  h_i is supermodular:  h(a+1,b) + h(a,b+1) <= h(a,b) + h(a+1,b+1)
   (from dim(A+B) + dim(A cap B) = dim A + dim B), i.e. h_i is hive-like;
   its boundary data are the positions alpha_i = pos(V_i, E) supseteq
   Lambda_i(u), beta_i = pos(V_i, F) supseteq Lambda_i(v), and a hypotenuse
   profile tau_i with c_{alpha_i, beta_i}^{tau_i} > 0 and
   tau_i subseteq gamma_i^vee subseteq Lambda_i(w).
*  LR monotonicity (verified computationally, see _selftest):
   c_{alpha,beta}^tau > 0, lambda subseteq alpha, mu subseteq beta
   ==> exists rho subseteq tau with c_{lambda,mu}^rho > 0.
   Hence tau_i lies in the *up-closure*

       T_i := { tau subseteq Lambda_i(w) : exists rho in R_i, rho subseteq tau }.

*  V_i subset V_{i+1} forces jump interlacing of single-flag positions:
   X_{i+1,k+1} <= X_{i,k} <= X_{i+1,k} + 1.  The corresponding relation for
   the hypotenuse profiles tau_i is a *candidate* (gated empirically on
   positives, never assumed).

Chains of positions (alpha_i, beta_i, gamma_i) alone are vacuous beyond MP_i
(the choice alpha_i = Lambda_i(u), ... always works), so any genuine
multi-cut obstruction must constrain the witnesses tau_i / rho_i themselves.
This module provides:

*  witness sets R_i and up-closures T_i per cut (lrcalc backend, memoized);
*  a DP engine over cut-indexed state sets with a pluggable pairwise
   relation ("does a full chain (tau_1, ..., tau_{N-1}) exist?");
*  a relation zoo (interlacing, one-sided interlacing, containment,
   horizontal/vertical strips, parametric slack versions);
*  per-triple witness statistics.

Soundness protocol: a relation is only a *candidate theorem* if the DP is
feasible on every known-positive triple (zero-violation gate); its power is
the kill-rate on the 9,662 MP-surviving zeros.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .shadows import (
    Partition,
    Perm,
    all_sorted_prefixes,
    common_ambient,
    contains,
    pad,
    shadow_from_prefix,
)

# ---------------------------------------------------------------------------
# LR backend (pip lrcalc works in the schubml env; sage wrapper also fine)
# ---------------------------------------------------------------------------


def _lrcalc_mult(lam: Partition, mu: Partition,
                 maxrows: int, maxcols: int) -> Dict[Partition, int]:
    import lrcalc
    res = lrcalc.mult(list(lam), list(mu), maxrows, maxcols)
    return {tuple(int(t) for t in p): int(c) for p, c in res.items()}


_LR_BACKEND: Callable = _lrcalc_mult


def set_lr_backend(fn: Callable) -> None:
    global _LR_BACKEND
    _LR_BACKEND = fn
    _lr_expansion.cache_clear()
    witness_set.cache_clear()
    up_closure.cache_clear()


@lru_cache(maxsize=600_000)
def _lr_expansion(lam: Partition, mu: Partition, maxrows: int,
                  maxcols: int) -> Tuple[Partition, ...]:
    return tuple(sorted(_LR_BACKEND(lam, mu, maxrows, maxcols).keys()))


# ---------------------------------------------------------------------------
# witness sets R_i and up-closures T_i
# ---------------------------------------------------------------------------


def _clean(p: Sequence[int]) -> Partition:
    return tuple(x for x in p if x > 0)


@lru_cache(maxsize=300_000)
def witness_set(lam: Partition, mu: Partition, nu: Partition
                ) -> Tuple[Partition, ...]:
    """R = all rho subseteq nu with c_{lam,mu}^rho > 0 (sorted tuple)."""
    lam, mu, nu = _clean(lam), _clean(mu), _clean(nu)
    if not contains(nu, lam) or not contains(nu, mu):
        return ()
    if sum(lam) + sum(mu) > sum(nu):
        return ()
    if not lam:
        return (mu,)
    if not mu:
        return (lam,)
    if len(lam) > len(mu):
        lam, mu = mu, lam
    rows, cols = len(nu), nu[0]
    return tuple(rho for rho in _lr_expansion(lam, mu, rows, cols)
                 if contains(nu, rho))


@lru_cache(maxsize=200_000)
def _partitions_inside(nu: Partition) -> Tuple[Partition, ...]:
    """All partitions tau subseteq nu."""
    nu = _clean(nu)
    if not nu:
        return ((),)
    out: List[Partition] = []

    def rec(row: int, maxpart: int, acc: Tuple[int, ...]) -> None:
        if row == len(nu):
            out.append(acc)
            return
        top = min(maxpart, nu[row])
        for c in range(top, -1, -1):
            if c == 0:
                out.append(acc)
                return
            rec(row + 1, c, acc + (c,))

    rec(0, nu[0], ())
    return tuple(out)


@lru_cache(maxsize=300_000)
def up_closure(lam: Partition, mu: Partition, nu: Partition
               ) -> Tuple[Partition, ...]:
    """T = all tau subseteq nu containing at least one rho in R(lam,mu,nu).

    By the (computationally verified) LR monotonicity lemma this equals
    { tau subseteq nu : exists alpha supseteq lam, beta supseteq mu with
      c_{alpha,beta}^tau > 0 } -- the possible hive-hypotenuse profiles.
    """
    R = witness_set(lam, mu, nu)
    if not R:
        return ()
    minsize = sum(R[0])
    out = []
    for tau in _partitions_inside(_clean(nu)):
        if sum(tau) < minsize:
            continue
        for rho in R:
            if contains(tau, rho):
                out.append(tau)
                break
    return tuple(out)


# ---------------------------------------------------------------------------
# per-triple cut data
# ---------------------------------------------------------------------------


def cut_data(u: Sequence[int], v: Sequence[int], w: Sequence[int],
             N: Optional[int] = None) -> List[dict]:
    """Shadows and witness sets at every cut i = 1..N-1."""
    if N is None:
        N = common_ambient(u, v, w)
    pu = all_sorted_prefixes(u, N)
    pv = all_sorted_prefixes(v, N)
    pw = all_sorted_prefixes(w, N)
    out = []
    for i in range(1, N):
        lam = shadow_from_prefix(pu[i])
        mu = shadow_from_prefix(pv[i])
        nu = shadow_from_prefix(pw[i])
        out.append({"i": i, "lam": lam, "mu": mu, "nu": nu,
                    "R": witness_set(lam, mu, nu)})
    return out


# ---------------------------------------------------------------------------
# relation zoo
# ---------------------------------------------------------------------------
#
# A relation is a function rel(A, B) -> bool matrix of shape (m, n) where
# A is an (n, L) int array of cut-i states (rows = partitions, zero padded)
# and B is an (m, L) array of cut-(i+1) states; entry [j, i0] says states
# A[i0] (cut i) and B[j] (cut i+1) may be consecutive members of a chain.


def _pad_matrix(parts: Sequence[Partition], L: int) -> np.ndarray:
    A = np.zeros((len(parts), L), dtype=np.int16)
    for r, p in enumerate(parts):
        if p:
            A[r, :len(p)] = p
    return A


def rel_none(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.ones((B.shape[0], A.shape[0]), dtype=bool)


def _rel_lower_s(A: np.ndarray, B: np.ndarray, s: int) -> np.ndarray:
    # B[j, k+s] <= A[i, k] for all k (rows of B beyond box vanish anyway)
    if B.shape[1] <= s:
        return np.ones((B.shape[0], A.shape[0]), dtype=bool)
    Bs = B[:, s:]                      # (m, L-s)
    Ac = A[:, :Bs.shape[1]]            # (n, L-s)
    return (Bs[:, None, :] <= Ac[None, :, :]).all(axis=2)


def _rel_upper_t(A: np.ndarray, B: np.ndarray, t: int) -> np.ndarray:
    # A[i, k] <= B[j, k] + t for all k
    return (A[None, :, :] <= B[:, None, :] + t).all(axis=2)


def rel_lower(A, B):
    return _rel_lower_s(A, B, 1)


def rel_upper(A, B):
    return _rel_upper_t(A, B, 1)


def rel_interlace(A, B):
    """Jump interlacing: B_{k+1} <= A_k <= B_k + 1 (Lambda-chains satisfy
    this exactly; candidate for witness chains)."""
    return _rel_lower_s(A, B, 1) & _rel_upper_t(A, B, 1)


def rel_contain(A, B):
    return _rel_upper_t(A, B, 0)


def rel_hstrip(A, B):
    """B/A is a horizontal strip: A subseteq B and B_{k+1} <= A_k."""
    return _rel_upper_t(A, B, 0) & _rel_lower_s(A, B, 1)


def make_rel(s: Optional[int] = None, t: Optional[int] = None) -> Callable:
    """Parametric relation: B_{k+s} <= A_k (if s) and A_k <= B_k + t (if t)."""
    def rel(A, B):
        m = np.ones((B.shape[0], A.shape[0]), dtype=bool)
        if s is not None:
            m &= _rel_lower_s(A, B, s)
        if t is not None:
            m &= _rel_upper_t(A, B, t)
        return m
    rel.__name__ = f"rel_s{s}_t{t}"
    return rel


RELATIONS: Dict[str, Callable] = {
    "none": rel_none,
    "interlace": rel_interlace,
    "lower": rel_lower,
    "upper": rel_upper,
    "contain": rel_contain,
    "hstrip": rel_hstrip,
}


# ---------------------------------------------------------------------------
# DP engine
# ---------------------------------------------------------------------------


def dp_feasible(state_sets: Sequence[Sequence[Partition]],
                rel: Callable, L: int,
                return_detail: bool = False):
    """Does a chain (x_1, ..., x_k), x_i in state_sets[i-1], exist with
    rel(x_i, x_{i+1}) for all consecutive pairs?

    Returns bool, or (bool, first_dead_cut, reach_counts) with
    return_detail=True.  first_dead_cut is the 1-based index (into
    state_sets) where the reachable set first became empty (None if
    feasible; 0-sized input sets count).
    """
    mats = [_pad_matrix(S, L) for S in state_sets]
    counts: List[int] = []
    reach: Optional[np.ndarray] = None
    for idx, M in enumerate(mats):
        if M.shape[0] == 0:
            counts.append(0)
            return (False, idx + 1, counts) if return_detail else False
        if reach is None:
            reach = np.ones(M.shape[0], dtype=bool)
        else:
            T = rel(mats[idx - 1], M)          # (m, n)
            reach = (T & reach[None, :]).any(axis=1)
        counts.append(int(reach.sum()))
        if not reach.any():
            return (False, idx + 1, counts) if return_detail else False
    return (True, None, counts) if return_detail else True


def analyze_triple(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                   relations: Optional[Dict[str, Callable]] = None,
                   modes: Tuple[str, ...] = ("R", "T"),
                   N: Optional[int] = None,
                   want_stats: bool = True) -> dict:
    """Witness statistics + DP feasibility for each (mode, relation)."""
    if N is None:
        N = common_ambient(u, v, w)
    cuts = cut_data(u, v, w, N)
    res: dict = {"N": N}
    if want_stats:
        res["R_sizes"] = [len(c["R"]) for c in cuts]
        res["T_sizes"] = ([len(up_closure(c["lam"], c["mu"], c["nu"]))
                           for c in cuts] if "T" in modes else None)
        res["tight"] = [sum(c["lam"]) + sum(c["mu"]) == sum(c["nu"])
                        for c in cuts]
        res["nontrivial"] = [bool(c["lam"]) or bool(c["mu"]) for c in cuts]
    if relations:
        L = N
        Rsets = [c["R"] for c in cuts]
        Tsets = ([up_closure(c["lam"], c["mu"], c["nu"]) for c in cuts]
                 if "T" in modes else None)
        feas = {}
        for name, rel in relations.items():
            if "R" in modes:
                feas[f"R:{name}"] = dp_feasible(Rsets, rel, L)
            if "T" in modes:
                feas[f"T:{name}"] = dp_feasible(Tsets, rel, L)
        res["feasible"] = feas
    return res


# ---------------------------------------------------------------------------
# multi-step parabolic projection test (proven necessary condition)
# ---------------------------------------------------------------------------
#
# For any set of cuts D = {n_1 < ... < n_r} subset {1..N-1} let
# P = P_D be the parabolic and pi_D(x) the minimal representative of x W_P
# (sort the values of x within each position block).  If c_{u,v}^w > 0 then
# the projected Schubert problem on Fl(n_1,...,n_r; N) has a point, so
#
#   sigma_{pi(u)} . sigma_{pi(v)} . sigma_{pi(w0 w)}  !=  0   in H*(Fl_N)
#
# (pullback along Fl_N -> Fl(D;N) is an injective ring map), equivalently
#
#   exists z in S_N :  c_{pi(u), pi(v)}^z > 0  and  z <= w0 . pi(w0 w).
#
# D = {i} recovers MP_i; |D| = 2 is the two-step compatibility test between
# cuts i and j.  This is the classical yardstick any new multi-cut
# combinatorial model must be measured against.


def parabolic_min_rep(x: Sequence[int], cuts: Sequence[int],
                      N: int) -> Perm:
    """Minimal length representative of x W_{P_D}: sort values within each
    position block [n_k + 1, n_{k+1}]."""
    xp = pad(x, N)
    bounds = [0] + sorted(set(cuts)) + [N]
    out: List[int] = []
    for a, b in zip(bounds, bounds[1:]):
        out.extend(sorted(xp[a:b]))
    return tuple(out)


def bruhat_leq(a: Sequence[int], b: Sequence[int],
               N: Optional[int] = None) -> bool:
    """a <= b in Bruhat order (prefix order-statistic criterion)."""
    if N is None:
        N = max(common_ambient(a, b), 1)
    pa = all_sorted_prefixes(a, N)
    pb = all_sorted_prefixes(b, N)
    for i in range(1, N):
        for s, t in zip(pa[i], pb[i]):
            if s > t:
                return False
    return True


_SCHUBMULT_MEMO: Dict[Tuple[Perm, Perm, int], Tuple[Perm, ...]] = {}


def _schubmult_support(a: Perm, b: Perm, N: int) -> Tuple[Perm, ...]:
    """Support of sigma_a . sigma_b in H*(Fl_N) (memoized)."""
    if a > b:
        a, b = b, a
    key = (a, b, N)
    hit = _SCHUBMULT_MEMO.get(key)
    if hit is None:
        import lrcalc
        prod = lrcalc.schubmult(list(a), list(b), N)
        hit = tuple(tuple(int(t) for t in z) for z in prod)
        _SCHUBMULT_MEMO[key] = hit
    return hit


def multistep_feasible(u: Sequence[int], v: Sequence[int],
                       w: Sequence[int], cuts: Sequence[int],
                       N: Optional[int] = None) -> bool:
    """True unless the projection to Fl(cuts; N) certifies c_{u,v}^w = 0."""
    if N is None:
        N = common_ambient(u, v, w)
    pu = parabolic_min_rep(u, cuts, N)
    pv = parabolic_min_rep(v, cuts, N)
    w0w = tuple(N + 1 - t for t in pad(w, N))
    zeta = tuple(N + 1 - t for t in parabolic_min_rep(w0w, cuts, N))
    zeta_pref = all_sorted_prefixes(zeta, N)
    for z in _schubmult_support(pu, pv, N):
        zp = pad(z, N)
        # z <= zeta by prefix order statistics (incremental, early exit)
        ok = True
        cur: List[int] = []
        from bisect import insort
        for i in range(N - 1):
            insort(cur, zp[i])
            zt = zeta_pref[i + 1]
            good = True
            for s, t in zip(cur, zt):
                if s > t:
                    good = False
                    break
            if not good:
                ok = False
                break
        if ok:
            return True
    return False


def multistep_violations(u: Sequence[int], v: Sequence[int],
                         w: Sequence[int],
                         max_step: int = 2,
                         N: Optional[int] = None) -> List[Tuple[int, ...]]:
    """All cut sets D with |D| <= max_step whose projection certifies zero
    (only minimal-size D are interesting; we return every violating D)."""
    from itertools import combinations
    if N is None:
        N = common_ambient(u, v, w)
    out = []
    for r in range(1, max_step + 1):
        for D in combinations(range(1, N), r):
            if not multistep_feasible(u, v, w, D, N):
                out.append(D)
    return out


# ---------------------------------------------------------------------------
# ground truth helper (pip lrcalc schubmult; 1-based one-line permutations)
# ---------------------------------------------------------------------------


def schubert_coefficient(u: Sequence[int], v: Sequence[int],
                         w: Sequence[int]) -> int:
    """c_{u,v}^w via lrcalc's Schubert polynomial multiplication."""
    import lrcalc
    n = max(common_ambient(u, v, w), len(u), len(v), len(w))
    from .shadows import pad
    prod = lrcalc.schubmult(list(pad(u, n)), list(pad(v, n)))
    target = pad(w, n)
    for key, c in prod.items():
        kk = tuple(int(t) for t in key)
        if pad(kk, max(len(kk), n)) == pad(target, max(len(kk), n)):
            return int(c)
    return 0


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def _selftest() -> None:
    import itertools
    import random

    from .shadows import shadow

    rng = random.Random(0)

    # Lambda-chains of genuine permutations satisfy rel_interlace exactly
    bad = 0
    for _ in range(500):
        n = rng.choice([4, 5, 6, 7, 8])
        x = tuple(rng.sample(range(1, n + 1), n))
        chain = [shadow(x, i) for i in range(1, n)]
        L = n
        for i in range(len(chain) - 1):
            A = _pad_matrix([chain[i]], L)
            B = _pad_matrix([chain[i + 1]], L)
            if not rel_interlace(A, B)[0, 0]:
                bad += 1
    assert bad == 0, f"Lambda-chain interlacing failed {bad} times"
    print("[ok] Lambda-chains satisfy rel_interlace (500 random perms)")

    # witness_set matches a hand check: lam=mu=(1) in nu=(2,1):
    # s1*s1 = s2 + s11, both inside (2,1)
    assert witness_set((1,), (1,), (2, 1)) == ((1, 1), (2,))
    # nu too small
    assert witness_set((2,), (2,), (2, 1)) == ()
    print("[ok] witness_set hand checks")

    # up-closure equivalence with direct definition on small boxes
    def parts_in_box(rows, cols):
        out = [()]
        def rec(row, maxc, acc):
            if row == rows:
                return
            for c in range(maxc, 0, -1):
                out.append(acc + (c,))
                rec(row + 1, c, acc + (c,))
        rec(0, cols, ())
        return out

    box33 = parts_in_box(3, 3)
    for _ in range(30):
        lam = rng.choice(box33)
        mu = rng.choice(box33)
        nu = rng.choice(box33)
        T = set(up_closure(lam, mu, nu))
        direct = set()
        for alpha in box33:
            if not contains(alpha, lam):
                continue
            for beta in box33:
                if not contains(beta, mu):
                    continue
                for tau in _lr_expansion(tuple(alpha), tuple(beta), 3, 3):
                    if contains(nu, tau):
                        direct.add(tau)
        assert T == direct, (lam, mu, nu, T ^ direct)
    print("[ok] up_closure == {tau : exists alpha>=lam, beta>=mu, "
          "c_{alpha,beta}^tau > 0} (30 random small cases)")

    # DP sanity: with rel_none, feasibility == all R_i nonempty == MP
    u, v, w = (1, 3, 2), (2, 1, 3), (3, 1, 2)   # c = 1 (known positive)
    r = analyze_triple(u, v, w, relations={"none": rel_none,
                                           "interlace": rel_interlace})
    assert r["feasible"]["R:none"] and r["feasible"]["T:none"]
    print("[ok] DP sanity on a positive S3 triple:", r["feasible"])

    # schubert_coefficient agrees with lrcalc on small products
    import lrcalc
    for _ in range(20):
        n = 4
        uu = tuple(rng.sample(range(1, n + 1), n))
        vv = tuple(rng.sample(range(1, n + 1), n))
        prod = lrcalc.schubmult(list(uu), list(vv))
        for key, c in list(prod.items())[:3]:
            ww = tuple(int(t) for t in key)
            assert schubert_coefficient(uu, vv, ww) == int(c)
    print("[ok] schubert_coefficient matches lrcalc.schubmult")

    # exhaustive S4: every positive triple is DP-feasible for R:none/T:none
    perms4 = list(itertools.permutations(range(1, 5)))
    n_pos = 0
    for uu in perms4:
        prod = None
        for vv in perms4:
            import lrcalc as _l
            prod = _l.schubmult(list(uu), list(vv))
            for key, c in prod.items():
                ww = tuple(int(t) for t in key)
                if len(ww) > 6:
                    continue
                r = analyze_triple(uu, vv, ww,
                                   relations={"none": rel_none},
                                   want_stats=False)
                assert r["feasible"]["R:none"], (uu, vv, ww)
                assert r["feasible"]["T:none"], (uu, vv, ww)
                n_pos += 1
    print(f"[ok] MP feasibility (R:none/T:none) on all {n_pos} "
          "positive S4xS4 products")

    # multistep: 1-step projection test must agree with MP_i (witness_set)
    import lrcalc as _lr
    agree = 0
    for _ in range(400):
        n = rng.choice([4, 5, 6])
        uu = tuple(rng.sample(range(1, n + 1), n))
        vv = tuple(rng.sample(range(1, n + 1), n))
        ww = tuple(rng.sample(range(1, n + 1), n))
        N = common_ambient(uu, vv, ww)
        cuts = cut_data(uu, vv, ww, N)
        for c in cuts:
            mp = bool(witness_set(c["lam"], c["mu"], c["nu"]))
            ms = multistep_feasible(uu, vv, ww, (c["i"],), N)
            assert mp == ms, (uu, vv, ww, c["i"], mp, ms)
            agree += 1
    print(f"[ok] 1-step projection test == MP_i on {agree} cuts")

    # multistep: no positive triple may violate any 2-step projection
    checked = 0
    for _ in range(60):
        n = 5
        uu = tuple(rng.sample(range(1, n + 1), n))
        vv = tuple(rng.sample(range(1, n + 1), n))
        prod = _lr.schubmult(list(uu), list(vv))
        for key, cval in prod.items():
            ww = tuple(int(t) for t in key)
            viols = multistep_violations(uu, vv, ww, max_step=2)
            assert not viols, (uu, vv, ww, cval, viols)
            checked += 1
    print(f"[ok] no 1/2-step violation on {checked} positive triples "
          "(S5 products)")

    print("multicut selftest passed")


if __name__ == "__main__":
    _selftest()
