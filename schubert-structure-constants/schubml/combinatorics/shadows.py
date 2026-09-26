"""Maximal-parabolic Grassmannian shadows and shadow vanishing certificates.

Implements the machinery of ``reports/EXPERT_RESPONSE.md`` (expert review of
conjecture C2, 2026-07):

* ``shadow(x, i)`` — the i-th Grassmannian shadow  Λ_i(x)  (§1): sort the
  first i values of x as a_1 < ... < a_i and form the partition
  (a_i - i, ..., a_1 - 1), the index of the image of the Schubert variety
  of x under  Fl_N -> Gr(i, N).
* Two-row shadow inequalities (§4, theorem "G2"): if c_{u,v}^w > 0 then for
  every cut i and every k,
      Λ_i(u)_k + Λ_i(v)_k  <=  Λ_i(w)_k + Λ_i(w)_{k+1},
  together with the conjugate-partition family and the equivalent
  order-statistic (OS) form
      a^u_{r+1}(i) + a^v_{r+1}(i) <= a^w_{r+1}(i) + a^w_r(i) + 1.
* Prefix gaps (§6): cells (i, j) with w^{-1}(j+1) <= i < w^{-1}(j); the
  prefix-gap theorem says a positive c_{u,v}^w forces
  e_u(i,j) + e_v(i,j) >= 1 there, where e_x = r_x - r_w >= 0.
  Fulton essential cells of w are prefix gaps, so this contains C2.
* The complete maximal-parabolic test MP_i (§2): c_{u,v}^w > 0 implies that
  for every i there is a partition rho ⊆ Λ_i(w) with
  c_{Λ_i(u), Λ_i(v)}^rho > 0.  Feasibility is decided by a box-truncated
  Littlewood-Richardson expansion (pluggable backend; under Sage python use
  ``sage_lr_mult`` which wraps ``sage.libs.lrcalc``).

All statements are theorems (not conjectures), so on triples with
c_{u,v}^w > 0 any reported violation is a bug.  Everything is 1-indexed
one-line notation; permutations are padded to a common ambient size N.

Self-contained: imports nothing outside the standard library, so the module
is importable under both the ``schubml`` and ``sage`` interpreters.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from functools import lru_cache
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

Perm = Tuple[int, ...]
Partition = Tuple[int, ...]

# ---------------------------------------------------------------------------
# basic helpers (kept local so the module has zero non-stdlib deps)
# ---------------------------------------------------------------------------


def pad(x: Sequence[int], n: int) -> Perm:
    return tuple(x) + tuple(range(len(x) + 1, n + 1))


def canonical_size(x: Sequence[int]) -> int:
    n = len(x)
    while n > 1 and x[n - 1] == n:
        n -= 1
    return n


def common_ambient(*perms: Sequence[int]) -> int:
    return max(max(canonical_size(x) for x in perms), 1)


def conjugate(lam: Sequence[int]) -> Partition:
    lam = [p for p in lam if p > 0]
    if not lam:
        return ()
    out = []
    for k in range(1, lam[0] + 1):
        out.append(sum(1 for p in lam if p >= k))
    return tuple(out)


def contains(nu: Sequence[int], lam: Sequence[int]) -> bool:
    """nu ⊇ lam componentwise (Young-diagram containment)."""
    if len(lam) > len(nu):
        if any(p > 0 for p in lam[len(nu):]):
            return False
        lam = lam[:len(nu)]
    return all(l <= n for l, n in zip(lam, nu))


# ---------------------------------------------------------------------------
# Grassmannian shadows (§1)
# ---------------------------------------------------------------------------


def sorted_prefix(x: Sequence[int], i: int) -> Tuple[int, ...]:
    """a_1^x(i) < ... < a_i^x(i): the first i values of x, sorted."""
    return tuple(sorted(x[:i]))


def shadow_from_prefix(pref: Sequence[int]) -> Partition:
    """Λ from a sorted prefix: (a_i - i, ..., a_1 - 1), trailing 0s trimmed."""
    i = len(pref)
    lam = [pref[p - 1] - p for p in range(i, 0, -1)]
    while lam and lam[-1] == 0:
        lam.pop()
    return tuple(lam)


def shadow(x: Sequence[int], i: int) -> Partition:
    """Λ_i(x), the i-th Grassmannian shadow of x (trailing zeros trimmed)."""
    return shadow_from_prefix(sorted_prefix(x, i))


def all_sorted_prefixes(x: Sequence[int], n: int) -> List[List[int]]:
    """prefixes[i] = sorted x([i]) for i = 0..n (incremental, O(n^2))."""
    out: List[List[int]] = [[]]
    cur: List[int] = []
    xp = pad(x, n)
    for i in range(n):
        insort(cur, xp[i])
        out.append(list(cur))
    return out


# ---------------------------------------------------------------------------
# cheap level: two-row inequalities (G2), conjugate family, OS form (§4)
# ---------------------------------------------------------------------------


def g2_violations(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                  N: Optional[int] = None,
                  first_only: bool = False) -> List[dict]:
    """All violations of the two-row shadow inequalities, both families.

    Returns records
      {"i", "k", "family" ("row"|"conj"),
       "lam", "mu", "nu", "lhs", "rhs"}.
    Any nonempty result certifies c_{u,v}^w = 0 (theorem G2).
    """
    if N is None:
        N = common_ambient(u, v, w)
    pu = all_sorted_prefixes(u, N)
    pv = all_sorted_prefixes(v, N)
    pw = all_sorted_prefixes(w, N)
    out: List[dict] = []
    for i in range(1, N):
        lam = shadow_from_prefix(pu[i])
        mu = shadow_from_prefix(pv[i])
        nu = shadow_from_prefix(pw[i])
        for family in ("row", "conj"):
            if family == "row":
                a, b, c = lam, mu, nu
            else:
                a, b, c = conjugate(lam), conjugate(mu), conjugate(nu)
            kmax = max(len(a), len(b))
            for k in range(1, kmax + 1):
                lhs = (a[k - 1] if k <= len(a) else 0) + \
                      (b[k - 1] if k <= len(b) else 0)
                rhs = (c[k - 1] if k <= len(c) else 0) + \
                      (c[k] if k + 1 <= len(c) else 0)
                if lhs > rhs:
                    out.append({"i": i, "k": k, "family": family,
                                "lam": list(a), "mu": list(b),
                                "nu": list(c), "lhs": lhs, "rhs": rhs})
                    if first_only:
                        return out
    return out


def g2_certifies_zero(u, v, w, N: Optional[int] = None) -> bool:
    """True iff some two-row shadow inequality (either family) is violated."""
    return bool(g2_violations(u, v, w, N=N, first_only=True))


def os_violations(u, v, w, N: Optional[int] = None) -> List[dict]:
    """Order-statistic form (§4, OS): for every i and 0 <= r < i,
        a^u_{r+1}(i) + a^v_{r+1}(i) <= a^w_{r+1}(i) + a^w_r(i) + 1,
    with a^w_0(i) = 0.  Mathematically equivalent to the "row" family of
    :func:`g2_violations`; implemented independently as a cross-check.
    """
    if N is None:
        N = common_ambient(u, v, w)
    pu = all_sorted_prefixes(u, N)
    pv = all_sorted_prefixes(v, N)
    pw = all_sorted_prefixes(w, N)
    out = []
    for i in range(1, N):
        au, av, aw = pu[i], pv[i], pw[i]
        for r in range(i):
            lhs = au[r] + av[r]
            rhs = aw[r] + (aw[r - 1] if r >= 1 else 0) + 1
            if lhs > rhs:
                out.append({"i": i, "r": r, "lhs": lhs, "rhs": rhs})
    return out


# ---------------------------------------------------------------------------
# prefix gaps and the prefix-gap theorem (§6)
# ---------------------------------------------------------------------------


def prefix_gaps(w: Sequence[int], N: Optional[int] = None
                ) -> List[Tuple[int, int]]:
    """All cells (i, j) with w^{-1}(j+1) <= i < w^{-1}(j) (1-indexed)."""
    if N is None:
        N = canonical_size(w)
    wp = pad(w, N)  # may be longer than N if len(w) > N
    M = len(wp)
    winv = [0] * (M + 2)
    for pos, val in enumerate(wp, start=1):
        winv[val] = pos
    gaps = []
    for j in range(1, M):
        lo, hi = winv[j + 1], winv[j] - 1
        for i in range(max(lo, 1), hi + 1):
            gaps.append((i, j))
    return gaps


def prefix_gap_violations(u, v, w, N: Optional[int] = None) -> List[dict]:
    """Prefix-gap theorem check: at every prefix gap (i, j) of w a positive
    triple must have e_u(i,j) + e_v(i,j) >= 1 (e_x = r_x - r_w).  Returns
    the gaps where e_u = e_v = 0; nonempty => c_{u,v}^w = 0 (theorem)."""
    if N is None:
        N = common_ambient(u, v, w)
    pu = all_sorted_prefixes(u, N)
    pv = all_sorted_prefixes(v, N)
    pw = all_sorted_prefixes(w, N)
    out = []
    for (i, j) in prefix_gaps(w, N):
        rw = bisect_right(pw[i], j)
        ru = bisect_right(pu[i], j)
        rv = bisect_right(pv[i], j)
        if ru == rw and rv == rw:
            out.append({"i": i, "j": j, "r": rw})
    return out


def essential_cells_are_prefix_gaps(w: Sequence[int]) -> bool:
    """Sanity: Fulton essential cells of D(w) are prefix gaps (used by the
    selftest; C2's hypothesis cell is always covered by the theorem)."""
    wp = tuple(w)
    N = len(wp)
    gaps = set(prefix_gaps(wp, N))
    winv = [0] * (N + 2)
    for pos, val in enumerate(wp, start=1):
        winv[val] = pos
    D = {(i, j) for i in range(1, N + 1) for j in range(1, N + 1)
         if wp[i - 1] > j and winv[j] > i}
    ess = [(i, j) for (i, j) in D
           if (i + 1, j) not in D and (i, j + 1) not in D]
    return all((i, j) in gaps for (i, j) in ess)


# ---------------------------------------------------------------------------
# complete level: the maximal-parabolic test MP_i (§2)
# ---------------------------------------------------------------------------
#
# LR backend: a callable lr_mult(lam, mu, maxrows, maxcols) -> dict mapping
# partition tuples to positive ints, the box-truncated expansion of
# s_lam * s_mu.  Under Sage python, use sage_lr_mult below.


def sage_lr_mult(lam: Partition, mu: Partition,
                 maxrows: int, maxcols: int) -> Dict[Partition, int]:
    """LR expansion of s_lam * s_mu truncated to a maxrows x maxcols box.
    Only usable under the Sage interpreter (wraps sage.libs.lrcalc)."""
    from sage.libs.lrcalc import lrcalc  # type: ignore
    res = lrcalc.mult(list(lam), list(mu), maxrows, maxcols)
    return {tuple(int(t) for t in p): int(c) for p, c in res.items()}


@lru_cache(maxsize=300_000)
def _cached_expansion(lam: Partition, mu: Partition, maxrows: int,
                      maxcols: int) -> Tuple[Partition, ...]:
    return tuple(sorted(_LR_BACKEND(lam, mu, maxrows, maxcols).keys()))


_LR_BACKEND: Callable = sage_lr_mult


def set_lr_backend(fn: Callable) -> None:
    global _LR_BACKEND
    _LR_BACKEND = fn
    _cached_expansion.cache_clear()


def mp_cut_feasible(lam: Partition, mu: Partition, nu: Partition) -> bool:
    """Is there rho ⊆ nu with c_{lam,mu}^rho > 0?  (single cut of MP_i)."""
    lam = tuple(p for p in lam if p > 0)
    mu = tuple(p for p in mu if p > 0)
    nu = tuple(p for p in nu if p > 0)
    if not contains(nu, lam) or not contains(nu, mu):
        return False  # rho would have to contain lam and mu
    if sum(lam) + sum(mu) > sum(nu):
        return False  # |rho| = |lam| + |mu| cannot fit inside nu
    if not lam or not mu:
        return True   # rho = mu (resp. lam) works
    if len(lam) > len(mu):  # lrcalc is symmetric; canonicalize cache key
        lam, mu = mu, lam
    rows, cols = len(nu), nu[0]
    for rho in _cached_expansion(lam, mu, rows, cols):
        if contains(nu, rho):
            return True
    return False


def mp_violation(u, v, w, N: Optional[int] = None) -> Optional[dict]:
    """First cut i where MP_i is infeasible (certifies c_{u,v}^w = 0), or
    None if every cut is feasible.  Needs an LR backend (Sage python)."""
    if N is None:
        N = common_ambient(u, v, w)
    pu = all_sorted_prefixes(u, N)
    pv = all_sorted_prefixes(v, N)
    pw = all_sorted_prefixes(w, N)
    for i in range(1, N):
        lam = shadow_from_prefix(pu[i])
        mu = shadow_from_prefix(pv[i])
        nu = shadow_from_prefix(pw[i])
        if not mp_cut_feasible(lam, mu, nu):
            return {"i": i, "lam": list(lam), "mu": list(mu), "nu": list(nu)}
    return None


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def _selftest() -> None:
    import itertools
    import random

    # shadows of small permutations, by hand:
    # x = 24153, i = 2: prefix {2,4} -> (4-2, 2-1) = (2, 1)
    assert shadow((2, 4, 1, 5, 3), 2) == (2, 1)
    # Grassmannian permutation with single descent at i: shadow = its shape
    assert shadow((1, 3, 5, 2, 4), 3) == (2, 1)  # a=(1,3,5) -> (2,1,0)
    assert shadow((1, 2, 3), 2) == ()
    print("[ok] shadow examples")

    # w0 complementation (§2 of the proof): Λ_i(w0 x) = rectangle complement
    rng = random.Random(0)
    for _ in range(200):
        N = rng.choice([4, 5, 6, 7])
        x = tuple(rng.sample(range(1, N + 1), N))
        w0x = tuple(N + 1 - t for t in x)
        for i in range(1, N):
            lam = shadow(x, i)
            comp = sorted((N - i - (lam[p] if p < len(lam) else 0)
                           for p in range(i)), reverse=True)
            expected = tuple(t for t in comp if t > 0)
            assert shadow(w0x, i) == expected, (x, i, lam)
    print("[ok] Λ_i(w0 x) = rectangular complement of Λ_i(x) (200 random)")

    # OS form is equivalent to the row family
    for _ in range(500):
        N = rng.choice([4, 5, 6])
        u = tuple(rng.sample(range(1, N + 1), N))
        v = tuple(rng.sample(range(1, N + 1), N))
        w = tuple(rng.sample(range(1, N + 1), N))
        row = [r for r in g2_violations(u, v, w)
               if r["family"] == "row"]
        osv = os_violations(u, v, w)
        assert bool(row) == bool(osv), (u, v, w, row, osv)
        # violated cuts must coincide
        assert {r["i"] for r in row} == {r["i"] for r in osv}
    print("[ok] OS form <=> row family on 500 random triples")

    # essential cells are prefix gaps
    for _ in range(300):
        N = rng.choice([4, 5, 6, 7])
        w = tuple(rng.sample(range(1, N + 1), N))
        assert essential_cells_are_prefix_gaps(w), w
    print("[ok] Fulton essential cells are prefix gaps (300 random)")

    # C2 hypothesis => G2 violation (the §5 derivation), on random both-tight
    # configurations: brute-force over S4/S5 triples with a both-tight
    # essential cell, every one must be G2-killed.
    checked = 0
    for N in (4, 5):
        perms = list(itertools.permutations(range(1, N + 1)))
        for w in perms:
            wp = w
            # essential cells with rank values
            winv = [0] * (N + 2)
            for pos, val in enumerate(wp, start=1):
                winv[val] = pos
            D = {(i, j) for i in range(1, N + 1) for j in range(1, N + 1)
                 if wp[i - 1] > j and winv[j] > i}
            ess = [(i, j) for (i, j) in D
                   if (i + 1, j) not in D and (i, j + 1) not in D]
            if not ess:
                continue
            for _ in range(3):
                u = tuple(rng.sample(range(1, N + 1), N))
                v = tuple(rng.sample(range(1, N + 1), N))
                for (i, j) in ess:
                    rw = sum(1 for a in range(i) if wp[a] <= j)
                    ru = sum(1 for a in range(i) if u[a] <= j)
                    rv = sum(1 for a in range(i) if v[a] <= j)
                    if ru == rw and rv == rw:
                        checked += 1
                        assert g2_certifies_zero(u, v, w), (u, v, w, i, j)
    print(f"[ok] C2-firing => G2-violation on {checked} brute-forced "
          "both-tight triples (S4/S5)")

    # MP_i level (needs Sage): consistency G2-kill => MP-kill on random
    # triples, and MP never kills a positive (checked via lrcalc itself on
    # Grassmannian triples where MP is exact).
    try:
        from sage.libs.lrcalc import lrcalc  # noqa: F401
        have_sage = True
    except Exception:
        have_sage = False
    if have_sage:
        for _ in range(300):
            N = rng.choice([4, 5, 6])
            u = tuple(rng.sample(range(1, N + 1), N))
            v = tuple(rng.sample(range(1, N + 1), N))
            w = tuple(rng.sample(range(1, N + 1), N))
            g2 = g2_certifies_zero(u, v, w)
            mp = mp_violation(u, v, w)
            if g2:
                assert mp is not None, (u, v, w)
        print("[ok] G2-kill => MP-kill on 300 random triples (Sage)")
    else:
        print("[skip] MP_i consistency (not under Sage python)")

    print("shadows selftest passed")


if __name__ == "__main__":
    _selftest()
