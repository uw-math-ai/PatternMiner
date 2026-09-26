"""Richmond factorization at dimension-tight cuts, and multiplicity cores.

Implements the machinery of ``reports/EXPERT_RESPONSE_2.md`` §7 (second expert
review, eq. 7.1/7.2) and §11 Priority C:

* **Dimension-tight cuts.**  For a support triple (u, v, w) in ambient S_N
  (N = common canonical ambient, l(u)+l(v)=l(w)), a cut i in {1..N-1} is
  *tight* if |Λ_i(u)| + |Λ_i(v)| = |Λ_i(w)| where Λ_i is the i-th
  Grassmannian shadow (see ``shadows.py``).  Because
  |Λ_i(x)| = x(1)+...+x(i) - i(i+1)/2, tightness is detected with prefix
  sums in O(N) per triple.

* **Richmond's product formula (7.1).**  At every tight cut i,

      c_{u,v}^w = LR_i · c_{uL,vL}^{wL} · c_{uR,vR}^{wR},

  where LR_i = c^{Λ_i(w)}_{Λ_i(u),Λ_i(v)} is a Grassmannian LR coefficient
  and the fibers use the *standardized position blocks*
  xL = std(x(1), ..., x(i)),  xR = std(x(i+1), ..., x(N))
  (std = relabel the values to 1..k preserving relative order).  This is a
  theorem (Richmond, arXiv:0812.1856, applied to Fl_N -> Gr(i,N); the
  dimension-tight equality is exactly the base codimension equality, no
  Levi-movability needed) — verified computationally by the selftest and by
  ``scripts/richmond_cores.py verify``.

* **The lcm law (7.2).**  lcm over tight cuts of LR_i divides c (immediate
  from C9 = per-cut divisibility).

* **Multiplicity cores.**  ``decompose`` recursively factors a triple at
  tight cuts until no factor admits one; the leaves ("Richmond cores",
  triples with NO tight cut in their canonical ambient) together with the
  Grassmannian base factors satisfy  c = prod(LR bases) · prod(core c).
  Identity leaves (l(w)=0) are dropped.  Coefficient propagation only ever
  *computes* the smaller fiber (block size <= N/2) via the Schubert-product
  backend and infers the larger one by exact division — the integrality of
  that division is itself a continuous test of (7.1).

Ambient convention: the *trivial* cut at i >= N (after padding) is always
tight with empty shadows and an identity factor; it is excluded — cuts range
over 1..N-1 in the triple's own canonical ambient.  (The earlier full-corpus
probe padded with +1 headroom, which is why every triple had a "tight level"
there; decomposability here refers to genuine cuts only.)

Backends (both only needed under the Sage interpreter, lazily imported):
  * LR coefficients:      sage.libs.lrcalc lrcoef
  * Schubert coefficients: sage.libs.lrcalc mult_schubert (memoized)
Both are pluggable via ``set_lr_backend`` / ``set_schub_backend`` so the
module stays importable (and the combinatorial half testable) anywhere.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from schubml.combinatorics.shadows import (Partition, Perm, pad,
                                           shadow_from_prefix)

# ---------------------------------------------------------------------------
# basic block combinatorics
# ---------------------------------------------------------------------------


def standardize(vals: Sequence[int]) -> Perm:
    """Relabel ``vals`` (distinct ints) to 1..k preserving relative order.

    std((5, 2, 6)) == (2, 1, 3).
    """
    order = sorted(range(len(vals)), key=lambda k: vals[k])
    out = [0] * len(vals)
    for rank, k in enumerate(order):
        out[k] = rank + 1
    return tuple(out)


def canonical(x: Sequence[int]) -> Perm:
    """Trim trailing fixed points (identity collapses to ``(1,)``)."""
    n = len(x)
    while n > 1 and x[n - 1] == n:
        n -= 1
    return tuple(x[:n])


def triple_ambient(u: Sequence[int], v: Sequence[int],
                   w: Sequence[int]) -> int:
    """Common canonical ambient N of the triple."""
    return max(len(canonical(u)), len(canonical(v)), len(canonical(w)))


def crossing_weights(x: Sequence[int], N: int) -> List[int]:
    """[|Λ_1(x)|, ..., |Λ_{N-1}(x)|] via prefix sums (x padded to N).

    |Λ_i(x)| = sum_{p<=i} x(p) - i(i+1)/2  = #inversions crossing cut i.
    """
    xp = x if len(x) >= N else tuple(x) + tuple(range(len(x) + 1, N + 1))
    out = []
    ps = 0
    for i in range(1, N):
        ps += xp[i - 1]
        out.append(ps - i * (i + 1) // 2)
    return out


def cut_slacks(u: Sequence[int], v: Sequence[int], w: Sequence[int],
               N: Optional[int] = None) -> List[int]:
    """slack_i = |Λ_i(w)| - |Λ_i(u)| - |Λ_i(v)| for i = 1..N-1.

    On support triples slack_i >= 0 for every i (maximal-parabolic theorem);
    slack_i == 0  <=>  cut i is dimension-tight.
    """
    if N is None:
        N = triple_ambient(u, v, w)
    cu = crossing_weights(u, N)
    cv = crossing_weights(v, N)
    cw = crossing_weights(w, N)
    return [cw[k] - cu[k] - cv[k] for k in range(N - 1)]


def tight_cuts(u: Sequence[int], v: Sequence[int], w: Sequence[int],
               N: Optional[int] = None) -> List[int]:
    """All dimension-tight cuts i in {1..N-1} (canonical ambient)."""
    return [k + 1 for k, s in enumerate(cut_slacks(u, v, w, N)) if s == 0]


# ---------------------------------------------------------------------------
# direct-sum (block) decomposition — EXPERT_RESPONSE_3 §8
# ---------------------------------------------------------------------------
#
# A permutation x (canonical length n) is a NONTRIVIAL DIRECT SUM iff some
# prefix {1..p} maps to itself for 0 < p < n; equivalently the crossing
# weight |Λ_p(x)| = x(1)+...+x(p) - p(p+1)/2 vanishes.  For a TRIPLE the
# relevant notion is a COMMON block split: a p (0 < p < N, N = canonical
# common ambient) with |Λ_p(u)| = |Λ_p(v)| = |Λ_p(w)| = 0, i.e.
# u = a ⊕ a', v = b ⊕ b', w = d ⊕ d' with blocks of sizes (p, N-p).  At such
# a cut Richmond's formula degenerates to the direct-sum product law
# c_{u,v}^w = c_{a,b}^d · c_{a',b'}^{d'} (all three shadows empty, LR = 1),
# so every common split IS a dimension-tight cut with trivial base — the
# subclass of Richmond decompositions that merely juxtaposes independent
# problems.  Triples with no common split are DIRECT-SUM-PRIMITIVE; cores
# (no tight cut at all) are automatically direct-sum-primitive.


def perm_split_points(x: Sequence[int], n: Optional[int] = None) -> List[int]:
    """All p (0 < p < n) with x({1..p}) = {1..p}; n = canonical length."""
    xc = canonical(x)
    if n is None:
        n = len(xc)
    return [k + 1 for k, cwt in enumerate(crossing_weights(xc, n)) if cwt == 0]


def is_perm_direct_sum(x: Sequence[int]) -> bool:
    """True iff canonical(x) is a nontrivial direct sum x = a ⊕ a'."""
    return bool(perm_split_points(x))


def direct_sum_splits(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                      N: Optional[int] = None) -> List[int]:
    """Common block splits of the triple: all p in {1..N-1} (canonical
    common ambient N) with |Λ_p(u)| = |Λ_p(v)| = |Λ_p(w)| = 0."""
    if N is None:
        N = triple_ambient(u, v, w)
    cu = crossing_weights(u, N)
    cv = crossing_weights(v, N)
    cw = crossing_weights(w, N)
    return [k + 1 for k in range(N - 1)
            if cu[k] == 0 and cv[k] == 0 and cw[k] == 0]


def is_direct_sum_primitive(u: Sequence[int], v: Sequence[int],
                            w: Sequence[int],
                            N: Optional[int] = None) -> bool:
    """True iff the triple admits no common block split (no p with
    u, v, w all stabilizing the prefix {1..p}, 0 < p < N)."""
    return not direct_sum_splits(u, v, w, N)


def direct_sum(x: Sequence[int], y: Sequence[int]) -> Perm:
    """Block direct sum x ⊕ y (one-line notation)."""
    p = len(x)
    return tuple(x) + tuple(t + p for t in y)


def shadow_triple(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                  i: int, N: int) -> Tuple[Partition, Partition, Partition]:
    """(Λ_i(u), Λ_i(v), Λ_i(w)) with all perms padded to N."""
    up, vp, wp = pad(u, N), pad(v, N), pad(w, N)
    return (shadow_from_prefix(sorted(up[:i])),
            shadow_from_prefix(sorted(vp[:i])),
            shadow_from_prefix(sorted(wp[:i])))


def factor_at_cut(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                  i: int, N: Optional[int] = None):
    """Richmond data at cut i: (lam, mu, nu), left triple, right triple.

    Returns ((lam, mu, nu), (uL, vL, wL), (uR, vR, wR)); the block
    permutations are standardized and canonicalized.
    """
    if N is None:
        N = triple_ambient(u, v, w)
    up, vp, wp = pad(u, N), pad(v, N), pad(w, N)
    lam = shadow_from_prefix(sorted(up[:i]))
    mu = shadow_from_prefix(sorted(vp[:i]))
    nu = shadow_from_prefix(sorted(wp[:i]))
    left = tuple(canonical(standardize(x[:i])) for x in (up, vp, wp))
    right = tuple(canonical(standardize(x[i:])) for x in (up, vp, wp))
    return (lam, mu, nu), left, right


# ---------------------------------------------------------------------------
# backends: Grassmannian LR and full-flag Schubert coefficients
# ---------------------------------------------------------------------------


def sage_lrcoef(nu: Partition, lam: Partition, mu: Partition) -> int:
    """c^nu_{lam,mu} via lrcalc (Sage interpreter only)."""
    from sage.libs.lrcalc import lrcalc  # type: ignore
    return int(lrcalc.lrcoef(list(nu), list(lam), list(mu)))


def sage_schub_expand(u: Perm, v: Perm) -> Dict[Perm, int]:
    """Full expansion S_u · S_v = sum_w c_{u,v}^w S_w via lrcalc
    mult_schubert; keys are canonical permutation tuples."""
    from sage.libs.lrcalc import lrcalc  # type: ignore
    res = lrcalc.mult_schubert(list(u), list(v))
    return {canonical(tuple(int(t) for t in p)): int(c)
            for p, c in res.items()}


_LR_BACKEND: Callable = sage_lrcoef
_SCHUB_BACKEND: Callable = sage_schub_expand
_LR_MEMO: Dict[tuple, int] = {}
_SCHUB_MEMO: Dict[tuple, Dict[Perm, int]] = {}
_SCHUB_MEMO_TERMS = [0]          # total stored expansion terms (size cap)
_SCHUB_MEMO_TERMS_CAP = 4_000_000


def set_lr_backend(fn: Callable) -> None:
    global _LR_BACKEND
    _LR_BACKEND = fn
    _LR_MEMO.clear()


def set_schub_backend(fn: Callable) -> None:
    global _SCHUB_BACKEND
    _SCHUB_BACKEND = fn
    _SCHUB_MEMO.clear()


def lr_coeff(nu: Partition, lam: Partition, mu: Partition) -> int:
    """Memoized c^nu_{lam,mu} (lam, mu canonicalized to a symmetric key)."""
    lam = tuple(p for p in lam if p > 0)
    mu = tuple(p for p in mu if p > 0)
    nu = tuple(p for p in nu if p > 0)
    if lam > mu:
        lam, mu = mu, lam
    key = (nu, lam, mu)
    hit = _LR_MEMO.get(key)
    if hit is None:
        if len(_LR_MEMO) > 2_000_000:
            _LR_MEMO.clear()
        hit = _LR_BACKEND(nu, lam, mu)
        _LR_MEMO[key] = hit
    return hit


def schub_coeff(u: Perm, v: Perm, w: Perm) -> int:
    """c_{u,v}^w, memoized per product (u, v).  Blocks of ambient <= 2 are
    resolved combinatorially (all S_1/S_2 support coefficients are 1)."""
    u, v, w = canonical(u), canonical(v), canonical(w)
    if len(w) <= 2 and len(u) <= 2 and len(v) <= 2:
        # S_2 exactly: c = 1 iff lengths are additive (l <= 1 each), else 0.
        lu, lv, lw = (1 if len(u) == 2 else 0), (1 if len(v) == 2 else 0), \
            (1 if len(w) == 2 else 0)
        return 1 if lu + lv == lw else 0
    if u > v:
        u, v = v, u
    key = (u, v)
    hit = _SCHUB_MEMO.get(key)
    if hit is None:
        if _SCHUB_MEMO_TERMS[0] > _SCHUB_MEMO_TERMS_CAP:
            _SCHUB_MEMO.clear()
            _SCHUB_MEMO_TERMS[0] = 0
        hit = _SCHUB_BACKEND(u, v)
        _SCHUB_MEMO[key] = hit
        _SCHUB_MEMO_TERMS[0] += len(hit) + 1
    return hit.get(w, 0)


# ---------------------------------------------------------------------------
# recursive Richmond decomposition -> base factors + multiplicity cores
# ---------------------------------------------------------------------------


def _is_identity(t: Tuple[Perm, Perm, Perm]) -> bool:
    return all(len(x) == 1 or all(x[k] == k + 1 for k in range(len(x)))
               for x in t)


def decompose(u: Sequence[int], v: Sequence[int], w: Sequence[int], c: int,
              collect_cuts: bool = False) -> dict:
    """Fully Richmond-factor the support triple (u, v, w) with c_{u,v}^w = c.

    Recursion: split at the *smallest* tight cut (deterministic); at each
    split compute the smaller fiber's coefficient with the Schubert backend
    and infer the larger one by exact division (raising AssertionError on
    any non-integrality / non-positivity, which would falsify (7.1) or the
    conventions).

    Returns a dict:
      bases:  [(lam, mu, nu, LR_i), ...]      Grassmannian base factors
      cores:  [((u, v, w), c), ...]           leaves with no tight cut
                                              (identity leaves dropped)
      n_splits, depth:                        recursion statistics
      product_ok:  c == prod(LR) * prod(core c)   (theorem check)
      fiber_ge2:   list of core coefficients >= 2 (the "exceptional fibers")
    """
    bases: List[tuple] = []
    cores: List[tuple] = []
    cuts_used: List[tuple] = []
    n_splits = 0
    depth = 0
    # stack of (triple, coefficient, level)
    t0 = tuple(canonical(x) for x in (u, v, w))
    stack: List[tuple] = [(t0, int(c), 0)]
    while stack:
        (tu, tv, tw), tc, lvl = stack.pop()
        depth = max(depth, lvl)
        if _is_identity((tu, tv, tw)):
            assert tc == 1, (tu, tv, tw, tc)
            continue
        N = triple_ambient(tu, tv, tw)
        cuts = tight_cuts(tu, tv, tw, N)
        if not cuts:
            cores.append(((tu, tv, tw), tc))
            continue
        i = cuts[0]
        (lam, mu, nu), left, right = factor_at_cut(tu, tv, tw, i, N)
        lr = lr_coeff(nu, lam, mu)
        assert lr >= 1, ("LR=0 at tight cut on support triple",
                         tu, tv, tw, i, lam, mu, nu)
        bases.append((lam, mu, nu, lr))
        if collect_cuts:
            cuts_used.append((N, i))
        n_splits += 1
        # compute the smaller block directly, infer the other by division
        if i <= N - i:
            small, big = left, right
        else:
            small, big = right, left
        c_small = schub_coeff(*small)
        assert c_small >= 1, ("fiber coefficient 0 (convention bug?)",
                              tu, tv, tw, i, small)
        rem = tc // (lr * c_small)
        assert rem * lr * c_small == tc and rem >= 1, (
            "non-exact Richmond division", tu, tv, tw, i, tc, lr, c_small)
        stack.append((small, c_small, lvl + 1))
        stack.append((big, rem, lvl + 1))
    prod = 1
    for _, _, _, lr in bases:
        prod *= lr
    for _, cc in cores:
        prod *= cc
    out = {
        "bases": bases,
        "cores": cores,
        "n_splits": n_splits,
        "depth": depth,
        "product_ok": (prod == int(c)),
        "fiber_ge2": [cc for _, cc in cores if cc >= 2],
    }
    if collect_cuts:
        out["cuts_used"] = cuts_used
    return out


def lcm_of_tight_lr(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                    N: Optional[int] = None) -> Tuple[int, int, int]:
    """(lcm, max, n_tight) of LR_i over all genuine tight cuts (1 if none)."""
    if N is None:
        N = triple_ambient(u, v, w)
    cuts = tight_cuts(u, v, w, N)
    lc, mx = 1, 1
    for i in cuts:
        lam, mu, nu = shadow_triple(u, v, w, i, N)
        lr = lr_coeff(nu, lam, mu)
        lc = math.lcm(lc, lr)
        mx = max(mx, lr)
    return lc, mx, len(cuts)


# ---------------------------------------------------------------------------
# selftest (needs the Sage interpreter for the LR / Schubert backends)
# ---------------------------------------------------------------------------


def _selftest() -> None:
    import itertools
    import random

    from schubml.combinatorics.shadows import shadow

    assert standardize((5, 2, 6)) == (2, 1, 3)
    assert standardize((3, 1, 2)) == (3, 1, 2)
    assert standardize((7,)) == (1,)
    assert canonical((2, 1, 3, 4)) == (2, 1)
    print("[ok] standardize / canonical")

    rng = random.Random(0)
    for _ in range(200):
        N = rng.choice([4, 5, 6, 7, 8])
        x = tuple(rng.sample(range(1, N + 1), N))
        ws = crossing_weights(x, N)
        for i in range(1, N):
            assert ws[i - 1] == sum(shadow(x, i)), (x, i)
    print("[ok] crossing_weights == |Λ_i| (200 random perms)")

    # --- direct-sum detection (EXPERT_RESPONSE_3 §8) --------------------
    assert perm_split_points((2, 1, 4, 3)) == [2]
    assert perm_split_points((2, 3, 1)) == []
    assert not is_perm_direct_sum((3, 1, 2))
    assert is_perm_direct_sum((1, 3, 2, 4, 5))          # canonical = (1,3,2)
    assert direct_sum((2, 1), (1, 3, 2)) == (2, 1, 3, 5, 4)
    rng = random.Random(7)
    n_brute = 0
    for _ in range(300):
        n = rng.choice([3, 4, 5, 6, 7])
        x = tuple(rng.sample(range(1, n + 1), n))
        xc = canonical(x)
        brute = [p for p in range(1, len(xc))
                 if set(xc[:p]) == set(range(1, p + 1))]
        assert perm_split_points(x) == brute, (x, brute)
        n_brute += 1
    print(f"[ok] perm_split_points == brute-force prefix test "
          f"({n_brute} random perms)")

    n_tri = n_split_tight = 0
    for _ in range(400):
        n = rng.choice([4, 5, 6, 7])
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        w = tuple(rng.sample(range(1, n + 1), n))
        N = triple_ambient(u, v, w)
        up = pad(canonical(u), N)
        vp = pad(canonical(v), N)
        wp = pad(canonical(w), N)
        brute = [p for p in range(1, N)
                 if set(up[:p]) == set(vp[:p]) == set(wp[:p])
                 == set(range(1, p + 1))]
        ds = direct_sum_splits(u, v, w)
        assert ds == brute, (u, v, w, ds, brute)
        assert is_direct_sum_primitive(u, v, w) == (not brute)
        # every common split is a dimension-tight cut with empty shadows
        tset = set(tight_cuts(u, v, w, N))
        for p in ds:
            assert p in tset, (u, v, w, p)
            lam, mu, nu = shadow_triple(u, v, w, p, N)
            assert not any(lam) and not any(mu) and not any(nu)
            n_split_tight += 1
        n_tri += 1
    print(f"[ok] direct_sum_splits == brute force on {n_tri} random triples; "
          f"every split is a tight cut with empty shadows "
          f"({n_split_tight} splits)")

    # constructed direct sums are detected at the block boundary
    for _ in range(100):
        p = rng.choice([2, 3, 4])
        q = rng.choice([2, 3, 4])
        a = tuple(rng.sample(range(1, p + 1), p))
        b = tuple(rng.sample(range(1, q + 1), q))
        x = direct_sum(a, b)
        if len(canonical(x)) > p:      # b nontrivial beyond the boundary
            assert p in perm_split_points(x), (a, b, x)
    print("[ok] constructed x = a ⊕ b detected at the block boundary")

    try:
        from sage.libs.lrcalc import lrcalc  # noqa: F401
    except Exception:
        print("[skip] backend tests (not under the Sage interpreter)")
        return

    # (7.1) at EVERY tight cut, brute force over full S4 x S4 and random
    # S5/S6 products; fibers computed INDEPENDENTLY (no division inference).
    n_checked, n_tightcuts = 0, 0
    perms4 = list(itertools.permutations(range(1, 5)))
    prods = [(u, v) for u in perms4 for v in perms4]
    rng = random.Random(1)
    for n, reps in ((5, 250), (6, 250)):
        vals = list(range(1, n + 1))
        for _ in range(reps):
            u = tuple(rng.sample(vals, n))
            v = tuple(rng.sample(vals, n))
            prods.append((u, v))
    for (u, v) in prods:
        exp = sage_schub_expand(canonical(u), canonical(v))
        for w, c in exp.items():
            N = triple_ambient(u, v, w)
            n_checked += 1
            for i in tight_cuts(u, v, w, N):
                (lam, mu, nu), left, right = factor_at_cut(u, v, w, i, N)
                lr = lr_coeff(nu, lam, mu)
                cl = schub_coeff(*left)
                cr = schub_coeff(*right)
                assert c == lr * cl * cr, (
                    "(7.1) FAILED", u, v, w, i, c, lr, cl, cr)
                n_tightcuts += 1
    print(f"[ok] (7.1) holds at every tight cut: {n_tightcuts} cuts over "
          f"{n_checked} support triples (S4 exhaustive + 500 random S5/S6)")

    # decompose(): product invariant + cores have no tight cut + lcm law
    n_dec, n_core_triples = 0, 0
    for (u, v) in prods[:576] + prods[576::7]:
        exp = sage_schub_expand(canonical(u), canonical(v))
        for w, c in exp.items():
            d = decompose(u, v, w, c)
            assert d["product_ok"], (u, v, w, c, d)
            for (tu, tv, tw), cc in d["cores"]:
                assert not tight_cuts(tu, tv, tw), (tu, tv, tw)
                assert cc >= 1
                n_core_triples += 1
            lc, mx, nt = lcm_of_tight_lr(u, v, w)
            assert c % lc == 0, ("(7.2) FAILED", u, v, w, c, lc)
            assert mx <= c
            n_dec += 1
    print(f"[ok] decompose invariant + core minimality + lcm law on "
          f"{n_dec} triples ({n_core_triples} core leaves)")

    # the documented minimal S5 example: u = v = 13524, w = 246135, c = 2
    u = v = (1, 3, 5, 2, 4)
    w = (2, 4, 6, 1, 3, 5)
    d = decompose(u, v, w, 2)
    assert d["product_ok"]
    print(f"[ok] S5 example c=2: bases {[b[3] for b in d['bases']]}, "
          f"cores {[cc for _, cc in d['cores']]}")

    # direct sums multiply Schubert constants (EXPERT_RESPONSE_3 §8 Prop.):
    # c_{a⊕a', b⊕b'}^{d⊕d'} = c_{a,b}^d · c_{a',b'}^{d'}, checked on random
    # support-triple pairs with independent full products.
    rng = random.Random(11)
    n_ds = 0
    smalls = []
    for n in (3, 4):
        vals = list(range(1, n + 1))
        for _ in range(30):
            a = tuple(rng.sample(vals, n))
            b = tuple(rng.sample(vals, n))
            exp = sage_schub_expand(canonical(a), canonical(b))
            ds = [(a, b, d0, c0) for d0, c0 in exp.items()
                  if len(d0) <= n + 1]
            if ds:
                smalls.append(rng.choice(ds))
    for _ in range(60):
        (a, b, d1, c1) = rng.choice(smalls)
        (a2, b2, d2, c2) = rng.choice(smalls)
        p = max(len(a), len(b), len(d1))
        U = direct_sum(pad(a, p), a2)
        V = direct_sum(pad(b, p), b2)
        W = direct_sum(pad(d1, p), d2)
        cUVW = schub_coeff(U, V, W)
        assert cUVW == c1 * c2, (a, b, d1, c1, a2, b2, d2, c2, cUVW)
        if len(canonical(W)) > p:
            assert p in direct_sum_splits(U, V, W), (U, V, W, p)
        n_ds += 1
    print(f"[ok] direct-sum product law c = c1*c2 on {n_ds} random pairs")

    # the expert's S12 two-core counterexample (EXPERT_RESPONSE_3 §8.1)
    A6 = (2, 4, 1, 3, 6, 5)
    B6 = (4, 2, 3, 6, 1, 5)
    D6 = (6, 4, 2, 5, 1, 3)
    assert schub_coeff(A6, B6, D6) == 2
    assert tight_cuts(A6, B6, D6) == []          # terminal core
    assert cut_slacks(A6, B6, D6) == [1, 1, 2, 2, 1]
    U12 = direct_sum(A6, A6)
    V12 = direct_sum(B6, B6)
    W12 = direct_sum(D6, D6)
    assert direct_sum_splits(U12, V12, W12) == [6]
    assert not is_direct_sum_primitive(U12, V12, W12)
    c12 = schub_coeff(U12, V12, W12)
    assert c12 == 4, c12
    d = decompose(U12, V12, W12, c12)
    nontriv = [cc for _, cc in d["cores"] if cc >= 2]
    assert d["product_ok"] and sorted(nontriv) == [2, 2], d
    print("[ok] S12 counterexample: c = 4 = 2*2, TWO nontrivial terminal "
          "cores — the at-most-one-core law is disproved in-pipeline")

    print("richmond selftest passed")


if __name__ == "__main__":
    _selftest()
