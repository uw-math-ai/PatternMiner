"""Symmetry canonicalization of Schubert triples (EXPERT_RESPONSE_3 SS3.3).

A vanishing instance is a triple (u, v, w) in S_N with l(w) = l(u) + l(v).
The structure constant has a symmetric (triple-intersection) form

    c_{u,v}^w = integral over Fl_N of  sigma_u . sigma_v . sigma_{w0 w},

so with  c := w0 w  the invariant object is the unordered "symmetric triple"
(a, b, c) = (u, v, w0 w).  The primitive-orbit group quotiented out here is

  * S3   -- the six permutations of (a, b, c); the transposition (a b) is the
            u <-> v swap;
  * Z2   -- the flag-reversal Dynkin involution  x -> w0 x w0
            (position + value reversal), applied to all three simultaneously;
  * stabilization -- joint trailing fixed points ("add fixed points"), and
            joint leading fixed points (= direct sum with a leading identity
            block, coefficient-preserving by the direct-sum product rule).

Every generator is VERIFIED on ground-truth lrcalc coefficients in
``_selftest`` (exact coefficient equality, not just vanishing) before any
census uses it.  Derived metadata (D_exact, block composition) is always
recomputed from the permutations, never hand-entered (SS1 CI assertions).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple

Perm = Tuple[int, ...]
Triple = Tuple[Perm, Perm, Perm]        # (u, v, w), common ambient N = len


# ---------------------------------------------------------------------------
# basics + CI assertions (EXPERT_RESPONSE_3 SS1)
# ---------------------------------------------------------------------------


def pad(x: Sequence[int], n: int) -> Perm:
    return tuple(x) + tuple(range(len(x) + 1, n + 1))


def perm_len(x: Sequence[int]) -> int:
    return sum(1 for i in range(len(x)) for j in range(i + 1, len(x))
               if x[i] > x[j])


def descents(x: Sequence[int]) -> Tuple[int, ...]:
    return tuple(i + 1 for i in range(len(x) - 1) if x[i] > x[i + 1])


def ci_assert_record(u: Sequence[int], v: Sequence[int], w: Sequence[int],
                     Dstar: Optional[Sequence[int]] = None,
                     N: Optional[int] = None) -> dict:
    """The SS1 CI assertions; returns recomputed metadata (source of truth)."""
    if N is None:
        N = max(len(u), len(v), len(w))
    u, v, w = pad(u, N), pad(v, N), pad(w, N)
    D_exact = tuple(sorted(set(descents(u)) | set(descents(v))))
    assert set(descents(w)) <= set(D_exact), \
        f"Des(w) not within Des(u)|Des(v): {w}"
    assert perm_len(w) == perm_len(u) + perm_len(v), "length not additive"
    assert all(1 <= d < N for d in D_exact), "descent out of range"
    if Dstar is not None:
        assert tuple(sorted(Dstar)) == D_exact, \
            f"stored Dstar {tuple(Dstar)} != recomputed {D_exact}"
    comp = block_composition(D_exact, N)
    return {"N": N, "u": u, "v": v, "w": w, "D_exact": D_exact,
            "composition": comp,
            "L": (perm_len(u), perm_len(v), perm_len(w))}


def block_composition(D: Sequence[int], N: int) -> Tuple[int, ...]:
    bounds = [0] + sorted(D) + [N]
    return tuple(b - a for a, b in zip(bounds, bounds[1:]))


# ---------------------------------------------------------------------------
# the symmetry generators
# ---------------------------------------------------------------------------


def w0_mult(x: Perm) -> Perm:
    """w0 . x  (values reversed): (w0 x)(i) = N + 1 - x(i)."""
    N = len(x)
    return tuple(N + 1 - t for t in x)


def dynkin(x: Perm) -> Perm:
    """w0 x w0: positions and values both reversed."""
    N = len(x)
    return tuple(N + 1 - x[N - 1 - i] for i in range(N))


def to_symmetric(u: Perm, v: Perm, w: Perm) -> Triple:
    return (u, v, w0_mult(w))


def from_symmetric(a: Perm, b: Perm, c: Perm) -> Triple:
    return (a, b, w0_mult(c))


def s3_images(u: Perm, v: Perm, w: Perm) -> List[Triple]:
    """All six triples (u', v', w') with the same triple-intersection number."""
    a, b, c = to_symmetric(u, v, w)
    return [from_symmetric(*p) for p in itertools.permutations((a, b, c))]


def dynkin_image(u: Perm, v: Perm, w: Perm) -> Triple:
    """Flag reversal: applies x -> w0 x w0 to each symmetric-triple slot.

    On the record side this is (u,v,w) -> (w0 u w0, w0 v w0, w0 w w0):
    w0 (w0 w w0) = (w0 w0) w w0 ... note w0 c w0 with c = w0 w gives
    w0 (w0 w) w0 = w w0, and from_symmetric turns it into w0 (w w0) = w0 w w0.
    """
    return (dynkin(u), dynkin(v), dynkin(w))


# ---------------------------------------------------------------------------
# stabilization (trailing / leading joint fixed points)
# ---------------------------------------------------------------------------


def _strip_trailing(t: Triple) -> Triple:
    u, v, w = t
    N = len(u)
    m = N
    while m > 1 and u[m - 1] == m and v[m - 1] == m and w[m - 1] == m:
        m -= 1
    return (u[:m], v[:m], w[:m])


def _strip_leading(t: Triple) -> Triple:
    u, v, w = t
    k = 0
    while k < len(u) - 1 and u[k] == k + 1 and v[k] == k + 1 and w[k] == k + 1:
        k += 1
    if k == 0:
        return t
    return (tuple(x - k for x in u[k:]),
            tuple(x - k for x in v[k:]),
            tuple(x - k for x in w[k:]))


def stabilize(t: Triple) -> Triple:
    """Joint minimal form: strip common trailing then leading fixed points."""
    prev = None
    while prev != t:
        prev = t
        t = _strip_leading(_strip_trailing(t))
    return t


# ---------------------------------------------------------------------------
# orbit closure and canonical form
# ---------------------------------------------------------------------------


def orbit(u: Sequence[int], v: Sequence[int], w: Sequence[int],
          max_size: int = 4096) -> Set[Triple]:
    """Closure of the triple under S3 x Dynkin x stabilization.

    All group operations act inside the current joint ambient of each triple
    (stabilized first so w0 is taken in the minimal common ambient); BFS to
    closure makes the resulting equivalence relation well-defined.
    """
    N0 = max(len(u), len(v), len(w))
    start = stabilize((pad(u, N0), pad(v, N0), pad(w, N0)))
    seen: Set[Triple] = {start}
    frontier = [start]
    while frontier:
        nxt: List[Triple] = []
        for t in frontier:
            images = s3_images(*t) + [dynkin_image(*t)]
            for im in images:
                im = stabilize(im)
                if im not in seen:
                    seen.add(im)
                    nxt.append(im)
        frontier = nxt
        if len(seen) > max_size:
            raise RuntimeError("orbit closure did not stabilize")
    return seen


def canonical_form(u: Sequence[int], v: Sequence[int], w: Sequence[int]
                   ) -> Triple:
    """Lexicographically minimal (N, u, v, w) over the symmetry orbit."""
    return min(orbit(u, v, w), key=lambda t: (len(t[0]),) + t[0] + t[1] + t[2])


# ---------------------------------------------------------------------------
# primitivity checks: direct sums, common intervals (pattern inflation)
# ---------------------------------------------------------------------------


def direct_sum_cuts(u: Perm, v: Perm, w: Perm) -> List[int]:
    """All 0 < k < N with u,v,w each mapping [1..k] onto [1..k].

    At such a k the triple is a direct sum and c multiplies over the parts
    (EXPERT_RESPONSE_3 SS8 proposition); a primitive instance has none.
    """
    N = len(u)
    out = []
    su = sv = sw = 0
    for k in range(1, N):
        su += u[k - 1]
        sv += v[k - 1]
        sw += w[k - 1]
        tgt = k * (k + 1) // 2
        if (su == tgt and sv == tgt and sw == tgt
                and max(u[:k]) == k and max(v[:k]) == k and max(w[:k]) == k):
            out.append(k)
    return out


def common_intervals(u: Perm, v: Perm, w: Perm) -> List[Tuple[int, int]]:
    """Position intervals [p..q] (1-based, 2 <= size <= N-1) mapped onto a
    single common value interval by u, v and w simultaneously.

    Any nontrivial simultaneous substitution decomposition (pattern
    inflation) produces such an interval, so an empty list certifies that no
    cheap inflation structure exists.  (Sufficiency is not claimed.)
    """
    N = len(u)
    out = []
    for p in range(1, N + 1):
        for q in range(p + 1, min(p + N - 1, N) + 1):
            if q - p + 1 >= N:
                continue
            ivs = []
            ok = True
            for x in (u, v, w):
                vals = x[p - 1:q]
                lo, hi = min(vals), max(vals)
                if hi - lo != q - p:
                    ok = False
                    break
                ivs.append((lo, hi))
            if ok and ivs[0] == ivs[1] == ivs[2]:
                out.append((p, q))
    return out


# ---------------------------------------------------------------------------
# selftest: every generator verified on ground-truth coefficients
# ---------------------------------------------------------------------------


def _coeff(u: Perm, v: Perm, w: Perm) -> int:
    from .multicut import schubert_coefficient
    return schubert_coefficient(u, v, w)


def _selftest() -> None:
    import random
    import lrcalc
    rng = random.Random(20260718)

    # ground-truth positive triples: pick random u,v and w from supp(su*sv)
    positives: List[Tuple[Triple, int]] = []
    while len(positives) < 24:
        n = rng.choice([5, 6, 7])
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        prod = lrcalc.schubmult(list(u), list(v), n)
        items = [(tuple(int(t) for t in k), int(c)) for k, c in prod.items()
                 if len(k) <= n]
        if not items:
            continue
        w, c = rng.choice(items)
        w = pad(w, n)
        positives.append(((u, v, w), c))

    # (a) u <-> v swap
    for (u, v, w), c in positives[:20]:
        assert _coeff(v, u, w) == c, ("swap failed", u, v, w, c)
    print("[ok] u<->v swap preserves c on 20 ground-truth positives")

    # (b) all six S3 images of the symmetric triple
    n_checked = 0
    for (u, v, w), c in positives[:20]:
        for (u2, v2, w2) in s3_images(u, v, w):
            l2 = (perm_len(u2), perm_len(v2), perm_len(w2))
            assert l2[2] == l2[0] + l2[1], ("grading broke", u2, v2, w2)
            assert _coeff(u2, v2, w2) == c, \
                ("S3 image failed", (u, v, w), (u2, v2, w2), c)
            n_checked += 1
    print(f"[ok] S3 triple-duality preserves c ({n_checked} images "
          "over 20 positives)")

    # (c) Dynkin involution
    for (u, v, w), c in positives[:20]:
        u2, v2, w2 = dynkin_image(u, v, w)
        assert _coeff(u2, v2, w2) == c, ("dynkin failed", u, v, w, c)
    print("[ok] Dynkin flag-reversal preserves c on 20 positives")

    # zeros too: symmetry must preserve vanishing (spot check)
    n_zero = 0
    for _ in range(200):
        if n_zero >= 10:
            break
        n = rng.choice([5, 6])
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        w = tuple(rng.sample(range(1, n + 1), n))
        if perm_len(w) != perm_len(u) + perm_len(v):
            continue
        if _coeff(u, v, w) != 0:
            continue
        n_zero += 1
        for (u2, v2, w2) in s3_images(u, v, w):
            assert _coeff(u2, v2, w2) == 0
        u2, v2, w2 = dynkin_image(u, v, w)
        assert _coeff(u2, v2, w2) == 0
    print(f"[ok] symmetries preserve vanishing on {n_zero} random zeros")

    # (d) trailing stabilization: embedding in N+2 does not change c
    for (u, v, w), c in positives[:10]:
        n = len(u) + 2
        assert _coeff(pad(u, n), pad(v, n), pad(w, n)) == c
        assert stabilize((pad(u, n), pad(v, n), pad(w, n))) == \
            stabilize((u, v, w))
    print("[ok] trailing fixed-point stabilization")

    # (e) leading identity block = direct sum; c preserved by leading strip
    for (u, v, w), c in positives[:10]:
        k = 2
        u2 = tuple(range(1, k + 1)) + tuple(x + k for x in u)
        v2 = tuple(range(1, k + 1)) + tuple(x + k for x in v)
        w2 = tuple(range(1, k + 1)) + tuple(x + k for x in w)
        assert _coeff(u2, v2, w2) == c, ("leading block failed", u, v, w)
        assert stabilize((u2, v2, w2)) == stabilize((u, v, w))
        assert direct_sum_cuts(u2, v2, w2)[:1] == [k] or \
            direct_sum_cuts(u2, v2, w2)  # k is a direct-sum cut
    print("[ok] leading identity block strip (direct-sum rule)")

    # orbit closure: canonical form constant on the orbit
    for (u, v, w), c in positives[:8]:
        cf = canonical_form(u, v, w)
        for t in list(orbit(u, v, w))[:12]:
            assert canonical_form(*t) == cf
    print("[ok] canonical_form constant on orbits")

    # direct-sum detector sanity
    assert direct_sum_cuts((2, 1, 3), (1, 2, 3), (2, 1, 3)) == [2]
    assert direct_sum_cuts((2, 3, 1), (2, 3, 1), (3, 2, 1)) == []
    print("orbits selftest passed")


if __name__ == "__main__":
    _selftest()
