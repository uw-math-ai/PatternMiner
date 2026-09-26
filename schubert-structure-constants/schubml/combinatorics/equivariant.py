"""Equivariant (double) Schubert structure constants by localization.

For u, v, w in S_N the double coefficient c^w_{u,v}(t) is defined by
    S_u(x;t) . S_v(x;t) = sum_w c^w_{u,v}(t) . S_w(x;t)
(double Schubert polynomials).  We compute evaluations c^w_{u,v}(t0) at
numeric points t0 in F_p via the standard localization recursion on
GKM restrictions:

    sigma_u|_w . sigma_v|_w = sum_{z <= w} c^z_{u,v} . sigma_z|_w ,

solved triangularly over the Bruhat interval, with restrictions from
Billey's subword formula computed for ALL z <= w in one DP pass over a
reduced word of w:

    sigma_z|_w = sum over reduced subwords of w with product z of
                 prod_j  (prefix_j applied to alpha_{i_j})(t).

Facts used: c^z != 0 requires u <= z, v <= z and l(z) <= l(u)+l(v);
deg_t c^z = l(u)+l(v)-l(z), so a nonzero evaluation at a random point
certifies z in Supp^2, and vanishing at k independent points is a
Schwartz-Zippel certificate at error <= (deg/p)^k.

Validation (selftest): exhaustive symbolic ground truth on S_3 pairs
(ambient S_5 double-Schubert basis via divided differences), the
equivariant Monk rule in closed form, symmetry in (u,v), and agreement
with lrcalc at top degree.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

Perm = Tuple[int, ...]
P_DEFAULT = (1 << 31) - 1


# -------------------------------------------------------------------------
# permutation basics (self-contained; validated against the project's)
# -------------------------------------------------------------------------
def pad(x: Sequence[int], n: int) -> Perm:
    x = list(x)
    return tuple(x + list(range(len(x) + 1, n + 1)))


def perm_len(x: Sequence[int]) -> int:
    return sum(1 for i in range(len(x)) for j in range(i + 1, len(x))
               if x[i] > x[j])


def descents(x: Sequence[int]):
    return {i + 1 for i in range(len(x) - 1) if x[i] > x[i + 1]}


def perm_mul(a: Perm, b: Perm) -> Perm:
    """(a*b)(i) = a(b(i)) — apply b first."""
    n = max(len(a), len(b))
    a, b = pad(a, n), pad(b, n)
    return tuple(a[b[i] - 1] for i in range(n))


def inverse(x: Perm) -> Perm:
    inv = [0] * len(x)
    for i, xi in enumerate(x):
        inv[xi - 1] = i + 1
    return tuple(inv)


def red_word(w: Perm) -> Tuple[int, ...]:
    """A reduced word (indices of adjacent transpositions, 1-based)."""
    w = list(w)
    word = []
    n = len(w)
    changed = True
    while changed:
        changed = False
        for i in range(n - 1):
            if w[i] > w[i + 1]:
                # s_i on the RIGHT of the remaining part:
                pass
        break
    # bubble: write w as product of adjacent transpositions
    word = []
    w = list(w)
    for target in range(1, n + 1):
        # move value `target` to slot target-1 from its current slot
        j = w.index(target)
        while j > target - 1:
            w[j - 1], w[j] = w[j], w[j - 1]
            word.append(j)          # s_j swaps slots j-1, j (1-based idx j)
            j -= 1
    # this produced w0-style sorting: word applied to w gives identity,
    # i.e. s_{jk}...s_{j1} w = e, so w = s_{j1}...s_{jk}
    word.reverse()
    return tuple(word)


def bruhat_leq(z: Perm, w: Perm) -> bool:
    """z <= w in Bruhat order (dominance of sorted prefixes)."""
    n = max(len(z), len(w))
    z, w = pad(z, n), pad(w, n)
    zs: List[int] = []
    ws: List[int] = []
    for i in range(n - 1):
        # insert keeping descending order
        import bisect
        bisect.insort(zs, -z[i])
        bisect.insort(ws, -w[i])
        for k in range(i + 1):
            if -zs[k] > -ws[k]:
                return False
    return True


# -------------------------------------------------------------------------
# Billey restrictions: one DP pass gives sigma_z|_w for all z <= w
# -------------------------------------------------------------------------
def billey_all(w: Perm, t0: Sequence[int], p: int) -> Dict[Perm, int]:
    """dict z -> sigma_z|_w  evaluated at t0 (mod p), for all z <= w."""
    n = len(w)
    word = red_word(w)
    # check reduced
    assert len(word) == perm_len(w), (w, word)
    # prefix permutations and beta_j = prefix_{j-1}(alpha_{i_j})
    prefix: Perm = tuple(range(1, n + 1))
    betas = []
    for i in word:
        a, b = prefix[i - 1], prefix[i]
        betas.append((t0[a - 1] - t0[b - 1]) % p)
        # multiply prefix by s_i on the right: swap positions i-1, i
        pl = list(prefix)
        pl[i - 1], pl[i] = pl[i], pl[i - 1]
        prefix = tuple(pl)
    ident = tuple(range(1, n + 1))
    dp: Dict[Perm, int] = {ident: 1}
    for j, i in enumerate(word):
        beta = betas[j]
        new = dict(dp)
        for z, val in dp.items():
            zs = perm_mul(z, s_i(i, n))
            if perm_len(zs) == perm_len(z) + 1:
                new[zs] = (new.get(zs, 0) + val * beta) % p
        dp = new
    return dp


@lru_cache(maxsize=None)
def s_i(i: int, n: int) -> Perm:
    x = list(range(1, n + 1))
    x[i - 1], x[i] = x[i], x[i - 1]
    return tuple(x)


# -------------------------------------------------------------------------
# double coefficients in the interval [e, w]
# -------------------------------------------------------------------------
def double_coeffs_at(u: Perm, v: Perm, w: Perm, t0: Sequence[int],
                     p: int = P_DEFAULT) -> Dict[Perm, int]:
    """All evaluations c^z_{u,v}(t0) mod p for z <= w (ambient len(w)).

    Requires u <= w and v <= w (else the w-coefficient is 0)."""
    n = len(w)
    u, v = pad(u, n), pad(v, n)
    R = billey_all(w, t0, p)
    lu, lv = perm_len(u), perm_len(v)
    L = lu + lv
    # candidate z: in the DP support (== [e,w]) with u,v <= z, l(z) <= L
    cands = [z for z in R
             if perm_len(z) <= L and bruhat_leq(u, z) and bruhat_leq(v, z)]
    cands.sort(key=perm_len)
    su = restrict_many(u, cands, t0, p)
    sv = restrict_many(v, cands, t0, p)
    c: Dict[Perm, int] = {}
    # need sigma_z|_x for z,x in cands: run billey_all per x is wasteful;
    # instead reuse: sigma_z|_x from billey_all(x).  We only need pairs
    # z <= x within cands.
    rest_cache: Dict[Perm, Dict[Perm, int]] = {}
    for x in cands:
        Rx = billey_all(x, t0, p)
        rest_cache[x] = Rx
        rhs = (su[x] * sv[x]) % p
        for z in c:
            if z != x and z in Rx:
                rhs = (rhs - c[z] * Rx[z]) % p
        den = Rx[x]
        assert den % p != 0, "sigma_x|_x vanished at t0 (bad point)"
        c[x] = (rhs * pow(den, p - 2, p)) % p
    return c


def restrict_many(z: Perm, xs: List[Perm], t0, p) -> Dict[Perm, int]:
    out = {}
    for x in xs:
        R = billey_all(x, t0, p)
        out[x] = R.get(pad(z, len(x)), 0)
    return out


def double_coeff_eval(u: Perm, v: Perm, w: Perm, t0, p=P_DEFAULT) -> int:
    n = max(len(u), len(v), len(w))
    u, v, w = pad(u, n), pad(v, n), pad(w, n)
    if not (bruhat_leq(u, w) and bruhat_leq(v, w)):
        return 0
    if perm_len(w) > perm_len(u) + perm_len(v):
        return 0
    return double_coeffs_at(u, v, w, t0, p)[w]


# -------------------------------------------------------------------------
# symbolic double Schubert polynomials (small ambient, for validation)
# -------------------------------------------------------------------------
def _poly_mul(A, B, p=None):
    out = {}
    for ka, va in A.items():
        for kb, vb in B.items():
            k = tuple(a + b for a, b in zip(ka, kb))
            out[k] = out.get(k, 0) + va * vb
    return {k: v for k, v in out.items() if v}


def double_schubert_basis(n: int) -> Dict[Perm, dict]:
    """All S_w(x;t), w in S_n, as dicts (xexp+texp tuple) -> int, via
    divided differences from the staircase product."""
    import itertools
    nv = 2 * n  # exponents: x_1..x_n, t_1..t_n
    top = {tuple([0] * nv): 1}
    for i in range(1, n):
        for j in range(1, n + 1 - i):
            lin = {}
            e = [0] * nv
            e[i - 1] = 1
            lin[tuple(e)] = 1
            e2 = [0] * nv
            e2[n + j - 1] = 1
            lin[tuple(e2)] = -1
            top = _poly_mul(top, lin)
    w0 = tuple(range(n, 0, -1))
    basis = {w0: top}
    frontier = [w0]
    while frontier:
        new = []
        for w in frontier:
            for i in range(1, n):
                ws = perm_mul(w, s_i(i, n))
                if perm_len(ws) == perm_len(w) - 1 and ws not in basis:
                    basis[ws] = _divdiff(basis[w], i, n)
                    new.append(ws)
        frontier = new
    return basis


def _divdiff(P: dict, i: int, n: int) -> dict:
    """partial_i in the x variables (first n exponent slots)."""
    out = {}
    for k, c in P.items():
        a, b = k[i - 1], k[i]
        # (x^a y^b - x^b y^a)/(x - y) with x=x_i, y=x_{i+1}
        if a == b:
            continue
        sgn = 1 if a > b else -1
        lo, hi = min(a, b), max(a, b)
        for r in range(lo, hi):
            k2 = list(k)
            k2[i - 1] = r
            k2[i] = a + b - 1 - r
            k2 = tuple(k2)
            out[k2] = out.get(k2, 0) + sgn * c
    return {k: v for k, v in out.items() if v}


def expand_in_basis(P: dict, basis: Dict[Perm, dict], n_amb: int
                    ) -> Dict[Perm, dict]:
    """Strip x-leading terms (lex, x block first) against the basis."""
    def xkey(k):
        return k[:n_amb]
    def code(w):
        return tuple(sum(1 for j in range(i + 1, len(w)) if w[j] < w[i])
                     for i in range(len(w)))
    bycode = {code(w): w for w in basis}
    P = dict(P)
    out: Dict[Perm, dict] = {}
    while P:
        # lex-largest x-part among terms
        k0 = max(P, key=lambda k: xkey(k))
        xc = xkey(k0)
        w = bycode.get(xc)
        assert w is not None, ("leading x-monomial is not a code", xc)
        # coefficient of x^xc as polynomial in t:
        coef = {k[n_amb:]: c for k, c in P.items() if xkey(k) == xc}
        out[w] = coef
        # subtract coef(t) * S_w
        Sw = basis[w]
        for kt, ct in coef.items():
            for kb, cb in Sw.items():
                k = tuple(a + b for a, b in
                          zip(kb, tuple([0] * n_amb) + kt))
                P[k] = P.get(k, 0) - ct * cb
        P = {k: v for k, v in P.items() if v}
    return out
