"""Pure-Python Schubert-basis expansion via lex-min peeling.

Independent of Sage.  Uses two facts:
  (1) every monomial of S_w arises from a pipe dream, and the bottom pipe dream
      gives x^{code(w)} with coefficient 1, all other dreams having lex-larger
      weight (ladder moves shift weight to earlier variables) — so the
      lex-minimal monomial of S_w is exactly x^{code(w)};
  (2) in a nonnegative combination P = sum c_{w'} S_{w'} there is no
      cancellation, so the lex-min monomial m* of P equals code(w'') for a
      unique w'' and [x^{m*}]P = c_{w''}.
Peeling m* repeatedly yields the full expansion.  Everything is exact integer
arithmetic on monomial dicts.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from . import permutations as P
from ..interp.pipedream_features import schubert_poly_from_dreams

Mono = Tuple[int, ...]


def _trim(m: Sequence[int]) -> Mono:
    m = tuple(int(x) for x in m)
    while m and m[-1] == 0:
        m = m[:-1]
    return m


def poly_mult(A: Dict[Mono, int], B: Dict[Mono, int]) -> Dict[Mono, int]:
    out: Dict[Mono, int] = {}
    for a, ca in A.items():
        for b, cb in B.items():
            n = max(len(a), len(b))
            m = tuple((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
                      for i in range(n))
            m = _trim(m)
            out[m] = out.get(m, 0) + ca * cb
    return {k: v for k, v in out.items() if v != 0}


def schubert_poly(w: Sequence[int]) -> Dict[Mono, int]:
    return { _trim(k): v for k, v in schubert_poly_from_dreams(tuple(w)).items() }


def _lex_min(P_: Dict[Mono, int]) -> Mono:
    # lex compare with implicit trailing zeros
    def key(m: Mono):
        return m + (0,) * (32 - len(m))
    return min(P_.keys(), key=key)


def schubert_expand(P_: Dict[Mono, int], max_terms: int = 100000) -> Dict[Tuple[int, ...], int]:
    """Expand a polynomial (known to be Schubert-positive) in the Schubert basis.

    Returns {canonical_perm: coefficient}.  Raises if a negative coefficient
    appears (i.e. the input was not Schubert-positive / peeling failed).
    """
    work = dict(P_)
    out: Dict[Tuple[int, ...], int] = {}
    for _ in range(max_terms):
        if not work:
            return out
        m = _lex_min(work)
        c = work[m]
        if c < 0:
            raise ValueError(f"negative coefficient {c} at {m}: not Schubert-positive")
        # pad the code with zeros so it is a valid Lehmer code (c_i <= n-i)
        need = max((i + 1 + c for i, c in enumerate(m)), default=1)
        code = list(m) + [0] * (need - len(m))
        wp = P.canonical_perm(P.code_to_perm(code))
        S = schubert_poly(wp)
        for k, v in S.items():
            nk = work.get(k, 0) - c * v
            if nk == 0:
                work.pop(k, None)
            else:
                work[k] = nk
        out[wp] = out.get(wp, 0) + c
    raise RuntimeError("max_terms exceeded")


def schubert_product_expand(u: Sequence[int], v: Sequence[int]) -> Dict[Tuple[int, ...], int]:
    """Full Schubert expansion of S_u * S_v, pure Python."""
    return schubert_expand(poly_mult(schubert_poly(u), schubert_poly(v)))
