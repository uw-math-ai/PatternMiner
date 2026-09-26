"""Bruhat order on the symmetric group.

We use the rank-matrix criterion: for ``u, w`` (padded to a common size),

    u <= w   <=>   r_u(i, j) >= r_w(i, j)   for all i, j.

(The identity has the maximal rank matrix ``min(i, j)`` and is the unique
minimum, consistent with ``e <= w`` for all ``w``.)
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

from .permutations import (
    length,
    pad,
    right_mult_t,
)
from .rank_matrix import rank_matrix


def bruhat_leq(u: Sequence[int], w: Sequence[int]) -> bool:
    """True iff ``u <= w`` in (strong) Bruhat order (trailing fixed points ok)."""
    n = max(len(u), len(w))
    u = pad(u, n)
    w = pad(w, n)
    Ru = rank_matrix(u)
    Rw = rank_matrix(w)
    return bool((Ru >= Rw).all())


def bruhat_lt(u: Sequence[int], w: Sequence[int]) -> bool:
    return (length(u) < length(w)) and bruhat_leq(u, w)


def bruhat_covers(u: Sequence[int], w: Sequence[int]) -> bool:
    """True iff ``u <| w`` (``u < w`` and ``length(w) = length(u) + 1``)."""
    n = max(len(u), len(w))
    return length(pad(w, n)) == length(pad(u, n)) + 1 and bruhat_leq(u, w)


def bruhat_covers_up(w: Sequence[int]) -> List[Tuple[int, int]]:
    """Position pairs ``(a, b)`` with ``a < b`` s.t. ``w * t_{a,b}`` covers ``w``.

    ``w t_{ab}`` (swap positions ``a, b``) covers ``w`` iff ``w(a) < w(b)`` and no
    ``c`` with ``a < c < b`` has ``w(a) < w(c) < w(b)`` (the standard cover
    criterion in Bruhat order).
    """
    n = len(w)
    covers = []
    for a in range(n):
        for b in range(a + 1, n):
            if w[a] < w[b]:
                # check nothing strictly between in both position and value
                if not any(w[a] < w[c] < w[b] for c in range(a + 1, b)):
                    covers.append((a + 1, b + 1))
    return covers


def cover_transpositions_up(w: Sequence[int]) -> List[Tuple[int, ...]]:
    """The permutations covering ``w`` (one step up in Bruhat order)."""
    return [right_mult_t(w, a, b) for (a, b) in bruhat_covers_up(w)]


def bruhat_interval(u: Sequence[int], w: Sequence[int]) -> List[Tuple[int, ...]]:
    """All permutations ``x`` with ``u <= x <= w`` (brute force; small ``n`` only)."""
    from .permutations import all_permutations, pad

    n = max(len(u), len(w))
    u = pad(u, n)
    w = pad(w, n)
    out = []
    for x in all_permutations(n):
        if bruhat_leq(u, x) and bruhat_leq(x, w):
            out.append(x)
    return out


def bruhat_chains(u: Sequence[int], w: Sequence[int]) -> Iterator[List[Tuple[int, ...]]]:
    """Yield all saturated chains ``u = x_0 <| x_1 <| ... <| x_k = w``."""
    from .permutations import pad

    n = max(len(u), len(w))
    u = pad(u, n)
    w = pad(w, n)
    lw = length(w)

    def _rec(x, chain):
        if x == w:
            yield list(chain)
            return
        for (a, b) in bruhat_covers_up(x):
            y = right_mult_t(x, a, b)
            if length(y) <= lw and bruhat_leq(y, w):
                chain.append(y)
                yield from _rec(y, chain)
                chain.pop()

    if bruhat_leq(u, w):
        yield from _rec(u, [u])
