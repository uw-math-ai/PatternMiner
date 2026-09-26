"""Core permutation combinatorics for Schubert calculus.

Conventions (see PROJECT.md section 1):

* Permutations are stored in **one-line notation**, 1-indexed values:
  ``w = [w(1), w(2), ..., w(n)]`` with values a permutation of ``{1, ..., n}``.
* Internally we represent permutations as tuples of Python ints so that they are
  hashable and usable as dictionary keys / dataframe cells.
* ``compose(p, q)`` is functional composition ``(p . q)(i) = p(q(i))``.
* Right multiplication ``w * s_i`` swaps *positions* ``i, i+1`` of the one-line
  word; left multiplication ``s_i * w`` swaps *values* ``i, i+1``.

All functions accept any iterable of ints and return tuples unless noted.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations as _iter_permutations
from typing import Iterable, List, Sequence, Tuple

Perm = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
def as_perm(w: Iterable[int]) -> Perm:
    """Coerce an iterable to a permutation tuple (no validation)."""
    return tuple(int(x) for x in w)


def is_permutation(w: Sequence[int]) -> bool:
    """True iff ``w`` is a permutation of ``{1, ..., len(w)}``."""
    n = len(w)
    seen = [False] * (n + 1)
    for x in w:
        if not isinstance(x, (int,)) and int(x) != x:
            return False
        x = int(x)
        if x < 1 or x > n or seen[x]:
            return False
        seen[x] = True
    return True


def identity(n: int) -> Perm:
    """Identity permutation of ``S_n``."""
    return tuple(range(1, n + 1))


def is_identity(w: Sequence[int]) -> bool:
    return all(w[i] == i + 1 for i in range(len(w)))


def size(w: Sequence[int]) -> int:
    return len(w)


# ---------------------------------------------------------------------------
# Group operations
# ---------------------------------------------------------------------------
def inverse(w: Sequence[int]) -> Perm:
    """Group inverse ``w^{-1}`` (same size)."""
    n = len(w)
    inv = [0] * n
    for i in range(n):
        inv[w[i] - 1] = i + 1
    return tuple(inv)


def compose(p: Sequence[int], q: Sequence[int]) -> Perm:
    """Functional composition ``(p . q)(i) = p(q(i))``.

    Both permutations are padded to the common (max) size with trailing fixed
    points first, so ``compose`` is well defined across different ambient sizes.
    """
    n = max(len(p), len(q))
    p = pad(p, n)
    q = pad(q, n)
    return tuple(p[q[i] - 1] for i in range(n))


def multiply(*perms: Sequence[int]) -> Perm:
    """Left-to-right product ``perms[0] * perms[1] * ...`` using :func:`compose`."""
    if not perms:
        raise ValueError("multiply requires at least one permutation")
    out = as_perm(perms[0])
    for q in perms[1:]:
        out = compose(out, q)
    return out


# ---------------------------------------------------------------------------
# Padding and canonicalization (trailing fixed points)
# ---------------------------------------------------------------------------
def pad(w: Sequence[int], n: int) -> Perm:
    """Extend ``w`` to ``S_n`` by appending fixed points. ``n >= len(w)``."""
    m = len(w)
    if n < m:
        raise ValueError(f"cannot pad permutation of size {m} down to {n}")
    return as_perm(w) + tuple(range(m + 1, n + 1))


def canonical_perm(w: Sequence[int]) -> Perm:
    """Trim trailing fixed points.

    ``canonical_perm([2,1,3,4,5]) == (2,1)``,
    ``canonical_perm([1,2,3,4])   == (1,)``  (identity collapses to ``(1,)``),
    ``canonical_perm([3,1,2,4])   == (3,1,2)``.
    """
    w = as_perm(w)
    k = len(w)
    while k > 1 and w[k - 1] == k:
        k -= 1
    return w[:k]


def canonical_size(w: Sequence[int]) -> int:
    """Smallest ``n`` such that ``w`` is non-trivial in ``S_n`` (>=1)."""
    return len(canonical_perm(w))


def same_perm(a: Sequence[int], b: Sequence[int]) -> bool:
    """Equality up to trailing fixed points."""
    return canonical_perm(a) == canonical_perm(b)


# ---------------------------------------------------------------------------
# Simple reflections
# ---------------------------------------------------------------------------
def simple_transposition(i: int, n: int) -> Perm:
    """``s_i`` in ``S_n`` (swaps values/positions ``i`` and ``i+1``). 1-indexed ``i``."""
    if i < 1 or i >= n:
        raise ValueError(f"s_{i} undefined in S_{n}")
    w = list(range(1, n + 1))
    w[i - 1], w[i] = w[i], w[i - 1]
    return tuple(w)


def right_mult_s(w: Sequence[int], i: int) -> Perm:
    """``w * s_i``: swap positions ``i, i+1`` of the one-line word (1-indexed)."""
    w = list(w)
    w[i - 1], w[i] = w[i], w[i - 1]
    return tuple(w)


def left_mult_s(w: Sequence[int], i: int) -> Perm:
    """``s_i * w``: swap the values ``i`` and ``i+1`` wherever they appear."""
    return tuple(i + 1 if x == i else (i if x == i + 1 else x) for x in w)


def transposition(a: int, b: int, n: int) -> Perm:
    """Transposition ``t_{a,b}`` swapping positions/values ``a`` and ``b`` in ``S_n``."""
    if not (1 <= a <= n and 1 <= b <= n):
        raise ValueError("transposition indices out of range")
    w = list(range(1, n + 1))
    w[a - 1], w[b - 1] = w[b - 1], w[a - 1]
    return tuple(w)


def right_mult_t(w: Sequence[int], a: int, b: int) -> Perm:
    """``w * t_{a,b}``: swap positions ``a`` and ``b`` of the one-line word."""
    w = list(w)
    w[a - 1], w[b - 1] = w[b - 1], w[a - 1]
    return tuple(w)


# ---------------------------------------------------------------------------
# Length, inversions, descents, code
# ---------------------------------------------------------------------------
def inversions(w: Sequence[int]) -> List[Tuple[int, int]]:
    """Inversion set ``{(i,j): i<j, w(i)>w(j)}`` as 1-indexed position pairs."""
    n = len(w)
    out = []
    for i in range(n):
        wi = w[i]
        for j in range(i + 1, n):
            if wi > w[j]:
                out.append((i + 1, j + 1))
    return out


def length(w: Sequence[int]) -> int:
    """Coxeter length = number of inversions."""
    n = len(w)
    count = 0
    for i in range(n):
        wi = w[i]
        for j in range(i + 1, n):
            if wi > w[j]:
                count += 1
    return count


def right_descents(w: Sequence[int]) -> List[int]:
    """``D_R(w) = {i : w(i) > w(i+1)}`` (1-indexed positions ``i in 1..n-1``)."""
    return [i + 1 for i in range(len(w) - 1) if w[i] > w[i + 1]]


def left_descents(w: Sequence[int]) -> List[int]:
    """``D_L(w) = D_R(w^{-1}) = {i : pos(i) > pos(i+1)}``."""
    return right_descents(inverse(w))


def n_right_descents(w: Sequence[int]) -> int:
    return sum(1 for i in range(len(w) - 1) if w[i] > w[i + 1])


def n_left_descents(w: Sequence[int]) -> int:
    return len(left_descents(w))


def lehmer_code(w: Sequence[int]) -> List[int]:
    """Lehmer/inversion code ``c_i = #{j > i : w(i) > w(j)}`` (length ``n``)."""
    n = len(w)
    code = [0] * n
    for i in range(n):
        wi = w[i]
        c = 0
        for j in range(i + 1, n):
            if wi > w[j]:
                c += 1
        code[i] = c
    return code


def code_to_perm(code: Sequence[int]) -> Perm:
    """Inverse of :func:`lehmer_code`: reconstruct ``w`` from its code."""
    n = len(code)
    available = list(range(1, n + 1))
    w = []
    for c in code:
        # c is the number of available values smaller than w(i) to its right,
        # i.e. w(i) is the (c+1)-th smallest still-available value.
        val = available.pop(c)
        w.append(val)
    return tuple(w)


# ---------------------------------------------------------------------------
# Pattern containment / avoidance
# ---------------------------------------------------------------------------
def contains_pattern(w: Sequence[int], pattern: Sequence[int]) -> bool:
    """True iff ``w`` contains ``pattern`` (classical permutation pattern)."""
    n, k = len(w), len(pattern)
    if k > n:
        return False
    # standardize pattern to relative order ranks
    from itertools import combinations

    prank = _standardize(pattern)
    for idx in combinations(range(n), k):
        if _standardize([w[i] for i in idx]) == prank:
            return True
    return False


def avoids_pattern(w: Sequence[int], pattern: Sequence[int]) -> bool:
    return not contains_pattern(w, pattern)


def _standardize(seq: Sequence[int]) -> Tuple[int, ...]:
    order = sorted(range(len(seq)), key=lambda i: seq[i])
    rank = [0] * len(seq)
    for r, i in enumerate(order):
        rank[i] = r + 1
    return tuple(rank)


# ---------------------------------------------------------------------------
# Enumeration helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def all_permutations(n: int) -> Tuple[Perm, ...]:
    """All ``n!`` permutations of ``S_n`` (cached)."""
    return tuple(_iter_permutations(range(1, n + 1)))


def permutations_by_length(n: int):
    """Dict ``{l: [perms of length l]}`` for ``S_n``."""
    out: dict[int, List[Perm]] = {}
    for w in all_permutations(n):
        out.setdefault(length(w), []).append(w)
    return out


def longest_element(n: int) -> Perm:
    """``w_0`` = reverse permutation ``[n, n-1, ..., 1]``."""
    return tuple(range(n, 0, -1))
