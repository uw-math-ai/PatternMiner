"""Known permutation families and independent structure-constant rules.

This module provides:

* permutation *family tags* (grassmannian, vexillary, boolean, dominant, ...),
* a Monk/Chevalley independent solver (one factor a simple reflection),
* a Littlewood--Richardson coefficient computation used to validate
  Grassmannian x Grassmannian products independently of Sage.

These are used for the known-rule "islands", curriculum, auxiliary targets,
and as a second backend for data validation (PROJECT.md sections 4, 6.3, 7.8).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .permutations import (
    Perm,
    avoids_pattern,
    canonical_perm,
    contains_pattern,
    inverse,
    is_identity,
    lehmer_code,
    length,
    n_right_descents,
    right_descents,
    right_mult_t,
)
from .bruhat import bruhat_covers_up


# ---------------------------------------------------------------------------
# Family predicates
# ---------------------------------------------------------------------------
def is_simple_reflection(w: Sequence[int]) -> bool:
    return length(w) == 1


def is_grassmannian(w: Sequence[int]) -> bool:
    """At most one right descent."""
    return n_right_descents(w) <= 1


def grassmannian_descent(w: Sequence[int]) -> Optional[int]:
    """The unique descent position, or ``None`` (identity or non-Grassmannian)."""
    d = right_descents(w)
    if len(d) == 1:
        return d[0]
    return None


def is_inverse_grassmannian(w: Sequence[int]) -> bool:
    """Inverse has at most one descent (<=1 left descent)."""
    return is_grassmannian(inverse(w))


def is_dominant(w: Sequence[int]) -> bool:
    """132-avoiding <=> Lehmer code is a partition (weakly decreasing)."""
    code = lehmer_code(w)
    return all(code[i] >= code[i + 1] for i in range(len(code) - 1))


def is_vexillary(w: Sequence[int]) -> bool:
    """2143-avoiding."""
    return avoids_pattern(w, (2, 1, 4, 3))


def is_321_avoiding(w: Sequence[int]) -> bool:
    return avoids_pattern(w, (3, 2, 1))


def is_fully_commutative(w: Sequence[int]) -> bool:
    """In type A, fully commutative == 321-avoiding."""
    return is_321_avoiding(w)


def is_boolean(w: Sequence[int]) -> bool:
    """Boolean (Tenner): avoids 321 and 3412."""
    return avoids_pattern(w, (3, 2, 1)) and avoids_pattern(w, (3, 4, 1, 2))


def is_2143_avoiding(w: Sequence[int]) -> bool:
    return avoids_pattern(w, (2, 1, 4, 3))


def is_two_descent(w: Sequence[int]) -> bool:
    return n_right_descents(w) == 2


def family_tags(w: Sequence[int]) -> List[str]:
    """All applicable family tags for a single permutation (PROJECT.md 7.8)."""
    tags: List[str] = []
    if is_identity(w):
        tags.append("identity")
    if is_simple_reflection(w):
        tags.append("simple_reflection")
    if is_grassmannian(w):
        tags.append("grassmannian")
    if is_inverse_grassmannian(w):
        tags.append("inverse_grassmannian")
    if is_vexillary(w):
        tags.append("vexillary")
    if is_boolean(w):
        tags.append("boolean")
    if is_dominant(w):
        tags.append("dominant")
    if is_fully_commutative(w):
        tags.append("fully_commutative")
    if is_321_avoiding(w):
        tags.append("321_avoiding")
    if is_2143_avoiding(w):
        tags.append("2143_avoiding")
    if is_two_descent(w):
        tags.append("two_descent")
    return tags


def product_family_tags(u: Sequence[int], v: Sequence[int]) -> List[str]:
    """Tags describing the *product* pair (u, v)."""
    tags: List[str] = []
    du = grassmannian_descent(u)
    dv = grassmannian_descent(v)
    if is_simple_reflection(u) or is_simple_reflection(v):
        tags.append("monk_case")
        tags.append("chevalley_case")
    if is_grassmannian(u) and is_grassmannian(v):
        tags.append("grassmannian_grassmannian")
        if du is not None and dv is not None and du == dv:
            tags.append("grassmannian_lr")
            tags.append("schur_times_schubert")
    if is_grassmannian(u) and grassmannian_partition(u) is not None and _is_pieri_shape(
        grassmannian_partition(u)
    ):
        tags.append("pieri_case")
    if is_grassmannian(v) and grassmannian_partition(v) is not None and _is_pieri_shape(
        grassmannian_partition(v)
    ):
        tags.append("pieri_case")
    if is_boolean(u) and is_boolean(v):
        tags.append("boolean")
    if is_vexillary(u) and is_vexillary(v):
        tags.append("vexillary")
    return sorted(set(tags))


# ---------------------------------------------------------------------------
# Grassmannian <-> partition
# ---------------------------------------------------------------------------
def grassmannian_partition(w: Sequence[int]) -> Optional[Tuple[int, ...]]:
    """Partition ``lambda`` of a Grassmannian permutation (descent at ``r``).

    ``lambda = reverse(code[:r])`` (weakly decreasing).  Returns ``None`` for the
    identity or non-Grassmannian permutations.
    """
    if not is_grassmannian(w):
        return None
    r = grassmannian_descent(w)
    if r is None:
        return tuple()  # identity -> empty partition
    code = lehmer_code(w)
    lam = tuple(sorted(code[:r], reverse=True))
    # strip trailing zeros
    while lam and lam[-1] == 0:
        lam = lam[:-1]
    return lam


def _is_pieri_shape(lam: Tuple[int, ...]) -> bool:
    """A single row (h_k) or single column (e_k) partition."""
    if not lam:
        return False
    return len(lam) == 1 or all(p == 1 for p in lam)


# ---------------------------------------------------------------------------
# Monk / Chevalley independent solver
# ---------------------------------------------------------------------------
def monk_product(r: int, w: Sequence[int]) -> Dict[Perm, int]:
    """``S_{s_r} * S_w`` via Monk's rule, returned as ``{canonical_w': coeff}``.

    Monk: ``S_{s_r} S_w = sum S_{w t_{ab}}`` over ``a <= r < b`` with
    ``length(w t_{ab}) = length(w) + 1`` (a Bruhat cover).
    """
    out: Dict[Perm, int] = {}
    for (a, b) in bruhat_covers_up(w):
        if a <= r < b:
            wp = canonical_perm(right_mult_t(w, a, b))
            out[wp] = out.get(wp, 0) + 1
    return out


def monk_product_perm(s_index: int, w: Sequence[int]) -> Dict[Perm, int]:
    """Alias: ``s_index`` is the simple-reflection index ``r``."""
    return monk_product(s_index, w)


# ---------------------------------------------------------------------------
# Littlewood--Richardson coefficients (Grassmannian x Grassmannian)
# ---------------------------------------------------------------------------
def _contains_partition(nu: Tuple[int, ...], lam: Tuple[int, ...]) -> bool:
    lam = lam + (0,) * (len(nu) - len(lam))
    if len(lam) > len(nu):
        return False
    return all(nu[i] >= lam[i] for i in range(len(nu)))


@lru_cache(maxsize=4096)
def lr_coefficient(lam: Tuple[int, ...], mu: Tuple[int, ...], nu: Tuple[int, ...]) -> int:
    """Littlewood--Richardson coefficient ``c^{nu}_{lam, mu}``.

    Counts semistandard skew tableaux of shape ``nu / lam`` with content ``mu``
    whose *reverse reading word* (rows top-to-bottom, each read right-to-left) is
    a lattice (ballot) word.  Pure Python backtracking, exact for small shapes.

    Cells are filled in reverse-reading-word order so the ballot condition can be
    enforced incrementally: when we place value ``val`` the running count must
    satisfy ``#val <= #(val-1)``.  SSYT constraints (rows weakly increasing,
    columns strictly increasing) are checked against already-placed neighbours.
    """
    lam = _trim(lam)
    mu = _trim(mu)
    nu = _trim(nu)
    if sum(lam) + sum(mu) != sum(nu):
        return 0
    if not _contains_partition(nu, lam):
        return 0
    if not mu:
        # nu/lam must be empty; equality already implies sums match
        return 1 if nu == lam else 0

    nrows = len(nu)
    lam_p = list(lam) + [0] * (nrows - len(lam))
    # reverse reading order: top-to-bottom rows, right-to-left within each row
    cells: List[Tuple[int, int]] = []
    for i in range(nrows):
        for j in range(nu[i] - 1, lam_p[i] - 1, -1):
            cells.append((i, j))

    mu_len = len(mu)
    tableau: Dict[Tuple[int, int], int] = {}
    content = [0] * (mu_len + 2)  # 1-indexed letters

    def rec(idx: int) -> int:
        if idx == len(cells):
            return 1
        i, j = cells[idx]
        total = 0
        for val in range(1, mu_len + 1):
            if content[val] >= mu[val - 1]:
                continue
            # column-strict: cell above must be strictly smaller
            above = tableau.get((i - 1, j))
            if above is not None and above >= val:
                continue
            # row weakly increasing: cell to the right (already placed) must be >= val
            right = tableau.get((i, j + 1))
            if right is not None and right < val:
                continue
            # lattice/ballot in reverse reading order
            if val > 1 and content[val] + 1 > content[val - 1]:
                continue
            tableau[(i, j)] = val
            content[val] += 1
            total += rec(idx + 1)
            content[val] -= 1
            del tableau[(i, j)]
        return total

    return rec(0)


def _trim(p: Tuple[int, ...]) -> Tuple[int, ...]:
    p = tuple(int(x) for x in p)
    while p and p[-1] == 0:
        p = p[:-1]
    return p


def grassmannian_lr_product(u: Sequence[int], v: Sequence[int], n: int) -> Optional[Dict[Perm, int]]:
    """Independent LR computation of ``S_u * S_v`` when both are Grassmannian
    with the *same* descent ``r``.  Returns ``{canonical_w: coeff}`` or ``None``
    if the rule does not apply.
    """
    if not (is_grassmannian(u) and is_grassmannian(v)):
        return None
    ru, rv = grassmannian_descent(u), grassmannian_descent(v)
    if ru is None or rv is None or ru != rv:
        return None
    r = ru
    lam = grassmannian_partition(u)
    mu = grassmannian_partition(v)
    if lam is None or mu is None:
        return None
    out: Dict[Perm, int] = {}
    # candidate nu: partitions with |nu| = |lam|+|mu| and at most r rows,
    # parts at most n-r (so that the Grassmannian perm fits in S_n).
    from .partitions import partitions_in_box

    target = sum(lam) + sum(mu)
    for nu in partitions_in_box(target, r, n - r):
        c = lr_coefficient(lam, mu, nu)
        if c:
            w = partition_to_grassmannian(nu, r, n)
            out[canonical_perm(w)] = c
    return out


def partition_to_grassmannian(lam: Sequence[int], r: int, n: int) -> Perm:
    """Grassmannian permutation in ``S_n`` with descent ``r`` and partition ``lam``.

    Inverse of :func:`grassmannian_partition`.  With ``lam`` weakly decreasing of
    length ``<= r``, the first ``r`` (increasing) values are ``{lam[i] + (r - i) :
    i = 0..r-1}`` and the remaining positions hold the unused values in order.
    """
    lam = list(lam) + [0] * (r - len(lam))
    first = sorted(lam[i] + (r - i) for i in range(r))
    used = set(first)
    rest = [x for x in range(1, n + 1) if x not in used]
    return tuple(first + rest)
