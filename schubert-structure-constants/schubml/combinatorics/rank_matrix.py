"""Rank matrices for permutations.

The rank function (see PROJECT.md 1.1) is

    r_w(i, j) = #{a <= i : w(a) <= j},   0 <= i, j <= n.

It counts permutation dots in the upper-left ``i x j`` window.  Rank matrices
are central to Bruhat order and to Schubert-variety incidence.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def rank_matrix(w: Sequence[int]) -> np.ndarray:
    """Full ``(n+1) x (n+1)`` rank matrix ``R[i, j] = r_w(i, j)`` (0-indexed i, j).

    ``R[i, j]`` counts ``{a <= i : w(a) <= j}`` for ``0 <= i, j <= n``.  Row/col 0
    are all zero.  Computed by a 2D prefix sum over the permutation matrix.
    """
    n = len(w)
    P = np.zeros((n + 1, n + 1), dtype=np.int64)
    for a in range(1, n + 1):
        P[a, w[a - 1]] = 1
    # inclusive 2D prefix sum
    R = P.cumsum(axis=0).cumsum(axis=1)
    return R


def rank_at(w: Sequence[int], i: int, j: int) -> int:
    """``r_w(i, j)`` for a single ``0 <= i, j <= n`` without building the matrix."""
    return sum(1 for a in range(i) if w[a] <= j)


def essential_set(w: Sequence[int]) -> List[Tuple[int, int]]:
    """Fulton's essential set: the SE corners of the Rothe diagram.

    A cell ``(i, j)`` of the Rothe diagram ``D(w)`` is essential iff neither
    ``(i+1, j)`` nor ``(i, j+1)`` lies in ``D(w)``.  Returned as 1-indexed
    ``(row, col)`` pairs.  The essential set together with the rank values
    ``r_w`` on it determines ``w`` (Fulton's rank conditions).
    """
    from .rothe import rothe_cells

    cells = set(rothe_cells(w))
    ess = [
        (i, j)
        for (i, j) in cells
        if (i + 1, j) not in cells and (i, j + 1) not in cells
    ]
    ess.sort()
    return ess


def essential_rank_conditions(w: Sequence[int]) -> List[Tuple[int, int, int]]:
    """``(i, j, r_w(i, j))`` for each essential cell — the defining rank conditions."""
    return [(i, j, rank_at(w, i, j)) for (i, j) in essential_set(w)]


def rank_matrix_sparse(w: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Sparse encoding of the essential rank data as ``(i, j, r)`` triples.

    We store the essential set rank conditions (a compact, ``w``-determining
    summary) rather than the dense ``n^2`` matrix.
    """
    return essential_rank_conditions(w)
