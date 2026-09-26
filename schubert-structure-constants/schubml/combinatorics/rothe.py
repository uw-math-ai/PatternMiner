"""Rothe diagrams.

The Rothe diagram of ``w`` is

    D(w) = {(i, j) : w(i) > j  and  w^{-1}(j) > i},   1 <= i, j <= n,

i.e. the cells that remain after deleting, for each dot ``(i, w(i))``, every
cell weakly south or weakly east of it (in the "hook" through the dot).
We have ``|D(w)| = length(w)``, the row lengths equal the Lehmer code, and
``D(w^{-1}) = D(w)^T``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .permutations import inverse, lehmer_code


def rothe_cells(w: Sequence[int]) -> List[Tuple[int, int]]:
    """Cells ``(i, j)`` of ``D(w)`` as 1-indexed pairs, ``|D(w)| = length(w)``."""
    n = len(w)
    winv = inverse(w)
    cells = []
    for i in range(1, n + 1):
        wi = w[i - 1]
        for j in range(1, n + 1):
            if wi > j and winv[j - 1] > i:
                cells.append((i, j))
    return cells


def rothe_row_lengths(w: Sequence[int]) -> List[int]:
    """Number of Rothe cells in each row; equals the Lehmer code of ``w``."""
    # row i length = #{j : w(i) > j, w^{-1}(j) > i} = #{j < w(i) : pos(j) > i}
    # which is exactly the Lehmer code c_i.
    return lehmer_code(w)


def rothe_col_lengths(w: Sequence[int]) -> List[int]:
    """Number of Rothe cells in each column; equals the Lehmer code of ``w^{-1}``."""
    return lehmer_code(inverse(w))


def rothe_shape(w: Sequence[int]) -> List[int]:
    """Sorted (weakly decreasing) row lengths — the *shape* of the diagram."""
    return sorted(rothe_row_lengths(w), reverse=True)


def is_partition_shape(w: Sequence[int]) -> bool:
    """True iff the Lehmer code is weakly decreasing (dominant permutation)."""
    code = lehmer_code(w)
    return all(code[i] >= code[i + 1] for i in range(len(code) - 1))


def rothe_diagram_matrix(w: Sequence[int]):
    """Dense 0/1 numpy matrix of ``D(w)`` (1 where a cell is present)."""
    import numpy as np

    n = len(w)
    M = np.zeros((n, n), dtype=np.int8)
    for (i, j) in rothe_cells(w):
        M[i - 1, j - 1] = 1
    return M


def rothe_ascii(w: Sequence[int], dot: str = "@", cell: str = "x", empty: str = ".") -> str:
    """ASCII rendering of the Rothe diagram with permutation dots marked."""
    n = len(w)
    cells = set(rothe_cells(w))
    rows = []
    for i in range(1, n + 1):
        line = []
        for j in range(1, n + 1):
            if w[i - 1] == j:
                line.append(dot)
            elif (i, j) in cells:
                line.append(cell)
            else:
                line.append(empty)
        rows.append(" ".join(line))
    return "\n".join(rows)
