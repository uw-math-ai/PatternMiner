"""Integer partitions and Young-diagram helpers (for LR / Grassmannian rules)."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator, List, Tuple


def partitions(n: int, max_part: int | None = None) -> Iterator[Tuple[int, ...]]:
    """All partitions of ``n`` (weakly decreasing), parts ``<= max_part``."""
    if max_part is None:
        max_part = n
    if n == 0:
        yield tuple()
        return
    for first in range(min(n, max_part), 0, -1):
        for rest in partitions(n - first, first):
            yield (first,) + rest


@lru_cache(maxsize=2048)
def partitions_in_box(n: int, max_rows: int, max_cols: int) -> Tuple[Tuple[int, ...], ...]:
    """All partitions of ``n`` fitting in a ``max_rows x max_cols`` box."""
    out: List[Tuple[int, ...]] = []
    for p in partitions(n, max_cols):
        if len(p) <= max_rows:
            out.append(p)
    return tuple(out)


def conjugate(lam: Tuple[int, ...]) -> Tuple[int, ...]:
    """Conjugate (transpose) partition."""
    if not lam:
        return tuple()
    m = lam[0]
    return tuple(sum(1 for p in lam if p > j) for j in range(m))


def contains(nu: Tuple[int, ...], lam: Tuple[int, ...]) -> bool:
    """True iff Young diagram of ``lam`` fits inside ``nu``."""
    if len(lam) > len(nu):
        return False
    return all(nu[i] >= lam[i] for i in range(len(lam)))


def cells(lam: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """0-indexed ``(row, col)`` cells of the Young diagram."""
    return [(i, j) for i, p in enumerate(lam) for j in range(p)]
