"""Reduced words and braid/commutation moves.

A reduced word for ``w`` is a sequence ``(a_1, ..., a_l)`` of simple-reflection
indices with ``l = length(w)`` such that ``w = s_{a_1} s_{a_2} ... s_{a_l}``,
where products use the right-multiplication convention: building up from the
identity, each letter ``a`` applies ``right_mult_s(., a)`` (swap positions
``a, a+1``).
"""

from __future__ import annotations

import random
from typing import Iterator, List, Sequence, Tuple

from .permutations import (
    Perm,
    identity,
    is_identity,
    length,
    right_descents,
    right_mult_s,
)


def word_to_perm(word: Sequence[int], n: int) -> Perm:
    """Reconstruct the permutation ``s_{a_1} ... s_{a_l}`` in ``S_n``."""
    w = identity(n)
    for a in word:
        w = right_mult_s(w, a)
    return w


def is_reduced_word(word: Sequence[int], w: Sequence[int]) -> bool:
    """True iff ``word`` is a reduced word for ``w``."""
    n = len(w)
    if len(word) != length(w):
        return False
    return word_to_perm(word, n) == tuple(w)


def canonical_reduced_word(w: Sequence[int]) -> List[int]:
    """A canonical reduced word via greedy right-descent reduction.

    Repeatedly take the *largest* right descent ``i`` and multiply by ``s_i`` to
    decrease length; the recorded moves, reversed, form a reduced word.
    """
    cur = list(w)
    moves: List[int] = []
    while not is_identity(cur):
        desc = right_descents(cur)
        i = desc[-1]  # largest descent -> lexicographically nice & deterministic
        moves.append(i)
        cur = list(right_mult_s(cur, i))
    return list(reversed(moves))


def random_reduced_word(w: Sequence[int], rng: random.Random | None = None) -> List[int]:
    """A uniform-ish random reduced word via random descent reduction."""
    rng = rng or random
    cur = list(w)
    moves: List[int] = []
    while not is_identity(cur):
        desc = right_descents(cur)
        i = rng.choice(desc)
        moves.append(i)
        cur = list(right_mult_s(cur, i))
    return list(reversed(moves))


def random_reduced_words(w: Sequence[int], k: int, rng: random.Random | None = None) -> List[List[int]]:
    """Up to ``k`` distinct random reduced words for ``w``."""
    rng = rng or random.Random(0)
    seen = set()
    out: List[List[int]] = []
    attempts = 0
    target = max(0, k)
    while len(out) < target and attempts < 50 * (k + 1):
        word = tuple(random_reduced_word(w, rng))
        attempts += 1
        if word not in seen:
            seen.add(word)
            out.append(list(word))
    return out


def commutation_move(word: Sequence[int], pos: int) -> List[int] | None:
    """Apply a commutation move ``s_a s_b = s_b s_a`` (``|a-b| >= 2``) at ``pos``.

    Returns the new word, or ``None`` if no commutation applies at ``pos``.
    """
    word = list(word)
    if pos < 0 or pos + 1 >= len(word):
        return None
    a, b = word[pos], word[pos + 1]
    if abs(a - b) >= 2:
        word[pos], word[pos + 1] = b, a
        return word
    return None


def braid_move(word: Sequence[int], pos: int) -> List[int] | None:
    """Apply a braid move ``s_a s_b s_a = s_b s_a s_b`` (``|a-b| = 1``) at ``pos``.

    Returns the new word, or ``None`` if no braid move applies at ``pos``.
    """
    word = list(word)
    if pos < 0 or pos + 2 >= len(word):
        return None
    a, b, c = word[pos], word[pos + 1], word[pos + 2]
    if a == c and abs(a - b) == 1:
        word[pos], word[pos + 1], word[pos + 2] = b, a, b
        return word
    return None


def all_reduced_words(w: Sequence[int], limit: int | None = None) -> List[Tuple[int, ...]]:
    """All reduced words for ``w`` via BFS over braid+commutation classes.

    ``limit`` caps the number returned (useful for large length).  Exact and
    exhaustive for small permutations.
    """
    start = tuple(canonical_reduced_word(w))
    seen = {start}
    frontier = [start]
    out = [start]
    while frontier:
        nxt = []
        for word in frontier:
            for pos in range(len(word)):
                for mover in (commutation_move, braid_move):
                    nw = mover(word, pos)
                    if nw is not None:
                        t = tuple(nw)
                        if t not in seen:
                            seen.add(t)
                            out.append(t)
                            nxt.append(t)
                            if limit is not None and len(out) >= limit:
                                return out
        frontier = nxt
    return out
