"""Select a simple reflection from permutation support inequalities.

Only the Python standard library is used. There are no coefficient routines,
training examples, model activations, or labelled configuration tables here.
"""
from collections import Counter


def _bounds(p):
    """Subset ranks of the flagged column matchings of the Rothe diagram.

    The last row of a permutation's Rothe diagram is empty. The returned
    list is indexed by subsets of the first n-1 rows, encoded as bit masks.
    """
    n = len(p)
    ip = [0] * n
    for r, x in enumerate(p):
        ip[x - 1] = r
    columns = Counter(tuple(r for r in range(ip[c - 1]) if p[r] > c)
                      for c in range(1, n + 1))
    active = max((max(col) + 1 for col in columns if col), default=0)
    total = [0] * (1 << active)
    for col, multiplicity in columns.items():
        if not col:
            continue
        rank, next_slot = [0], [0]
        k, lower = len(col), 0
        for e in range(col[-1] + 1):
            while lower < k and col[lower] < e:
                lower += 1
            size = len(rank)
            for prev in range(size):
                slot = max(lower, next_slot[prev])
                rank.append(rank[prev] + (slot < k))
                next_slot.append(min(k, slot + 1))
        expanded = rank * (len(total) // len(rank))
        total = [a + multiplicity * b for a, b in zip(total, expanded)]
    return total * (1 << (n - 1 - active))


def _candidate_stats(u, b, w, bounds):
    n = len(u)
    triple = (u, b, tuple(n + 1 - x for x in w))
    maximum, mass, tight = 0, 0, 0
    counts = []
    for reflect in (False, True):
        pp = tuple(tuple(n + 1 - x for x in p[::-1]) for p in triple) if reflect else triple
        bb = [bounds(p) for p in pp]
        for k in range(3):
            target = bounds(tuple(n + 1 - x for x in pp[k]))
            zeros = 0
            for a, b0, c in zip(bb[(k + 1) % 3], bb[(k + 2) % 3], target):
                slack = a + b0 - c
                if slack < 0:
                    maximum = max(maximum, -slack)
                    mass -= slack
                elif slack == 0:
                    zeros += 1
            counts.append(zeros)
            tight += zeros
    return maximum, mass, tight, counts


def _transfers(pp):
    """Move a sole right descent to either of the other two factors."""
    for r in range(len(pp[0]) - 1):
        descending = [k for k, p in enumerate(pp) if p[r] > p[r + 1]]
        if len(descending) != 1:
            continue
        a = descending[0]
        for b in range(3):
            if b == a:
                continue
            qq = list(pp)
            for k in (a, b):
                q = list(qq[k])
                q[r], q[r + 1] = q[r + 1], q[r]
                qq[k] = tuple(q)
            yield tuple(qq)


def _common_ascent(pp):
    return any(all(p[r] < p[r + 1] for p in pp)
               for r in range(len(pp[0]) - 1))


def _bounded_zero(pp):
    """A necessary vanishing obstruction, explored to depth three only."""
    pp = tuple(sorted(pp))
    frontier, seen = [pp], {pp}
    for depth in range(4):
        following = []
        for qq in frontier:
            if _common_ascent(qq):
                return True
            if depth == 3:
                continue
            for rr in _transfers(qq):
                rr = tuple(sorted(rr))
                if rr not in seen:
                    seen.add(rr)
                    following.append(rr)
        frontier = following
    return False


def select_j(u, v, w, i, candidates):
    """Return a one-based j. Left multiplication swaps values j and j+1.

    The promised i witnesses existence. The rule checks necessary conditions
    on the proposed second-factor move directly, so it does not otherwise use i.
    """
    if not candidates:
        raise ValueError("candidates must be nonempty")
    u, v, w = tuple(u), tuple(v), tuple(w)
    cache = {}
    def bounds(p):
        if p not in cache:
            cache[p] = _bounds(p)
        return cache[p]
    ranked = []
    for j in sorted(candidates):
        b = tuple(j + 1 if x == j else j if x == j + 1 else x for x in v)
        maximum, mass, tight, _ = _candidate_stats(u, b, w, bounds)
        key = (maximum, mass, -tight, j)
        ranked.append((key, j, b))
    ranked.sort()
    n = len(u)
    for key, j, b in ranked:
        if key[0] > 0:
            continue
        pp = (u, b, tuple(n + 1 - x for x in w))
        if _bounded_zero(pp):
            continue
        passes = True
        for qq in sorted(set(_transfers(pp))):
            if _common_ascent(qq):
                passes = False
                break
            transformed_w = tuple(n + 1 - x for x in qq[2])
            if _candidate_stats(qq[0], qq[1], transformed_w, bounds)[0] > 0:
                passes = False
                break
        if passes:
            return j
    # This fallback keeps the interface total if inputs fall outside the promise.
    return ranked[0][1]
