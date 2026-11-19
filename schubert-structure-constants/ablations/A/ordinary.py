"""Proved, bounded descent-cycling certificates for full-flag vanishing.

This module uses no labels, precomputed triples, or coefficient computations.
"""

_MAX_STATES = 8192


def _length(p):
    return sum(x > y for i, x in enumerate(p) for y in p[i + 1:])


def _bruhat_leq(p, q):
    """Rank criterion for the ordinary strong Bruhat order."""
    for k in range(1, len(p)):
        if any(x > y for x, y in zip(sorted(p[:k]), sorted(q[:k]))):
            return False
    return True


def _screen_fails(triple):
    n = len(triple[0])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if not _bruhat_leq(triple[i], tuple(n + 1 - x for x in triple[j])):
            return True
    for lo in range(n - 1):
        count = 0
        for hi in range(lo + 1, n):
            count += sum(p[k] > p[hi] for p in triple for k in range(lo, hi))
            if count < (hi - lo + 1) * (hi - lo) // 2:
                return True
    return False


def _canonical(triple):
    """Factor symmetry and the Dynkin diagram involution."""
    ordered = tuple(sorted(triple))
    n = len(ordered[0])
    reflected = tuple(sorted(tuple(n + 1 - x for x in reversed(p)) for p in ordered))
    return min(ordered, reflected)


def _cycle_neighbors(triple):
    n = len(triple[0])
    for i in range(n - 1):
        descents = [k for k, p in enumerate(triple) if p[i] > p[i + 1]]
        if len(descents) != 1:
            continue
        source = descents[0]
        p = list(triple[source])
        p[i], p[i + 1] = p[i + 1], p[i]
        p = tuple(p)
        for target in range(3):
            if target == source:
                continue
            q = list(triple[target])
            q[i], q[i + 1] = q[i + 1], q[i]
            neighbor = list(triple)
            neighbor[source], neighbor[target] = p, tuple(q)
            yield _canonical(neighbor)


def certifies_zero(u, v, w):
    """Certify c_{u,v}^w = 0, or abstain with False.

    Inputs are permutations in one-line notation. The search stops after
    examining at most _MAX_STATES symmetry classes. Exhausting this budget
    gives an abstention, never a vanishing assertion.
    """
    u, v, w = tuple(u), tuple(v), tuple(w)
    n = len(u)
    expected = set(range(1, n + 1))
    if len(v) != n or len(w) != n or any(set(p) != expected for p in (u, v, w)):
        return False
    if _length(u) + _length(v) != _length(w):
        return True
    initial = (u, v, tuple(n + 1 - x for x in w))
    if _screen_fails(initial):
        return True
    initial = _canonical(initial)
    queue = [initial]
    seen = {initial}
    for triple in queue:
        for neighbor in _cycle_neighbors(triple):
            if neighbor in seen:
                continue
            if len(seen) >= _MAX_STATES:
                return False
            seen.add(neighbor)
            if _screen_fails(neighbor):
                return True
            queue.append(neighbor)
    return False
