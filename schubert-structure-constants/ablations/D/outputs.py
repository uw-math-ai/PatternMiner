"""Permutation-only necessary-condition selector, using the Python standard library.

The six Newton-polytope containment checks are necessary for positivity.
Their converse is not asserted. The final ordering is an empirical heuristic.
No labelled configurations, coefficients, or external model files are used.
"""
from functools import lru_cache
from itertools import combinations


def _left_swap(p, j):
    return tuple(j + 1 if x == j else j if x == j + 1 else x for x in p)


def _complement(p):
    n = len(p)
    return tuple(n + 1 - x for x in p)


def _conjugate(p):
    return _complement(p[::-1])


@lru_cache(maxsize=4)
def _context(n):
    # Full subset inequalities on the supplied range n <= 14. For larger
    # permutations, retain a deterministic subfamily to bound inference time.
    if n <= 14:
        masks = tuple(range(1 << n))
    else:
        full = (1 << n) - 1
        keep = {0, full}
        for k in (1, 2):
            for inds in combinations(range(n), k):
                s = sum(1 << r for r in inds)
                keep.update((s, full ^ s))
        for a in range(n):
            for b in range(a + 1, n + 1):
                s = ((1 << b) - 1) ^ ((1 << a) - 1)
                keep.update((s, full ^ s))
        # The choice of masks depends only on n, never on development labels.
        state = 23063
        for _ in range(1024):
            state = (1664525 * state + 1013904223) & 0xffffffff
            keep.add(state & full)
        masks = tuple(sorted(keep))
    # Integer lanes evaluate many independent small-integer inequalities.
    bits = max(16, (n * n + 1).bit_length() + 2)
    bits = ((bits + 7) // 8) * 8
    width = bits // 8
    one = int.from_bytes(b''.join((1).to_bytes(width, 'little') for _ in masks), 'little')
    high = one << (bits - 1)
    prefix = []
    for t in range(n + 1):
        cut = (1 << t) - 1
        raw = b''.join((s & cut).bit_count().to_bytes(width, 'little') for s in masks)
        prefix.append(int.from_bytes(raw, 'little'))
    return bits, one, high, tuple(prefix)


def _lane_min(x, y, bits, high):
    # All lanes of x and y are below the lane's high bit. Adding high
    # prevents borrowing between lanes when their entries are compared.
    choose_y = (((x + high - y) & high) >> (bits - 1)) * ((1 << bits) - 1)
    return x ^ ((x ^ y) & choose_y)


@lru_cache(maxsize=64)
def _theta(p):
    """Pack Schubert-matroid rank sums for the selected row subsets."""
    n = len(p)
    bits, one, high, prefix = _context(n)
    position = [0] * (n + 1)
    for r, x in enumerate(p):
        position[x] = r
    total = 0
    columns = {}
    for j in range(1, n + 1):
        col = tuple(r + 1 for r in range(position[j]) if p[r] > j)
        if col:
            columns[col] = columns.get(col, 0) + 1
    for col, multiplicity in columns.items():
        k = len(col)
        rank = k * one
        for q, t in enumerate(col, 1):
            value = prefix[t] + (k - q) * one
            rank = _lane_min(rank, value, bits, high)
        total += multiplicity * rank
    return total


def _contained(x, y, z, high):
    """Check theta_z(S) <= theta_x(S) + theta_y(S) in every lane."""
    return ((_theta(x) + _theta(y) + high - _theta(z)) & high) == high


def _passes(u, b, w):
    # A common right ascent of the three classes makes their top pairing zero.
    c = _complement(w)
    if any(u[k] < u[k + 1] and b[k] < b[k + 1] and c[k] < c[k + 1]
           for k in range(len(w) - 1)):
        return False
    high = _context(len(w))[2]
    # Symmetry of the top pairing gives three presentations of the same
    # structure constant. Flag duality supplies their conjugate presentations.
    for x, y, z in ((u, b, w), (u, c, _complement(b)),
                    (b, c, _complement(u))):
        if not _contained(x, y, z, high):
            return False
        if not _contained(_conjugate(x), _conjugate(y), _conjugate(z), high):
            return False
    return True


def select_j(u, v, w, i, candidates):
    """Return one one-based candidate, using only u, v, w, and candidates.

    The given i certifies the task's premise. The necessary conditions concern
    the proposed product itself and do not otherwise use i.
    """
    u, v, w = tuple(u), tuple(v), tuple(w)
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError('candidates must be nonempty')
    # Prefer the descent whose larger value occurs latest in the second factor.
    ordered = sorted(candidates, key=lambda j: (-v.index(j + 1), j))
    for j in ordered:
        if _passes(u, _left_swap(v, j), w):
            return j
    # This branch also ensures a legal output on inputs outside the premise.
    return ordered[0]
