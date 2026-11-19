"""Proved structural zero certificates for finite full flag Schubert constants.

No data files, coefficient evaluations, or persistent triple cache are used.
See REPORT.md for the geometric argument and the bounded search convention.
"""

_STATE_LIMIT = 8192


def _length(p):
    return sum(x > y for i, x in enumerate(p) for y in p[i + 1:])


def certifies_zero(u, v, w):
    """Certify vanishing, or abstain by returning False.

    Inputs are permutations in one-line notation. Invalid inputs give False.
    The local rewriting rule transfers a sole right descent between factors.
    All inequalities checked at every visited triple are necessary conditions.
    """
    n = len(u)
    if n < 1 or len(v) != n or len(w) != n:
        return False
    target_set = set(range(1, n + 1))
    if any(set(p) != target_set for p in (u, v, w)):
        return False
    if _length(u) + _length(v) != _length(w):
        return True
    if n == 1:
        return False

    # Packed integer fields evaluate many elementary inequalities at once.
    # Every field has a guard bit strictly larger than any fieldwise sum.
    total = n * (n - 1) // 2
    width = (3 * total).bit_length() + 1
    guard_bit = 1 << (width - 1)
    roots = [(i, j) for i in range(n) for j in range(i + 1, n)]
    intervals = []
    interval_target = interval_guard = 0
    for size in range(2, n):
        for lo in range(n - size + 1):
            mask = sum(1 << r for r, (i, j) in enumerate(roots)
                       if lo <= i < j < lo + size)
            shift = width * len(intervals)
            intervals.append((mask, shift))
            interval_target += (size * (size - 1) // 2) << shift
            interval_guard += guard_bit << shift

    # A value at position i contributes to all prefix ranks of size > i.
    tails = [sum(1 << (width * k) for k in range(i, n - 1))
             for i in range(n)]
    rank_target = sum((k + 1) << (width * k) for k in range(n - 1))
    rank_guard = sum(guard_bit << (width * k) for k in range(n - 1))
    compositions = [(a, b, n - a - b)
                    for a in range(n + 1) for b in range(n - a + 1)]
    features = {}

    def get_features(p):
        found = features.get(p)
        if found is not None:
            return found
        inversions = sum(1 << r for r, (i, j) in enumerate(roots)
                         if p[i] > p[j])
        interval_counts = sum((inversions & mask).bit_count() << shift
                              for mask, shift in intervals)
        positions = [0] * (n + 1)
        descents = 0
        for i, value in enumerate(p):
            positions[value] = i
            if i + 1 < n and value > p[i + 1]:
                descents |= 1 << i
        low = [0]
        for value in range(1, n + 1):
            low.append(low[-1] + tails[positions[value]])
        high = tuple(rank_target - low[n - a] for a in range(n + 1))
        found = (interval_counts, tuple(low), high, descents)
        features[p] = found
        return found

    first = tuple(sorted((u, v, tuple(n + 1 - x for x in w))))
    queue = [first]
    seen = {first}
    for triple in queue:
        f, g, h = (get_features(p) for p in triple)
        interval_sum = f[0] + g[0] + h[0]
        if ((interval_sum + interval_guard - interval_target)
                & interval_guard) != interval_guard:
            return True

        # For every k and A+B+C=n, sum R_p(k,A) >= k.
        # The dual direct-sum inequality uses upper tails and has sum <= k.
        fl, gl, hl = f[1], g[1], h[1]
        fu, gu, hu = f[2], g[2], h[2]
        for a, b, c in compositions:
            low_sum = fl[a] + gl[b] + hl[c]
            if ((low_sum + rank_guard - rank_target)
                    & rank_guard) != rank_guard:
                return True
            high_sum = fu[a] + gu[b] + hu[c]
            if ((rank_target + rank_guard - high_sum)
                    & rank_guard) != rank_guard:
                return True

        if len(seen) >= _STATE_LIMIT:
            continue
        d0, d1, d2 = f[3], g[3], h[3]
        sole = ((d0 & ~d1 & ~d2) | (d1 & ~d0 & ~d2)
                | (d2 & ~d0 & ~d1))
        while sole and len(seen) < _STATE_LIMIT:
            bit = sole & -sole
            sole -= bit
            i = bit.bit_length() - 1
            k = 0 if d0 & bit else (1 if d1 & bit else 2)
            p = triple[k]
            shortened = p[:i] + (p[i + 1], p[i]) + p[i + 2:]
            for j in range(3):
                if j == k:
                    continue
                q = triple[j]
                lengthened = q[:i] + (q[i + 1], q[i]) + q[i + 2:]
                nxt = list(triple)
                nxt[k], nxt[j] = shortened, lengthened
                nxt = tuple(sorted(nxt))
                if nxt not in seen:
                    if len(seen) >= _STATE_LIMIT:
                        break
                    seen.add(nxt)
                    queue.append(nxt)
    return False
