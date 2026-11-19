"""A uniform family of homogeneous vanishing Schubert triples.

For n >= 6 the coefficient vanishes and the depth is n-3, as proved
in REPORT.md.  The supplied oracle independently checked n=6,...,9.
"""

def _append(p, x):
    return tuple(y + (y >= x) for y in p) + (x,)

def _top(n):
    if n < 6:
        return None
    a = (1, 3, 5, 2, 6, 4)
    b = (1, 5, 4, 2, 3, 6)
    c = (3, 2, 1, 6, 5, 4)
    for m in range(6, n):
        a, b, c = _append(b, m), _append(c, m+1), _append(a, 2)
    return a, b, c

def example(n):
    """Return (u,v,w) in S_n, or None when n < 6."""
    if not isinstance(n, int) or n < 6:
        return None
    top = _top(n)
    # Both dimensions with a single top-factor descent belong to this factor.
    holder = (2 - (n-6)) % 3
    u = top[(holder+1) % 3]
    v = top[(holder+2) % 3]
    third = top[holder]
    w = tuple(n+1-x for x in third)
    return list(u), list(v), list(w)
