"""Exact permutation and polynomial primitives extracted unchanged."""
from functools import lru_cache

def inv(w):
    return sum(w[i] > w[j] for i in range(len(w)) for j in range(i + 1, len(w)))


def rs(w, i):
    w = list(w)
    w[i], w[i + 1] = w[i + 1], w[i]
    return tuple(w)


@lru_cache(None)
def leq(u, v):
    """Strong Bruhat comparison via southwest ranks."""
    n = len(u)
    if len(v) != n or inv(u) > inv(v):
        return False
    for p in range(1, n + 1):
        for q in range(n):
            if sum(x >= q for x in u[:p]) > sum(x >= q for x in v[:p]):
                return False
    return True


def padd(a, b, sign=1):
    c = [0] * max(len(a), len(b))
    for i, x in enumerate(a): c[i] += x
    for i, x in enumerate(b): c[i] += sign * x
    while len(c) > 1 and c[-1] == 0: c.pop()
    return tuple(c)


def qshift(a, k=1):
    return (0,) * k + a if any(a) else (0,)
