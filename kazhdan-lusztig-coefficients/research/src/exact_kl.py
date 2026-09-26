"""Interval-local exact KL recurrence extracted unchanged."""
from functools import lru_cache
from kl_type_a import inv, leq, padd, qshift, rs

@lru_cache(None)
def down_covers(w):
    ans = []
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if w[i] <= w[j]:
                continue
            if any(w[j] < w[k] < w[i] for k in range(i + 1, j)):
                continue
            z = list(w)
            z[i], z[j] = z[j], z[i]
            ans.append(tuple(z))
    return tuple(ans)


@lru_cache(None)
def interval(u, v):
    if not leq(u, v):
        return ()
    floor = inv(u)
    seen, stack = {v}, [v]
    while stack:
        y = stack.pop()
        if inv(y) == floor:
            continue
        for z in down_covers(y):
            if z not in seen and leq(u, z):
                seen.add(z)
                stack.append(z)
    return tuple(sorted(seen, key=lambda z: (inv(z), z)))


@lru_cache(None)
def P(u, v):
    if not leq(u, v):
        return (0,)
    if u == v:
        return (1,)
    i = next(i for i in range(len(v) - 1) if v[i] > v[i + 1])
    us, vs = rs(u, i), rs(v, i)
    c = int(inv(us) < inv(u))
    ans = padd(qshift(P(us, vs), 1 - c), qshift(P(u, vs), c))
    for z in interval(u, vs):
        if z == vs or inv(rs(z, i)) >= inv(z):
            continue
        gap = inv(vs) - inv(z)
        if gap <= 0 or gap % 2 == 0:
            continue
        pz = P(z, vs)
        degree = (gap - 1) // 2
        mu = pz[degree] if degree < len(pz) else 0
        if mu:
            ans = padd(ans, qshift(P(u, z),
                                    (inv(v) - inv(z)) // 2), sign=-mu)
    assert all(a >= 0 for a in ans), (u, v, ans)
    return ans
