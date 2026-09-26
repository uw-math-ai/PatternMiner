#!/usr/bin/env python3
"""Independent bounded controls: no imports from originating research code."""
from functools import lru_cache
from itertools import permutations
from collections import Counter
import json
import time

@lru_cache(None)
def universe(n):
    return tuple(permutations(range(n)))

@lru_cache(None)
def length(p):
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i+1, len(p)))

@lru_cache(None)
def ranks(p):
    return tuple(sum(v >= j for v in p[:i]) for i in range(len(p)+1) for j in range(len(p)))

def le(a, b):
    return all(x <= y for x, y in zip(ranks(a), ranks(b)))

def swap(p, i, j=None):
    j = i+1 if j is None else j
    q = list(p)
    q[i], q[j] = q[j], q[i]
    return tuple(q)

def coefficient(p, j):
    return p[j] if 0 <= j < len(p) else 0

@lru_cache(None)
def kl(a, b):
    if not le(a, b):
        return ()
    if a == b:
        return (1,)
    s = max(i for i in range(len(b)-1) if b[i] > b[i+1])
    v, u = swap(b, s), swap(a, s)
    c = int(a[s] > a[s+1])
    out = [0]*(length(b)-length(a)+1)
    def add(p, shift, scale=1):
        for i, value in enumerate(p):
            out[i+shift] += scale*value
    add(kl(a, v), c)
    add(kl(u, v), 1-c)
    for z in universe(len(a)):
        dz = length(v)-length(z)
        if dz > 0 and dz % 2 and z[s] > z[s+1] and le(a,z) and le(z,v):
            mz = coefficient(kl(z,v), (dz-1)//2)
            if mz:
                add(kl(a,z), (length(b)-length(z))//2, -mz)
    while out and out[-1] == 0:
        out.pop()
    assert all(v >= 0 for v in out)
    assert len(out)-1 <= (length(b)-length(a)-1)//2
    return tuple(out)

def mu(a,b):
    d = length(b)-length(a)
    return coefficient(kl(a,b), (d-1)//2) if d > 0 and d % 2 else 0

def naked(a,b):
    return tuple(i for i,v in enumerate(a) if v == b[i] and sum(t>=v for t in a[:i]) == sum(t>=v for t in b[:i]))

def cycles(a,b):
    inv = {v:i for i,v in enumerate(a)}
    r = tuple(inv[v] for v in b)
    seen = set()
    c = 0
    for i in range(len(a)):
        if i not in seen:
            c += 1
            while i not in seen:
                seen.add(i)
                i = r[i]
    return c

@lru_cache(None)
def components(a,b):
    parent = list(range(len(a)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    p = b
    while p != a:
        candidates = ((i,j,swap(p,i,j)) for i in range(len(a)) for j in range(i+1,len(a)))
        i,j,q = next((i,j,q) for i,j,q in candidates if length(q) == length(p)-1 and le(a,q))
        parent[root(i)] = root(j)
        p = q
    return len({root(i) for i in range(len(a))})

def gain(a,b,s):
    v = swap(b,s)
    d = length(b)-length(a)
    A = coefficient(kl(a,v),(d-3)//2)
    C = sum(mu(a,z)*mu(z,v) for z in universe(len(a)) if a != z and le(a,z) and le(z,v) and z[s] > z[s+1] and (length(z)-length(a)) % 2)
    return A,C

def main():
    start = time.monotonic()
    counts = Counter()
    negative_gain = None
    unequal_polynomials = None
    for n in range(1,6):
        for a in universe(n):
            for b in universe(n):
                d = length(b)-length(a)
                if d < 3 or not d % 2 or not le(a,b):
                    continue
                counts['odd_noncover_intervals'] += 1
                M = mu(a,b)
                aa,bb = tuple(n-1-v for v in b),tuple(n-1-v for v in a)
                assert mu(aa,bb) == M
                assert cycles(aa,bb) == cycles(a,b)
                assert components(aa,bb) == components(a,b)
                assert naked(aa,bb) == naked(a,b)
                counts['long_element_checks'] += 1
                if unequal_polynomials is None and kl(a,b) != kl(aa,bb):
                    unequal_polynomials = dict(x=a,w=b,P=kl(a,b),dual_P=kl(aa,bb),mu=M)
                for s in range(n-1):
                    if not (a[s] > a[s+1] and b[s] > b[s+1]):
                        continue
                    A,C = gain(a,b,s)
                    assert M == mu(swap(a,s),swap(b,s))+A-C
                    counts['leading_recurrences'] += 1
                    if not le(a,swap(b,s)):
                        assert A == C == 0
                        counts['off_order_recurrences'] += 1
                    if A < C and negative_gain is None:
                        negative_gain = dict(x=a,w=b,s=s,A=A,C=C)
                if M <= 0 or naked(a,b):
                    continue
                assert components(a,b) == 1
                counts['canonical_positive_starts'] += 1
                if length(aa) < length(a):
                    a0,b0 = aa,bb
                else:
                    a0,b0 = a,b
                maximum = (n*(n-1)//2-d)//2
                assert 1 <= length(a0) <= maximum
                for policy in (min,max):
                    x,w = a0,b0
                    gains = []
                    while mu(x,w):
                        assert not naked(x,w)
                        assert components(x,w) == 1
                        assert cycles(x,w) == cycles(a,b)
                        descents = [s for s in range(n-1) if w[s] > w[s+1]]
                        assert descents and all(x[s] > x[s+1] for s in descents)
                        s = policy(descents)
                        A,C = gain(x,w,s)
                        gains.append(A-C)
                        x,w = swap(x,s),swap(w,s)
                        assert le(x,w)
                        assert length(w)-length(x) == d
                    assert 1 <= len(gains) <= length(a0) <= maximum
                    assert sum(gains) == M
                    assert max(gains) >= (M+maximum-1)//maximum
                    counts['telescoping_paths'] += 1
                    counts['path_steps'] += len(gains)
    assert unequal_polynomials is not None
    print(json.dumps(dict(status='PASS',counts=dict(counts),negative_gain_control=negative_gain,full_polynomial_symmetry_counterexample=unequal_polynomials,seconds=round(time.monotonic()-start,3)),indent=2))

if __name__ == '__main__':
    main()
