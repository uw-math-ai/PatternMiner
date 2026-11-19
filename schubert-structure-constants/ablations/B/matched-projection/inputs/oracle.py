"""Finite-flag research oracle. Use coefficient, empty, depth, usage."""
import sqlite3
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import lrcalc

MAX_RANK = 9
LIMIT = 50_000
DATABASE = Path(__file__).with_name('query-ledger.sqlite')
def _checked(u, v, w):
    values = tuple(tuple(p) for p in (u,v,w))
    n = len(values[0])
    if n < 2 or n > MAX_RANK or any(sorted(p) != list(range(1,n+1)) for p in values):
        raise ValueError('Supply permutations of a common permitted rank')
    return values

def _ledger():
    conn = sqlite3.connect(DATABASE)
    conn.execute('CREATE TABLE IF NOT EXISTS answers (question TEXT PRIMARY KEY, answer INTEGER NOT NULL)')
    return conn

def usage():
    conn = _ledger()
    try:return conn.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    finally:conn.close()

def _query(question,calculate):
    conn=_ledger()
    try:
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            key=repr(question)
            old=conn.execute('SELECT answer FROM answers WHERE question=?',(key,)).fetchone()
            if old is not None:return old[0]
            if conn.execute('SELECT COUNT(*) FROM answers').fetchone()[0]>=LIMIT:raise RuntimeError('Exact-question allowance exhausted')
            result=int(calculate())
            conn.execute('INSERT INTO answers VALUES (?,?)',(key,result))
            return result
    finally:conn.close()

def desc(p):
    return {i for i in range(1,len(p)) if p[i-1]>p[i]}

def complement(p):
    return tuple(len(p)+1-x for x in p)

def _minimal(p,e):
    cuts=(0,*e,len(p))
    return tuple(x for a,b in zip(cuts,cuts[1:]) for x in sorted(p[a:b]))

@lru_cache(maxsize=1024)
def _product(a,b):
    return {tuple(z)+tuple(range(len(z)+1,len(a)+1)):v for z,v in lrcalc.schubmult(a,b,len(a)).items() if v}

def _below(a,b):
    return all(x<=y for k in range(1,len(a)) for x,y in zip(sorted(a[:k]),sorted(b[:k])))

def _projected(u,v,w,e):
    a,b,t=(_minimal(p,e) for p in (u,v,complement(w)))
    bound=complement(t)
    return not any(_below(z,bound) for z in _product(a,b))

def coefficient(u,v,w):
    u,v,w=_checked(u,v,w)
    return _query(('coefficient',u,v,w),lambda:_product(u,v).get(w,0))

def empty(u,v,w,retained):
    u,v,w=_checked(u,v,w)
    e=tuple(sorted(set(retained)))
    if not set(e)<=set(range(1,len(w))):raise ValueError('Invalid flag dimension')
    return bool(_query(('empty',u,v,w,e),lambda:_projected(u,v,w,e)))

def depth(u,v,w):
    u,v,w=_checked(u,v,w)
    if coefficient(u,v,w):return 0
    dims=sorted(desc(u)|desc(v))
    for k in range(len(dims)+1):
        if any(empty(u,v,w,e) for e in combinations(dims,k)):return k
    raise ValueError('The union of descents did not certify zero; check homogeneity')
