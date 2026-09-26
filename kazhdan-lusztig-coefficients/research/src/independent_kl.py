"""Independent audit engine: extracted definitions, no originating imports."""
from functools import lru_cache
from itertools import combinations

@lru_cache(None)
def length(p): return sum(a>b for i,a in enumerate(p) for b in p[i+1:])


def swap(p,i,j):
    q=list(p);q[i],q[j]=q[j],q[i];return tuple(q)


def desc(p): return {i for i in range(len(p)-1) if p[i]>p[i+1]}


@lru_cache(None)
def below(a,b):
    return all(all(x<=y for x,y in zip(sorted(a[:k]),sorted(b[:k]))) for k in range(1,len(a)+1))


def covers(p,direction):
    return [q for i,j in combinations(range(len(p)),2) if length(q:=swap(p,i,j))==length(p)+direction]


@lru_cache(None)
def interval(a,b):
    if not below(a,b): return ()
    seen={a};todo=[a]
    while todo:
        p=todo.pop()
        for q in covers(p,1):
            if q not in seen and below(q,b):seen.add(q);todo.append(q)
    return tuple(seen)


def addto(out,p,shift=0,mul=1):
    for k,c in enumerate(p):
        while len(out)<=k+shift:out.append(0)
        out[k+shift]+=mul*c


@lru_cache(None)
def polynomial(a,b):
    if not below(a,b):return (0,)
    if a==b:return (1,)
    k=max(desc(b));bs=swap(b,k,k+1);as_=swap(a,k,k+1);c=int(k in desc(a));out=[]
    addto(out,polynomial(a,bs),c);addto(out,polynomial(as_,bs),1-c)
    for z in interval(a,bs):
        gap=length(bs)-length(z)
        if k not in desc(z) or gap<=0 or gap%2==0:continue
        p=polynomial(z,bs);d=(gap-1)//2;mu=p[d] if d<len(p) else 0
        if mu:addto(out,polynomial(a,z),(length(b)-length(z))//2,-mu)
    while len(out)>1 and out[-1]==0:out.pop()
    assert out and all(v>=0 for v in out) and len(out)-1<=(length(b)-length(a)-1)//2
    return tuple(out)
