#!/usr/bin/env python3
"""Independent bounded controls for support-factor/rank/horizon candidate."""
import json,runpy
from pathlib import Path
from itertools import permutations,product
from functools import lru_cache
from independent_kl import length, below, covers, polynomial, swap, desc
def flat(p):
 s=sorted(p);return tuple(s.index(v) for v in p)
def naked(a,b):return [i for i in range(len(a)) if a[i]==b[i] and sum(v>=a[i] for v in a[:i])==sum(v>=a[i] for v in b[:i])]
def components(a,b):
 parent=list(range(len(a)))
 def root(i):
  while parent[i]!=i:i=parent[i]
  return i
 p=a
 while p!=b:
  q=min(q for q in covers(p,1) if below(q,b));i,j=[k for k in range(len(a)) if p[k]!=q[k]];parent[root(i)]=root(j);p=q
 return tuple(tuple(i for i in range(len(a)) if root(i)==r) for r in sorted({root(i) for i in range(len(a))}))
def mul(p,q):
 out=[0]*(len(p)+len(q)-1)
 for i,a in enumerate(p):
  for j,b in enumerate(q):out[i+j]+=a*b
 return tuple(out)
@lru_cache(None)
def R(a,b):
 if not below(a,b):return (0,)
 if a==b:return (1,)
 k=max(desc(b));bs=swap(b,k,k+1);as_=swap(a,k,k+1)
 if k in desc(a):return R(as_,bs)
 q=R(a,bs);p=R(as_,bs);out=[0]*(max(len(q),len(p))+1)
 for i,c in enumerate(q):out[i]-=c;out[i+1]+=c
 for i,c in enumerate(p):out[i+1]+=c
 while len(out)>1 and out[-1]==0:out.pop()
 return tuple(out)

def main():
 total=factored=positive=nonnaked=0
 for n in range(1,6):
  ws=list(permutations(range(n)));count=0
  for a in ws:
   for b in ws:
    if not below(a,b):continue
    count+=1;total+=1;d=length(b)-length(a);blocks=components(a,b)
    intervals=[];gaps=[];rp=(1,);pp=(1,)
    for block in blocks:
     aa=flat(tuple(a[i] for i in block));bb=flat(tuple(b[i] for i in block));assert sorted(a[i] for i in block)==sorted(b[i] for i in block)
     elems=[t for t in permutations(range(len(block))) if below(aa,t) and below(t,bb)]
     intervals.append(elems);gaps.append(length(bb)-length(aa));rp=mul(rp,R(aa,bb));pp=mul(pp,polynomial(aa,bb))
    assert sum(gaps)==d and R(a,b)==rp and polynomial(a,b)==pp
    expanded=set()
    for choice in product(*intervals):
     p=list(a)
     for block,t in zip(blocks,choice):
      vals=sorted(a[i] for i in block)
      for i,v in zip(block,t):p[i]=vals[v]
     expanded.add(tuple(p))
    actual={p for p in ws if below(a,p) and below(p,b)};assert expanded==actual
    for i in [block[0] for block in blocks if len(block)==1]:assert i in naked(a,b)
    if sum(g>0 for g in gaps)>1:factored+=1
    if not naked(a,b):
     nonnaked+=1;assert n<=2*d
     p=polynomial(a,b)
     if d%2 and (d-1)//2<len(p) and p[(d-1)//2]>0:
      positive+=1;assert n<=d+1 and sum(g>0 for g in gaps)==1
      assert length(a)<=d*(d-1)//2
    # Any pure downward descent transports full KL and R.
    for k in desc(a)&desc(b):
     bs=swap(b,k,k+1)
     if not below(a,bs):
      aa=swap(a,k,k+1);assert polynomial(a,b)==polynomial(aa,bs) and R(a,b)==R(aa,bs)
  print(json.dumps(dict(n=n,intervals=count)),flush=True)
 print(json.dumps(dict(status='PASS',intervals=total,multiple_nontrivial_components=factored,naked_free_intervals=nonnaked,naked_free_positive_mu=positive)),flush=True)
if __name__=='__main__':main()
