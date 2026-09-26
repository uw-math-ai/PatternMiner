"""Finite screening of Chen--Fan--Xiong--Yao (2026), Conjecture B.

Equivariant coefficients are evaluated by Billey's localization formula at a
strictly dominant integral weight. Graham positivity makes this exact for
nonvanishing. Triple support uses the proved Theorem 2.2 (left weak order).
We test outputs w in S_n only; stable outputs outside S_n are not enumerated.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import itertools
import json
import random
import time
import lrcalc

HERE=Path(__file__).resolve().parent
def length(w):return sum(a>b for i,a in enumerate(w) for b in w[i+1:])
def desc(w):return {i for i in range(len(w)-1) if w[i]>w[i+1]}
def inv(w):return tuple(w.index(i)+1 for i in range(1,len(w)+1))
def swap(w,i):
    p=list(w);p[i],p[i+1]=p[i+1],p[i];return tuple(p)
def trim(w):
    w=list(w)
    while w and w[-1]==len(w):w.pop()
    return tuple(w)
def reduced_word(w):
    steps=[]
    while desc(w):
        i=min(desc(w));steps.append(i);w=swap(w,i)
    return steps[::-1]


@lru_cache(None)
def ordinary(u,v):
    n=len(u)
    if length(u)+length(v)>n*(n-1)//2:return {}
    return {tuple(w)+tuple(range(len(w)+1,n+1)):int(c)
            for w,c in lrcalc.schubmult(list(trim(u)),list(trim(v))).items() if len(w)<=n}


@lru_cache(None)
def suffixes(u):
    out={u};stack=[u]
    while stack:
        w=stack.pop();wi=inv(w)
        for i in desc(wi):
            z=tuple(i+2 if a==i+1 else i+1 if a==i+2 else a for a in w)
            if z not in out:out.add(z);stack.append(z)
    return tuple(sorted(out))


def localizations(w):
    n=len(w);ident=tuple(range(1,n+1));prefix=ident;dp={ident:1}
    for i in reduced_word(w):
        beta=prefix[i+1]-prefix[i];assert beta>0
        nxt=dict(dp)
        for u,c in dp.items():
            if u[i]<u[i+1]:
                z=swap(u,i);nxt[z]=nxt.get(z,0)+c*beta
        dp=nxt;prefix=swap(prefix,i)
    assert prefix==w
    return dp


def known_case(u,v):
    du,dv=desc(u),desc(v)
    if not du or not dv:return True
    if max(du)<=min(dv) or max(dv)<=min(du):return True
    if len(desc(inv(u)))<=1 and len(desc(inv(v)))<=1:return True
    for z in (u,v):
        word=reduced_word(z)
        if word and (all(b-a==1 for a,b in zip(word,word[1:])) or
                     all(b-a==-1 for a,b in zip(word,word[1:]))):return True
    return False


def screen(n,sample=None):
    start=time.time();perms=sorted(itertools.permutations(range(1,n+1)),key=lambda w:(length(w),w))
    loc={w:localizations(w) for w in perms}
    pairs=list(itertools.combinations_with_replacement(perms,2))
    if sample is not None:pairs=random.Random(20260908+n).sample(pairs,min(sample,len(pairs)))
    stats=dict(n=n,unordered_pairs=len(pairs),exhaustive_pairs=sample is None,outputs_per_pair=len(perms),
               outside_three_proved_cases=0,candidate_memberships=0,equivariant_nonzero=0,
               degree_tight_cross_checks=0,counterexamples=[],example_outside_known=None)
    for u,v in pairs:
        coeff={}
        for w in perms:
            numerator=loc[w].get(u,0)*loc[w].get(v,0)-sum(c*loc[w].get(x,0) for x,c in coeff.items())
            c,rem=divmod(numerator,loc[w][w]);assert rem==0 and c>=0,(u,v,w,numerator)
            if c:coeff[w]=c
            if length(w)==length(u)+length(v):
                assert c==ordinary(u,v).get(w,0),(u,v,w,c)
                stats["degree_tight_cross_checks"]+=1
        left=set().union(*(set(ordinary(z,v)) for z in suffixes(u)))
        right=set().union(*(set(ordinary(z,u)) for z in suffixes(v)))
        candidate=left&right;actual=set(coeff)
        assert actual<=candidate,"Contradicts the proved direction: check conventions"
        residual=candidate-actual
        stats["candidate_memberships"]+=len(candidate);stats["equivariant_nonzero"]+=len(actual)
        if not known_case(u,v):
            stats["outside_three_proved_cases"]+=1
            if stats["example_outside_known"] is None and candidate:
                stats["example_outside_known"]=dict(u=u,v=v,candidate=sorted(candidate),coefficient_values=[coeff.get(w,0) for w in sorted(candidate)])
        if residual:stats["counterexamples"].append(dict(u=u,v=v,w=sorted(residual)))
    stats["seconds"]=time.time()-start
    return stats


if __name__=="__main__":
    out=dict(reference="https://arxiv.org/abs/2608.17378v1",conjecture="B",date="2026-09-08",
             limitations="Finite S_n output restriction; existing conjecture, not a new conjecture or proof.",results=[])
    for n,sample in ((3,None),(4,None),(5,600)):
        r=screen(n,sample);out["results"].append(r)
        print(json.dumps(r),flush=True)
    out["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/"support_conjecture_results.json").write_text(json.dumps(out,indent=2)+"\n")
