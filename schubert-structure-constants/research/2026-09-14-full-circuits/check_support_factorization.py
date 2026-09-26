"""Check the support-component corollary on retained higher-degree polynomials."""
from collections import defaultdict
from functools import reduce
import json
import math
from pathlib import Path
import session as s
from full_resources import Budget


def check(q):
    width=len(next(iter(q)));degree=sum(next(iter(q)))
    assert all(c>0 and sum(e)==degree for e,c in q.items())
    support=[i for i in range(width) if any(e[i] for e in q)];components=[]
    for i in support:
        if not components or i>components[-1][-1]+1:components.append([])
        components[-1].append(i)
    assert len(components)<=degree
    factors=[]
    for component in components:
        assert len({sum(e[i] for i in component) for e in q})==1
        marginal=defaultdict(int)
        for e,c in q.items():marginal[tuple(e[i] for i in component)]+=c
        content=math.gcd(*marginal.values());factors.append({e:c//content for e,c in marginal.items()})
    product={(0,)*width:1}
    for component,factor in zip(components,factors):
        out={}
        for e,c in product.items():
            for a,b in factor.items():
                exponent=list(e)
                for i,power in zip(component,a):exponent[i]=power
                out[tuple(exponent)]=c*b
        product=out
    content=math.gcd(*q.values());assert q=={e:content*c for e,c in product.items()}
    return components,factors,content


def main(budget):
    source=s.HERE.parent/'2026-09-12-leading-form';records={};hashes={}
    for name in ['FACTORIZATION_S5.json','FACTORIZATION_SAMPLE_S6.json']:
        path=source/name;hashes[name]=s.sha(path)
        for row in json.loads(path.read_text())['examples']:
            key=tuple(tuple(row[k]) for k in ['u','v','w']);q={tuple(term[:-1]):term[-1] for term in row['q']}
            if key in records:assert q==records[key]
            records[key]=q
    disconnected=[];nonconstant=0
    for (u,v,w),q in records.items():
        budget.check();components,factors,content=check(q);nonconstant+=sum(next(iter(q)))>0
        if len(components)>1:disconnected.append(dict(u=u,v=v,w=w,q=[list(e)+[c] for e,c in q.items()],components=[[i+1 for i in C] for C in components],primitive_factors=[[list(e)+[c] for e,c in factor.items()] for factor in factors],content=content))
    report=dict(status='passed',source_sha256=s.sha(Path(__file__)),retained_sources=hashes,
        unique_coefficients=len(records),nonconstant_coefficients=nonconstant,disconnected_support_coefficients=len(disconnected),examples=disconnected[:20],
        exact_queries=0,checks=['Homogeneity in each connected support component','Number of support components at most total degree','Original polynomial equals its content times the primitive component marginals'],
        scope='Only the retained higher-degree examples from the earlier factorization studies, deduplicated by (u,v,w). This is not a new exhaustive census or independent recomputation.')
    s.write('SUPPORT_FACTORIZATION_CHECK.json',report);print(json.dumps({k:v for k,v in report.items() if k!='examples'}),flush=True)


if __name__=='__main__':
    budget=Budget('support_factorization_retained_check',60);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
