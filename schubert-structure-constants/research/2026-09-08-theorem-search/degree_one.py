"""Exact investigation of degree-one equivariant Schubert coefficients.

The Cauchy identity gives q(t)=sum_i (C_i-A_i-B_i)(t_1+...+t_i),
where A_i=c^w_{s_i u,v}, B_i=c^w_{u,s_i v}, C_i=c^{s_i w}_{u,v};
terms are retained only when the relevant left multiplication lowers u or v,
or raises w. This is an identity, not a claimed new theorem.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2026-09-08-followup'))
from support_conjecture import *
from collections import Counter
import argparse

HERE=Path(__file__).resolve().parent

def left_swap(w,i):
    return tuple(i+2 if a==i+1 else i+1 if a==i+2 else a for a in w)

def vectors(u,v,w):
    n=len(u);du=desc(inv(u));dv=desc(inv(v));dw=desc(inv(w))
    a=[ordinary(left_swap(u,i),v).get(w,0) if i in du else 0 for i in range(n-1)]
    b=[ordinary(u,left_swap(v,i)).get(w,0) if i in dv else 0 for i in range(n-1)]
    # The first-order Cauchy expansion is stable: s_n w may contribute even
    # for u,v,w in S_n (already s_1*s_1 illustrates this). Pad by one point.
    up=u+(n+1,);vp=v+(n+1,);wp=w+(n+1,)
    a.append(0);b.append(0)
    c=[ordinary(up,vp).get(left_swap(wp,i),0) if i not in dw else 0 for i in range(n)]
    f=[z-x-y for x,y,z in zip(a,b,c)]
    # t_j=sum_{k<j} alpha_k, using t_1=0. Coefficient of alpha_k.
    roots=[sum((i-k)*f[i] for i in range(k+1,n)) for k in range(n-1)]
    assert sum((i+1)*f[i] for i in range(n))==0,(u,v,w,a,b,c)
    assert min(roots)>=0,(u,v,w,roots)
    return a,b,c,roots

def investigate(n,sample=None):
    start=time.time()
    perms=sorted(itertools.permutations(range(1,n+1)),key=lambda w:(length(w),w))
    loc={w:localizations(w) for w in perms}; lens={w:length(w) for w in perms}
    pairs=list(itertools.combinations_with_replacement(perms,2))
    if sample is not None:pairs=random.Random(20260908+n).sample(pairs,min(sample,len(pairs)))
    stats=dict(n=n,pairs=len(pairs),exhaustive_pairs=sample is None,outputs_in_same_Sn=True,
               by_degree={},outside_stated_cases=0,counterexamples=[],linear_checks=0,
               stronger_claim_counterexamples={},examples=[])
    hist=Counter()
    def refute(name,u,v,w,**kw):
        if name not in stats['stronger_claim_counterexamples']:
            stats['stronger_claim_counterexamples'][name]=dict(u=u,v=v,w=w,**kw)
    for pair_index,(u,v) in enumerate(pairs):
        d=lens[u]+lens[v]; coeff={}
        for w in perms:
            if lens[w]>d:break
            if u not in loc[w] or v not in loc[w]:continue
            numerator=loc[w][u]*loc[w][v]-sum(c*loc[w].get(x,0) for x,c in coeff.items())
            q,rem=divmod(numerator,loc[w][w]);assert rem==0 and q>=0
            if q:coeff[w]=q
        left={};right={}
        for z in suffixes(u):
            for w,c in ordinary(z,v).items():left.setdefault(w,[]).append((z,c))
        for z in suffixes(v):
            for w,c in ordinary(u,z).items():right.setdefault(w,[]).append((z,c))
        candidate=left.keys()&right.keys()
        assert coeff.keys()<=candidate
        for w in candidate:
            degree=d-lens[w];hist[degree]+=1
            if w not in coeff:stats['counterexamples'].append(dict(u=u,v=v,w=w,degree=degree))
        if not known_case(u,v):stats['outside_stated_cases']+=1
        # Degree one: inspect even one-sided witnesses, including hard zeros.
        for w in left.keys()|right.keys():
            if lens[w]!=d-1:continue
            a,b,c,roots=vectors(u,v,w)
            assert sum(roots)==coeff.get(w,0),(u,v,w,a,b,c,roots,coeff.get(w,0))
            stats['linear_checks']+=1
            if any(a) and any(b) and not any(x*y for x,y in zip(a,b)):
                refute('common_simple_left_deletion_necessary',u,v,w,A=a,B=b,C=c,root_coefficients=roots)
            inds=[i for i,x in enumerate(roots) if x]
            if inds and inds!=list(range(min(inds),max(inds)+1)):
                refute('root_support_is_an_interval',u,v,w,A=a,B=b,C=c,root_coefficients=roots)
            if any(a) and any(b) and max([i for i,x in enumerate(a) if x])<min([i for i,x in enumerate(b) if x]):
                refute('deletion_supports_must_overlap_in_convex_hull',u,v,w,A=a,B=b,C=c,root_coefficients=roots)
            if any(a) and any(b) and not known_case(u,v) and len(stats['examples'])<12:
                stats['examples'].append(dict(u=u,v=v,w=w,A=a,B=b,C=c,root_coefficients=roots))
        if pair_index and pair_index%1000==0:
            print(json.dumps(dict(n=n,done=pair_index,seconds=round(time.time()-start,2),counterexamples=len(stats['counterexamples']))),flush=True)
    stats['by_degree']=dict(sorted(hist.items()));stats['seconds']=time.time()-start
    stats['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);p.add_argument('--sample',type=int);args=p.parse_args()
    r=investigate(args.n,args.sample)
    path=HERE/f'degree_one_S{args.n}_{args.sample or "all"}.json'
    path.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('examples','stronger_claim_counterexamples')}),flush=True)
