"""Check the elementary steps of the derived S6 obstruction, without product queries."""
import json
from pathlib import Path
import numpy as np
import session as s
from demazure_projection import block_projection,all_levi_restrictions,inversion_length
from full_resources import Budget


def rank(p):return (np.asarray(p)[None,:]<=np.arange(1,len(p)+1)[:,None]).cumsum(1)


def desc(p):
    inverse=np.argsort(p)
    return (np.flatnonzero(inverse[:-1]>inverse[1:])+1).tolist()


def main(budget):
    companion=json.loads((s.HERE/'LEVI_COUNTEREXAMPLE_COMPANION.json').read_text())
    derived=json.loads((s.HERE/'LEVI_COMPANION_VERIFICATION.json').read_text())
    u,v,w=map(tuple,[companion['u'],companion['v'],companion['w']]);L,R=3,8
    a,ella,_,ellx=block_projection(u,w,L,R);b,ellb,_,_=block_projection(v,w,L,R)
    x=[value-L+1 for value in w if L<=value<=R]
    Q=derived['second_root_coefficients'][L-1:R-1]
    assert a==[1,5,2,3,6,4] and b==[4,3,1,2,5,6] and x==[4,6,1,2,5,3]
    assert Q==[1,2,2,2,1] and ella==4 and ellb==5 and ellx==8
    assert (rank(a)>=rank(x)).all() and (rank(b)>=rank(x)).all()
    assert 3 not in desc(a) and 3 in desc(x) and 3 in desc(b)
    padded=[0,*Q,0];cartan=[2*Q[i]-padded[i]-padded[i+2] for i in range(5)]
    assert cartan[2]==0 and cartan[3]==1 and 4 not in desc(b)
    checks=[]
    for n in range(6,18):
        budget.check();aa,bb,xx=[tuple(p)+tuple(range(7,n+1)) for p in [a,b,x]]
        maximum,_=all_levi_restrictions(aa,bb,xx);assert maximum[2]==0
        checks.append(dict(n=n,candidate_3_maximum_violation=maximum[2]))
    report=dict(status='verified_algebraic_deduction',source_sha256=s.sha(Path(__file__)),
        u=a,v=b,w=x,Q=Q,lengths=[ella,ellb,ellx],left_descents=dict(u=desc(a),v=desc(b),w=desc(x)),
        cartan_coefficients=cartan,supplied_index=4,zero_candidate=3,
        positive_supplied_derivation='The Cauchy identity at index 4 gives A_4=1+D_4>=1; full Levi divisibility applied to the retained global ordinary coefficient one sharpens this to A_4=1.',
        zero_derivation='At index 3, u is a left ascent and w is a left descent, so A_3=D_3=0 and B_3=2Q_3-Q_2-Q_4=0.',
        polynomial_derivation='Full Levi divisibility and primitive global content identify this local polynomial with the shifted retained global polynomial. See LEVI_LENGTH_OBSTRUCTION.md.',
        exact_product_queries=0,stability_checks=checks,
        mathematical_dependency='Frozen parabolic order and full Levi divisibility theorems; the recorded first-order Cauchy identity. This is not a new independent product evaluation.')
    s.write('SMALL_LEVI_OBSTRUCTION.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    budget=Budget('small_levi_obstruction_checks',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
