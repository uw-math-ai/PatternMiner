"""Replay the matched counterexample and deduce its second root polynomial."""
from itertools import product
import json
from pathlib import Path
import numpy as np
import session as s
from demazure_projection import all_levi_restrictions,block_projection,inversion_length
from full_resources import Budget


def left(p,i):return tuple(i+1 if v==i else i if v==i+1 else v for v in p)


def main(budget):
    first=json.loads((s.HERE/'ALL_LEVI_COUNTEREXAMPLE.json').read_text())
    second=json.loads((s.HERE/'LEVI_COUNTEREXAMPLE_COMPANION.json').read_text())
    u,v,w,z=map(tuple,[first['u'],first['v'],first['w'],second['w']]);n=len(w)
    assert u==tuple(second['u']) and v==tuple(second['v'])
    counts=np.array(second['retained_coefficients'],dtype=np.int64)
    assert counts[0,0,5]==counts[1,0,5]==1
    assert np.flatnonzero(counts[0,1]).tolist()==[4] and np.flatnonzero(counts[1,1]).tolist()==[3]
    assert inversion_length(w)==inversion_length(z)==inversion_length(u)+inversion_length(v)-1
    assert all(w.index(i)==z.index(i) for i in [4,5,6,7])
    # The common cover identifies D_2(z)=D_9(w), using the retained first root polynomial.
    assert left(w,9)==left(z,2)
    q=np.array(first['Q'],dtype=np.int64);padded=np.r_[0,q,0]
    cover=np.r_[counts[0].sum(0)-(2*q-padded[:-2]-padded[2:]),q[-1]]
    assert cover[8]==1
    inverse=np.argsort(z);ascents=[i+1 for i in range(n-1) if inverse[i]<inverse[i+1]]+[n]
    ab=np.r_[counts[1].sum(0),0];mass=int(np.dot(np.arange(1,n+1),ab));assert mass==10
    free=[i for i in ascents if i!=2];solutions=[]
    for values in product(*(range(mass//i+1) for i in free)):
        budget.check();D=np.zeros(n,dtype=np.int64);D[1]=cover[8]
        for i,value in zip(free,values):D[i-1]=value
        if np.dot(np.arange(1,n+1),D)!=mass:continue
        f=D-ab;Q=np.array([np.dot(np.arange(1,n-j),f[j+1:]) for j in range(n-1)])
        if (Q>=0).all():solutions.append((D,Q))
    assert len(solutions)==1;D,Q=solutions[0]
    assert (Q[None,:]>=counts[1]).all()
    extensions=[]
    for N in range(10,18):
        extended=[p+tuple(range(n+1,N+1)) for p in [u,v,w,z]]
        g1,_=all_levi_restrictions(*extended[:3]);g2,_=all_levi_restrictions(extended[0],extended[1],extended[3])
        assert g1[3:5]==g2[3:5]==[0,0]
        extensions.append(dict(n=N,candidate_maxima_first=g1[3:5],candidate_maxima_second=g2[3:5]))
    support=np.flatnonzero(Q)+1;left_value=int(support.min());right_value=int(support.max()+1)
    a,ell_a,_,ell_x=block_projection(u,z,left_value,right_value)
    b,ell_b,_,_=block_projection(v,z,left_value,right_value)
    report=dict(status='verified',source_sha256=s.sha(Path(__file__)),first_counterexample_sha256=s.sha(s.HERE/'ALL_LEVI_COUNTEREXAMPLE.json'),
        companion_sha256=s.sha(s.HERE/'LEVI_COUNTEREXAMPLE_COMPANION.json'),lengths=dict(u=inversion_length(u),v=inversion_length(v),w=inversion_length(w)),
        supplied_index=6,candidate_indices=[4,5],coefficient_vectors=counts.tolist(),
        second_root_coefficients=Q.tolist(),second_cover_coefficients=D.tolist(),root_derivation='Unique nonnegative solution of the first-order Cauchy identity, the boundary identity, the left-ascent support condition, and the shared-cover coefficient D_2(z)=D_9(w)=1.',
        support_levi=dict(value_interval=[left_value,right_value],projected_u=a,projected_v=b,degree_at_x=ell_a+ell_b-ell_x),
        stability_checks=extensions,exact_product_queries=0,
        limitation='The second root polynomial is deduced from retained exact coefficients and the recorded Cauchy identity, rather than a new product query. Stability beyond the displayed finite checks follows from the stable Schubert conventions and fixed-point extensions of the projections.')
    s.write('LEVI_COMPANION_VERIFICATION.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    budget=Budget('levi_companion_verification',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
