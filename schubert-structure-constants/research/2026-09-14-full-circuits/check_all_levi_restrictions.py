"""Verify Demazure projections and test every Levi subset on exposed examples."""
from functools import lru_cache
from itertools import permutations
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_features as rf
from demazure_projection import block_projection,all_levi_restrictions,inversion_length
from full_resources import Budget


def rank(p):
    return (np.asarray(p)[None,:]<=np.arange(1,len(p)+1)[:,None]).cumsum(1)


def components(mask,n):
    blocks=[];i=1
    while i<n:
        if not mask&(1<<(i-1)):i+=1;continue
        left=i
        while i<n and mask&(1<<(i-1)):i+=1
        blocks.append((left,i))
    return blocks


def verify(budget):
    checked=0;cases=0;rng=np.random.default_rng(s.SEED+90000)
    for n in [4,5]:
        pp=sorted(permutations(range(1,n+1)),key=lambda p:(inversion_length(p),p));ranks=np.stack([rank(p) for p in pp])
        eligible=[(h,w) for w in range(len(pp)) for h in range(len(pp)) if (ranks[h]>=ranks[w]).all()]
        if n==5:eligible=[eligible[i] for i in rng.choice(len(eligible),100,replace=False)]
        for hi,wi in eligible:
            budget.check();h,w=pp[hi],pp[wi];cases+=1
            for mask in range(1<<(n-1)):
                blocks=components(mask,n);z=list(w);fast=list(range(1,n+1))
                for left,right in blocks:
                    positions=[i for i,v in enumerate(w) if left<=v<=right]
                    for i,value in zip(positions,range(left,right+1)):z[i]=value
                    local,*_=block_projection(h,w,left,right)
                    fast[left-1:right]=[v+left-1 for v in local]
                candidates=[]
                for p in pp:
                    allowed=set()
                    for left,right in blocks:
                        if set(p[left-1:right])!=set(range(left,right+1)):break
                        allowed.update(range(left-1,right))
                    else:
                        if all(p[i]==i+1 for i in range(n) if i not in allowed):candidates.append(p)
                answer=next(p for p in candidates if (rank(h)>=rank([p[v-1] for v in z])).all())
                assert tuple(fast)==answer,(h,w,mask,fast,answer);checked+=1
    return dict(permutation_pairs=cases,all_subset_projection_checks=checked,scope='All comparable pairs in S4 and one hundred fixed comparable pairs in S5, with every Levi subset; compare with direct Bruhat enumeration.')


def main(budget):
    checks=verify(budget)
    data=dict(np.load(s.ART/'first_exposed_S10_pairs.npz'))
    with np.load(s.ART/'SECOND_ERROR_COMPARISON.npz') as archive:
        scores=archive['FINAL_REFINEMENT_READOUTS_scores'].copy();good=archive['FINAL_REFINEMENT_READOUTS_good'].copy()
    with np.load(s.ART/'LARGER_LEVI_RESTRICTIONS.npz') as archive:
        take=archive['selected_rows'].copy();earlier=archive['excluded'].copy()
    pp,nn=rf.permutations(torch.as_tensor(data['x'][take]));pp=pp.numpy();nn=nn.numpy()
    excluded=np.zeros_like(data['mask'][take]);maxima=np.zeros_like(excluded,dtype=np.int32)
    for row,(p,n) in enumerate(zip(pp,nn)):
        budget.check();u,v,w=[tuple(map(int,q[:n])) for q in p]
        values,blocks=all_levi_restrictions(u,v,w);maxima[row,:n-1]=values;excluded[row,:n-1]=np.asarray(values)>0
        # Check the dynamic program independently by enumerating every subset.
        exact=[-999]*(n-1)
        for mask in range(1<<(n-1)):
            cs=components(mask,n);delta=sum(blocks[c][0] for c in cs)
            for left,right in cs:
                for i in range(left,right):exact[i-1]=max(exact[i-1],delta-int(blocks[left,right][1][i-left]))
        assert exact==values,(row,exact,values)
        short=np.zeros(16,bool)
        for (left,right),(delta,desc) in blocks.items():
            if 5<=right-left+1<=8:short[left-1:right-1]|=delta-np.asarray(desc,dtype=int)>0
        assert np.array_equal(short,earlier[row]),row
    y=data['y'][take];mask=data['mask'][take]
    assert not (excluded&y).any()
    pred=np.where(mask&~excluded,scores[take],-1e9).argmax(1);after=y[np.arange(len(take)),pred]
    output=s.ART/'ALL_LEVI_RESTRICTIONS.npz'
    np.savez_compressed(output,selected_rows=take,maximum_violation=maxima,excluded=excluded,before=good[take],after=after,predictions=pred)
    report=dict(status='completed',source_sha256=s.sha(Path(__file__)),algorithm_sha256=s.sha(s.HERE/'demazure_projection.py'),
        projection_checks=checks,dynamic_program_checks=len(take),larger_block_replay_exact=True,
        inputs=len(take),initially_incorrect_inputs=int((~good[take]).sum()),remaining_incorrect_inputs=int((~after).sum()),
        fixed=int((~good[take]&after).sum()),lost=int((good[take]&~after).sum()),excluded_valid_targets=0,
        excluded_candidates=int((excluded&mask).sum()),exact_product_queries=0,artifact_sha256=s.sha(output),
        scope='The same exposed S10 errors and controls as LARGER_LEVI_RESTRICTIONS, now including every Levi subset, connected or disconnected. No model parameters change and no replacement-assessment predictions are evaluated.')
    s.write('ALL_LEVI_RESTRICTIONS.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('all_levi_restriction_check',300);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
