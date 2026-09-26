"""Recompute a zero that survives all Levi necessary length restrictions."""
import json
from pathlib import Path
import numpy as np
import torch
import lrcalc
import session as s
import compact
import richardson_features as rf
from demazure_projection import all_levi_restrictions,inversion_length
from exact_roots import root_targets
from full_resources import Budget


def trim(p):
    p=list(p)
    while len(p)>1 and p[-1]==len(p):p.pop()
    return tuple(p)


def left_simple(p,i):
    return tuple(i+1 if x==i else i if x==i+1 else x for x in p)


def coefficient(u,v,w,budget):
    budget.exact_query();product=lrcalc.schubmult(list(trim(u)),list(trim(v)))
    target=trim(w)
    return sum(int(c) for p,c in product.items() if trim(p)==target)


def main(budget):
    data=dict(np.load(s.ART/'first_exposed_S10_pairs.npz'))
    with np.load(s.ART/'ALL_LEVI_RESTRICTIONS.npz') as saved:
        bad=np.flatnonzero(~saved['after']);take=saved['selected_rows'].copy();pred=saved['predictions'].copy()
    pp,nn=rf.permutations(torch.as_tensor(data['x'][take[bad]]));pp=pp.numpy();nn=nn.numpy()
    choice=min(range(len(bad)),key=lambda k:(int(nn[k]),inversion_length(pp[k,0,:nn[k]])+inversion_length(pp[k,1,:nn[k]]),tuple(pp[k,0,:nn[k]]),tuple(pp[k,1,:nn[k]]),tuple(pp[k,2,:nn[k]])))
    selected=int(bad[choice]);source_row=int(take[selected]);n=int(nn[choice]);u,v,w=[tuple(map(int,p[:n])) for p in pp[choice]]
    i=int(np.flatnonzero(data['a'][source_row])[0])+1;j=int(pred[selected])+1
    with np.load(s.ART/'fresh_assessment_S10_assessment.npz') as archive:pair=archive['pairs'][source_row//2]
    record=np.array(compact.load(s.HERE/'cache/data/S10_assessment.bin')[[pair[source_row%2]]],copy=True)
    if pair[2]:
        record['permutations'][:,[0,1]]=record['permutations'][:,[1,0]]
        record['coefficients'][:,[0,1]]=record['coefficients'][:,[1,0]]
    assert np.array_equal(record['permutations'][0,:,:n],np.array([u,v,w]))
    A=coefficient(left_simple(u,i),v,w,budget);B=coefficient(u,left_simple(v,j),w,budget)
    assert A>0 and B==0
    assert A==int(record['coefficients'][0,0,i-1]) and B==int(record['coefficients'][0,1,j-1])
    violations,blocks=all_levi_restrictions(u,v,w);assert violations[j-1]<=0
    budget.exact_query();q,info=root_targets(record,dict(u=u,v=v),0)
    output=dict(status='verified',u=u,v=v,w=w,supplied_index=i,zero_candidate=j,
        supplied_coefficient=A,zero_coefficient=B,A=record['coefficients'][0,0,:n-1].tolist(),
        B=record['coefficients'][0,1,:n-1].tolist(),Q=q[0,:n-1].tolist(),
        maximum_levi_length_violation=violations,
        blocks=[dict(left=a,right=b,degree_excess=value[0],projected_v_left_descents=[a+k for k,x in enumerate(value[1]) if x]) for (a,b),value in blocks.items()],
        exact_product_queries=budget.exact_queries,exact_backend='lrcalc 2.1; fresh calls to the same exact backend, not an independent implementation.',
        root_product=info,source_sha256=s.sha(Path(__file__)),projection_sha256=s.sha(s.HERE/'demazure_projection.py'),
        conclusion='The necessary length inequalities for every Levi subset do not suffice to characterize positivity of the candidate coefficient, even when a positive supplied coefficient is given. This is a counterexample to that possible strengthening, not an external priority claim.')
    s.write('ALL_LEVI_COUNTEREXAMPLE.json',output);print(json.dumps(output),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('all_levi_counterexample_exact',120,query_limit=3);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
