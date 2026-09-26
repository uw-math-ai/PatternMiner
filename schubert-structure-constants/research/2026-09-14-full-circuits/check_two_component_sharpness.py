"""Retain an explicit zero detected by two components but no connected Levi."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_features as rf
from demazure_projection import all_levi_restrictions,block_projection,inversion_length
from full_resources import Budget


def main(budget):
    source=json.loads((s.HERE/'TWO_COMPONENT_LEVI_REDUCTION.json').read_text())
    items=source['additional_exclusions_from_two_components']
    rows=np.array([item['row'] for item in items])
    with np.load(s.ART/'first_exposed_S10_pairs.npz') as a:
        data={key:a[key][rows].copy() for key in ['x','mask','y','a']}
    p,n=rf.permutations(torch.as_tensor(data['x']));p=p.numpy();n=n.numpy()
    k=min(range(len(rows)),key=lambda k:(int(n[k]),sum(int(inversion_length(z[:n[k]])) for z in p[k,:2]),p[k,:,:n[k]].tolist()))
    size=int(n[k]);u,v,w=p[k,:,:size].tolist();j=min(items[k]['indices'])
    assert data['a'][k].any() and data['mask'][k,j-1] and not data['y'][k,j-1]
    maximum,blocks=all_levi_restrictions(u,v,w)
    containing=[(iv,val[0]-int(val[1][j-iv[0]])) for iv,val in blocks.items() if iv[0]<=j<iv[1]]
    assert max(value for _,value in containing)==0 and maximum[j-1]==1
    witnesses=[(iv,other) for iv,value in containing for other,oval in blocks.items()
        if (iv[1]<other[0] or other[1]<iv[0]) and value+oval[0]>0]
    intervals=min(witnesses,key=lambda pair:(sum(b-a for a,b in pair),pair))
    projections=[]
    for L,R in sorted(intervals):
        a,la,_,lx=block_projection(u,w,L,R);b,lb,desc,_=block_projection(v,w,L,R)
        projections.append(dict(value_interval=[L,R],simple_root_indices=list(range(L,R)),u_projection=a,v_projection=b,
            w_levi=[z-L+1 for z in w if L<=z<=R],lengths=[la,lb,lx],degree_excess=la+lb-lx,
            candidate_in_component=L<=j<R,projected_v_has_candidate_descent=bool(desc[j-L]) if L<=j<R else None))
    output=dict(status='verified_from_retained_data',u=u,v=v,w=w,lengths=[int(inversion_length(z)) for z in [u,v,w]],
        supplied_indices=(np.flatnonzero(data['a'][k])+1).tolist(),zero_candidate=j,source_row=int(rows[k]),
        maximum_connected_violation=0,maximum_all_levi_violation=1,projections=projections,
        all_connected_tests=[dict(value_interval=list(iv),violation=value) for iv,value in containing],
        source_sha256=s.sha(Path(__file__)),reduction_report_sha256=s.sha(s.HERE/'TWO_COMPONENT_LEVI_REDUCTION.json'),
        exact_product_queries=0,conclusion='Two components can detect a zero that no connected Levi necessary length inequality detects. The degree-one bound of two components cannot be reduced to one for this family of tests.')
    s.write('TWO_COMPONENT_LEVI_SHARPNESS.json',output);print(json.dumps(output),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('two_component_levi_sharpness',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
