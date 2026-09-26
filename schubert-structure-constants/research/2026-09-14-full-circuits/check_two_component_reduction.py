"""Compare all Levi tests with those having at most two components."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_features as rf
from demazure_projection import all_levi_restrictions
from full_resources import Budget


def main(budget):
    with np.load(s.ART/'ALL_LEVI_RESTRICTIONS.npz') as a:take=a['selected_rows'].copy()
    with np.load(s.ART/'first_exposed_S10_pairs.npz') as a:x=a['x'][take].copy();mask=a['mask'][take].copy()
    p,n=rf.permutations(torch.as_tensor(x));p=p.numpy();n=n.numpy();records=[];additional=[]
    for k in range(len(take)):
        budget.check();size=int(n[k]);u,v,w=p[k,:,:size].tolist();maximum,blocks=all_levi_restrictions(u,v,w)
        positive=[interval for interval,value in blocks.items() if value[0]>0]
        assert all(blocks[interval][0]==1 for interval in positive)
        assert all(not(a[1]<b[0] or b[1]<a[0]) for a in positive for b in positive)
        one=[];two=[]
        for j in range(1,size):
            containing=[(interval,value[0]-int(value[1][j-interval[0]])) for interval,value in blocks.items() if interval[0]<=j<interval[1]]
            connected=max(value for _,value in containing)
            reduced=max(value+max([0]+[blocks[other][0] for other in blocks if other[1]<interval[0] or interval[1]<other[0]]) for interval,value in containing)
            one.append(connected);two.append(reduced)
        assert two==maximum
        candidates=np.flatnonzero(mask[k])
        extra=[int(j+1) for j in candidates if one[j]<=0<two[j]]
        if extra:additional.append(dict(row=int(take[k]),indices=extra))
        records.append(dict(row=int(take[k]),all_levis=maximum,connected_levis=one,at_most_two_components=two,positive_blocks=positive))
    report=dict(status='verified',inputs=len(take),additional_exclusions_from_two_components=additional,
        source_sha256=s.sha(Path(__file__)),records=records,exact_product_queries=0,
        scope='The previously fixed 124 exposed error/control inputs; no new predictions or product queries.')
    s.write('TWO_COMPONENT_LEVI_REDUCTION.json',report);print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('two_component_levi_reduction',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
