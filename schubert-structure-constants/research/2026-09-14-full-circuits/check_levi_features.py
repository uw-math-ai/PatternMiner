"""Check tensor Bruhat projections against direct finite Weyl-group enumeration."""
import json
from itertools import permutations
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import levi_features as lf
from expanded_decomposition import bank
from full_resources import Budget


def rank(p):
    p=np.asarray(p)
    return (p[None,:]<=np.arange(1,len(p)+1)[:,None]).cumsum(1)


def scalar(h,w,start,m):
    def length(p):return sum(a>b for i,a in enumerate(p) for b in p[i+1:])
    values=sorted(permutations(range(1,m+1)),key=lambda p:(length(p),p))
    positions=sorted(w.index(a) for a in range(start,start+m))
    rh=rank(h);valid=[]
    for p in values:
        full=list(w)
        for k,pos in enumerate(positions):full[pos]=start+p[k]-1
        if (rh>=rank(full)).all():valid.append(p)
    assert valid
    least=valid[0]
    assert all((rank(least)>=rank(p)).all() for p in valid)
    return least


def main(budget):
    data,names,_=bank();p,n=lf.permutations(torch.as_tensor(data['x']))
    cases=np.unique(np.linspace(0,len(p)-1,24,dtype=int));comparisons=commutations=0
    for row in cases:
        budget.check();x=torch.as_tensor(data['x'][row:row+1]);actual=lf.projections(x)
        N=int(n[row]);values=p[row,:,:N].tolist();w=values[2]
        for m,entry in actual.items():
            for start in range(1,N-m+2):
                for side,h in enumerate(values[:2]):
                    expected=scalar(h,w,start,m)
                    assert tuple(entry['permutations'][0,side,start-1].tolist())==expected
                    comparisons+=1
                    for local in range(1,m):
                        i=start+local-1;down=list(h)
                        if h.index(i)>h.index(i+1):
                            down=[i+1 if a==i else i if a==i+1 else a for a in h]
                        projected=list(expected)
                        if projected.index(local)>projected.index(local+1):
                            projected=[local+1 if a==local else local if a==local+1 else a for a in projected]
                        assert tuple(projected)==scalar(down,w,start,m)
                        commutations+=1
    features=[];rejected=positives=0
    for begin in range(0,len(p),128):
        budget.check();x=torch.as_tensor(data['x'][begin:begin+128]);f=lf.features(x)
        assert f.shape==(len(x),16,48) and torch.isfinite(f).all()
        forbidden=(f[:,:,6::8]>1e-6).any(-1).numpy()&data['mask'][begin:begin+128]
        y=data['y'][begin:begin+128]
        assert not (forbidden&y).any(), (begin,np.argwhere(forbidden&y)[:10])
        rejected+=int(forbidden.sum());positives+=int(y.sum());features.append(f.numpy())
    archive=s.ART/'LEVI_FEATURES_DEVELOPMENT.npz'
    np.savez_compressed(archive,features=np.concatenate(features),population=data['population'],group=data['group'])
    result=dict(status='passed',source_sha256=s.sha(Path(__file__)),feature_source_sha256=s.sha(Path(lf.__file__)),
        direct_enumeration_comparisons=comparisons,downward_demazure_commutations=commutations,
        development_inputs=len(p),positive_target_entries=positives,excluded_candidate_entries=rejected,
        false_exclusions=0,populations=names,artifact_sha256=s.sha(archive),
        exact_product_queries=0,role='Implementation checks of consequences of the frozen parabolic-order theorem; not a learned nonvanishing rule')
    s.write('LEVI_FEATURE_CHECKS.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('levi_projection_feature_checks',600);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
