"""Verify a retained obstruction for all candidate features and relations."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_model as rm
import richardson_features as rf
import models
import supplied_independent_model
import root_action_model
from demazure_projection import inversion_length
from full_resources import Budget


def roots(permutations,j):
    out={}
    for a in range(3):
        for b in range(3):
            if a==b:continue
            inverse=np.argsort(permutations[b]);composition=np.asarray(permutations[a])[inverse]
            out[f'{a}{b}']=[int(composition[j-1]),int(composition[j])]
    return out


def main(budget):
    source=s.ART/'EXPANDED_DECOMPOSITION.npz'
    with np.load(source) as archive:
        take=np.flatnonzero(archive['equal_candidate_features'])
        assert np.array_equal(take[::2]^1,take[1::2])
        data={k:archive[k][take].copy() for k in ['x','mask','y','a','group']}
    x=torch.as_tensor(data['x']);p,n=rf.permutations(x);p=p.numpy();n=n.numpy()
    torch.manual_seed(s.SEED);model=rm.extend(models.AllPairOrderSelector(4,False,4),'relative_roots')
    supplied_independent_model.use(model);root_action_model.use(model)
    with torch.inference_mode():features=model.feature_map(x).numpy();relations=model.comparisons(x).numpy()
    records=[]
    for k in range(0,len(take),2):
        budget.check();size=int(n[k]);left=p[k,:,:size];right=p[k+1,:,:size]
        candidates=np.flatnonzero(data['mask'][k]);indices=(candidates+1).tolist()
        assert np.array_equal(data['mask'][k],data['mask'][k+1])
        assert np.array_equal(features[k,candidates],features[k+1,candidates])
        assert np.array_equal(relations[k][np.ix_(candidates,candidates)],relations[k+1][np.ix_(candidates,candidates)])
        assert not (data['y'][k]&data['y'][k+1]).any()
        assert np.array_equal(left[:2],right[:2])
        exact=[]
        for j in indices:
            # Integer rank matrices at both candidate thresholds.
            ranks=[]
            for h in [left,right]:
                ranks.append([[(np.asarray(z)<=threshold).cumsum().tolist() for threshold in [j,j+1]] for z in h])
            assert ranks[0]==ranks[1]
            endpoints=[(np.argsort(h)+1)[:,[j-1,j]].tolist() for h in [left,right]]
            assert endpoints[0]==endpoints[1]
            relative=[roots(h,j) for h in [left,right]];assert relative[0]==relative[1]
            exact.append(dict(j=j,inverse_positions=endpoints[0],rank_columns=ranks[0],relative_root_endpoints=relative[0]))
        records.append(dict(n=size,source_rows=take[k:k+2].tolist(),u=left[0].tolist(),v=left[1].tolist(),
            w=left[2].tolist(),other_w=right[2].tolist(),lengths=[int(inversion_length(z)) for z in left],
            other_w_length=int(inversion_length(right[2])),candidate_indices=indices,
            positive_candidate_indices=[(np.flatnonzero(data['y'][i])+1).tolist() for i in [k,k+1]],
            supplied_indices=[(np.flatnonzero(data['a'][i])+1).tolist() for i in [k,k+1]],exact_candidate_data=exact))
    chosen=min(records,key=lambda r:(r['n'],sum(r['lengths'][:2]),r['u'],r['v'],r['w']))
    output=s.ART/'AUGMENTED_FEATURE_OBSTRUCTION.npz'
    np.savez_compressed(output,**data,source_rows=take,features=features,relations=relations)
    result=dict(status='verified_from_retained_data',source_sha256=s.sha(Path(__file__)),
        source_archive_sha256=s.sha(source),artifact_sha256=s.sha(output),pairs=len(records),
        factor_orbits=len(np.unique(data['group'])),feature_channels=features.shape[-1],relation_channels=relations.shape[-1],
        exact_product_queries=0,chosen=chosen,records=records,
        conclusion='Candidate feature vectors and all candidate-to-candidate relation tensors coincide, but the sets of valid witnesses are disjoint. A selector using only those data cannot solve both inputs.',
        limitation='Coefficients are the retained exact labels, not new independent oracle evaluations. This verifies an obstruction for these specified data, not for every possible local invariant.')
    s.write('AUGMENTED_FEATURE_OBSTRUCTION.json',result);print(json.dumps(dict(status=result['status'],pairs=len(records),chosen=chosen)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('augmented_feature_obstruction_serialization_retry',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
