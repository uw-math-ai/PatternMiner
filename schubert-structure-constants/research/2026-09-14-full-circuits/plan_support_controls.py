"""Select same-factor donors before measuring any intervention effect."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from expanded_decomposition import bank
import richardson_features as rf
from full_resources import Budget


def main(budget):
    data,names,_=bank()
    permutations,sizes=rf.permutations(torch.as_tensor(data['x']))
    pp=permutations.numpy();nn=sizes.numpy()
    ii=np.rint(data['x'][:,0,1]*nn).astype(int)-1
    # Ordered factors, rather than their symmetry orbit, must agree exactly.
    keys=[(int(n),tuple(p[0,:n]),tuple(p[1,:n])) for p,n in zip(pp,nn)]
    by_factor={}
    for row,key in enumerate(keys):by_factor.setdefault(key,[]).append(row)
    selected=[];counts={}
    for row in range(len(pp)):
        budget.check()
        group=int(data['group'][row])
        if counts.get(group,0)>=8:continue
        n=int(nn[row]);pool=np.asarray(by_factor[keys[row]])
        valid=(data['y'][pool]==data['y'][row]).all(1)&data['a'][pool,ii[row]]
        valid&=(data['mask'][pool]==data['mask'][row]).all(1)
        valid&=(pp[pool,2,:n]!=pp[row,2,:n]).any(1)
        pool=pool[valid]
        if not len(pool):continue
        # Match L1 distance of inverse permutations. No model weights enter.
        dest=np.argsort(pp[row,2,:n]);opposite=np.argsort(pp[row^1,2,:n])
        distance=np.abs(opposite-dest).sum()
        donor_distance=np.abs(np.argsort(pp[pool,2,:n],axis=1)-dest).sum(1)
        discrepancy=np.abs(donor_distance-distance)/max(1,distance)
        best=int(np.argmin(discrepancy))
        if discrepancy[best]>.25:continue
        selected.append([row,int(pool[best]),int(distance),int(donor_distance[best])])
        counts[group]=counts.get(group,0)+1
    entries=np.asarray(selected,dtype=np.int64).reshape(-1,4)
    path=s.ART/'SUPPORT_CONTROL_MEMBERSHIP.npy';assert not path.exists();np.save(path,entries)
    report=dict(status='fixed_before_predictions_and_interventions',inputs=len(entries),factor_orbits=len(counts),
        bank_inputs=len(pp),populations=names,source_sha256=s.sha(Path(__file__)),artifact_sha256=s.sha(path),
        matching='Same ordered factors, n, candidate mask, and exact destination target support; supplied index positive in donor; distinct output permutation. L1 inverse-permutation distance to the destination within 25% of that of the opposite-support paired output. Closest distance, then bank order.',
        sampling='First eight eligible recorded inputs per canonical factor orbit, in frozen bank order; no correctness filter.',
        columns=['destination_bank_row','same_support_donor_bank_row','opposite_distance','same_support_distance'],
        limitation='Distance matching does not equate all other mathematical information. These are development controls, not formal confirmation.')
    s.write('SUPPORT_CONTROL_MEMBERSHIP.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('plan_support_controls',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
