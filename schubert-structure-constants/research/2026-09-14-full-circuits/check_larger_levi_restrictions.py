"""Test consecutive Levi length restrictions of semisimple ranks four to seven."""
import json,time
from pathlib import Path
import numpy as np
import torch
import session as s
from larger_levi_projections import projections
from full_resources import Budget


def main(budget):
    data=dict(np.load(s.ART/'first_exposed_S10_pairs.npz'))
    with np.load(s.ART/'SECOND_ERROR_COMPARISON.npz') as saved:
        scores=saved['FINAL_REFINEMENT_READOUTS_scores'].copy()
        good=saved['FINAL_REFINEMENT_READOUTS_good'].copy()
    bad_pairs=np.flatnonzero(~(good[::2]&good[1::2]))
    rng=np.random.default_rng(s.SEED+89100)
    control_pairs=rng.choice(np.flatnonzero(good[::2]&good[1::2]),32,replace=False)
    pairs=np.r_[bad_pairs,control_pairs];take=np.stack([2*pairs,2*pairs+1],1).reshape(-1)
    mask=data['mask'][take];y=data['y'][take];excluded=np.zeros_like(mask);counts={};details=[]
    for begin in range(0,len(take),2):
        budget.check();x=torch.as_tensor(data['x'][take[begin:begin+2]])
        for m,entry in projections(x).items():
            invalid=entry['lengths'][:,0]+entry['lengths'][:,1]-entry['x_length']
            remove=invalid[:,:,None]-entry['descents'][:,1].int()>0
            remove&=entry['valid'][:,:,None]
            local=np.zeros((len(x),16),bool)
            for k,start in enumerate(entry['starts'].tolist()):
                local[:,start-1:start+m-2]|=remove[:,k].numpy()
            assert not (local&y[begin:begin+len(x)]).any(),(begin,m)
            excluded[begin:begin+len(x)]|=local
            counts[m-1]=counts.get(m-1,0)+int((local&mask[begin:begin+len(x)]).sum())
        if begin%20==0:print(json.dumps(dict(completed_inputs=begin+2,total_inputs=len(take))),flush=True)
    allowed=mask&~excluded;assert (allowed&y).any(1).all()
    predictions=np.where(allowed,scores[take],-1e9).argmax(1)
    after=y[np.arange(len(take)),predictions];before=good[take]
    output=s.ART/'LARGER_LEVI_RESTRICTIONS.npz'
    np.savez_compressed(output,selected_rows=take,excluded=excluded,before=before,after=after,predictions=predictions)
    report=dict(status='completed',source_sha256=s.sha(Path(__file__)),projection_source_sha256=s.sha(s.HERE/'larger_levi_projections.py'),
        inputs=len(take),originally_failed_pairs=len(bad_pairs),correct_control_pairs=len(control_pairs),
        initially_incorrect_inputs=int((~before).sum()),remaining_incorrect_inputs=int((~after).sum()),
        fixed=int((~before&after).sum()),lost=int((before&~after).sum()),
        candidate_exclusions_by_semisimple_rank=counts,excluded_valid_targets=0,
        exact_product_queries=0,artifact_sha256=s.sha(output),
        scope='All thirty newly exposed S10 pair failures and thirty-two deterministic correct control pairs. Test every consecutive block with five through eight values. Inference uses only permutations and the proven necessary length condition; exact targets are used for evaluation only. No prediction on the replacement assessment is read.')
    s.write('LARGER_LEVI_RESTRICTIONS.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    # Leave the two designated CPU threads to exact generation until it finishes.
    while not (s.HERE/'SECOND_NEIGHBOR_PREPARED.json').exists():time.sleep(2)
    torch.set_num_threads(1);budget=Budget('larger_levi_restriction_check',300);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
