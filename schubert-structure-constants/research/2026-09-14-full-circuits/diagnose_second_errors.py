"""Compare retained predictions on the newly exposed difficult S10 population."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_model as rm
from full_resources import Budget


def main(budget):
    path=s.ART/'first_exposed_S10_pairs.npz';data=dict(np.load(path));results={};arrays={}
    names=['FINAL_REFINEMENT_READOUTS','neighbor_refinement_final','neighbor_refinement_step007500','pinned_error_control_common5000']
    for name in names:
        checkpoint=s.ART/(name+'.pt');model=rm.load(checkpoint).eval();scores=[]
        with torch.inference_mode():
            for start in range(0,len(data['x']),512):
                budget.check();scores.append(model(torch.as_tensor(data['x'][start:start+512],device='cuda')).cpu().numpy())
        score=np.concatenate(scores);pred=np.where(data['mask'],score,-1e9).argmax(1)
        good=data['y'][np.arange(len(pred)),pred];pairs=good[::2]&good[1::2]
        results[name]=dict(checkpoint_sha256=s.sha(checkpoint),pairs=len(pairs),correct_pairs=int(pairs.sum()),accuracy=float(pairs.mean()))
        arrays[name+'_scores']=score;arrays[name+'_good']=good;arrays[name+'_pairs']=pairs
        del model
    baseline=arrays[names[0]+'_pairs']
    for name in names:
        good=arrays[name+'_pairs'];results[name].update(fixed=int((~baseline&good).sum()),lost=int((baseline&~good).sum()))
    correct_union=np.stack([arrays[name+'_pairs'] for name in names]).any(0)
    np.savez_compressed(s.ART/'SECOND_ERROR_COMPARISON.npz',**arrays)
    s.write('SECOND_ERROR_COMPARISON.json',dict(status='completed',models=results,union_correct_pairs=int(correct_union.sum()),
        union_role='Oracle comparison of retained predictions, not an executable selection rule.',source_sha256=s.sha(Path(__file__)),
        data_sha256=s.sha(path),role='Development diagnosis after the first independent assessment failed; replacement assessment remains unopened.'))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('second_error_checkpoint_comparison',120,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
