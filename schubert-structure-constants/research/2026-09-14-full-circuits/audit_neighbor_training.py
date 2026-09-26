"""Check the actual concatenated tensors and pair offsets before training."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import compact
import neighbor_training_data as data
from exposed_assessment import append_validation
from full_resources import Budget


def main(budget):
    values=data.load(True,True,True)
    tr,eligible,offsets,weights,roots,layouts,hard,hard_layout,pairs,pair_layout=values
    meta=json.loads((s.HERE/'NEIGHBOR_DATA.json').read_text())
    prepared=json.loads((s.HERE/'NEIGHBOR_PREPARED.json').read_text())
    count=prepared['pairs'];base=offsets[-2]
    raw=compact.load(s.HERE/'cache/data/neighbors_train.bin')
    for key in tr:
        for row in [0,len(raw)//2,len(raw)-1]:
            assert np.array_equal(tr[key][base+row].cpu().numpy(),raw[key][row])
    rng=np.random.default_rng(s.SEED+840)
    take=rng.choice(np.arange(len(pairs)-count,len(pairs)),min(1024,count),replace=False)
    chosen=pairs[take];indices=chosen[:,:2].reshape(-1)
    records={k:v[indices] for k,v in tr.items()}
    x,mask,y,a=compact.encode(records,chosen[:,2].bool().repeat_interleave(2),chosen[:,3].repeat_interleave(2))
    assert torch.equal(mask[::2],mask[1::2]) and not (y[::2]&y[1::2]).any()
    assert torch.equal(x[::2][mask[::2]],x[1::2][mask[1::2]])
    assert a[torch.arange(len(x),device='cuda'),chosen[:,3].repeat_interleave(2)].all()
    assert (roots[indices,None,:]>=records['coefficients']).all()
    validation=append_validation(data.validation(True));assert len(validation)==19
    for name,pop in validation.items():
        assert len(pop['x'])==len(pop['mask'])==len(pop['y'])==len(pop['a'])
    report=dict(status='passed',rows=offsets[-1],new_rows=len(raw),new_pairs=count,pairs_checked=len(take),
        old_rows=base,source_offsets=offsets,ordinary_weights=weights,validation_populations={k:len(v['x']) for k,v in validation.items()},
        screened_error_offset=data.screened_offset,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        data_metadata_sha256=s.sha(s.HERE/'NEIGHBOR_DATA.json'),prepared_sha256=s.sha(s.HERE/'NEIGHBOR_PREPARED.json'),
        source_sha256=s.sha(Path(__file__)),loader_sha256=s.sha(Path(data.__file__)),
        checks=['Actual appended fields equal source records at three offsets','1024 sampled new pairs preserve common candidate features, positive supplied index, and disjoint target supports','Auxiliary root coefficients dominate both ordinary coefficient vectors on sampled rows','Nineteen distinct development populations loaded'])
    s.write('NEIGHBOR_TRAINING_AUDIT.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('neighbor_training_audit',180,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
