"""Combine frozen and fresh training sources without copying existing caches."""
import json
import numpy as np
import torch
import context as c
import compact
import auxiliary
import training_sampling

SPEC='FINAL_NEIGHBOR_BALANCED_TRAINING_DATA.json'


def load(fresh=False,adaptive=False,screened=False):
    data,eligible,offsets,weights=c.previous_training.training_tensors(True,SPEC)
    roots,_=auxiliary.load_roots(SPEC)
    layouts=training_sampling.layouts(SPEC,offsets)
    hard=torch.as_tensor(np.load(c.OLD/'artifacts/neighbor_complete_training_pairs.npy'),device='cuda')
    hard_layout={k:torch.as_tensor(v,device='cuda') for k,v in dict(np.load(c.OLD/'artifacts/neighbor_training_hard_groups.npz')).items()}
    fresh_pairs=None;fresh_layout=None
    if fresh:
        report=json.loads((c.HERE/'DATA.json').read_text());assert report['status']=='completed'
        prepared=json.loads((c.HERE/'PREPARED.json').read_text());assert prepared['status']=='completed'
        root_report=json.loads((c.HERE/'ROOT_TARGETS.json').read_text());assert root_report['status']=='completed'
        pieces={k:[v] for k,v in data.items()};qq=[roots];ee=[eligible];pairs=[];pairgroups=[];groupoffset=0
        weights=[.60*w for w in weights]
        for name,weight in [('S8_train',.01),('S9_train',.24),('S10_train',.15)]:
            spec=report['populations'][name];path=c.HERE/f'cache/data/{name}.bin'
            assert c.sha(path)==spec['sha256'];records=compact.load(path)
            qpath=c.HERE/f'cache/data/{name}_Q.npy';assert c.sha(qpath)==root_report['populations'][name]['sha256']
            for k in data:
                values=np.array(records[k],copy=True)
                if k=='coefficients':values=values.astype(np.int32)
                pieces[k].append(torch.as_tensor(values,device='cuda'))
            qq.append(torch.as_tensor(np.load(qpath).astype(np.int32),device='cuda'))
            ee.append(torch.ones(len(records),dtype=torch.bool,device='cuda'))
            pp=np.load(c.ART/f'{name}_pairs.npy');assert c.sha(c.ART/f'{name}_pairs.npy')==prepared['training'][name]['pair_sha256']
            pairgroups.append(records['group'][pp[:,0]].astype(np.int64)+groupoffset)
            pp=pp.copy();pp[:,:2]+=offsets[-1];pairs.append(pp);groupoffset+=len(spec['offsets'])-1
            rows=np.asarray(spec['offsets'])+offsets[-1];nonempty=np.flatnonzero(np.diff(rows)>0)
            layouts.append(dict(parents=torch.arange(len(nonempty)+1,device='cuda'),
                                groups=torch.as_tensor(nonempty,device='cuda'),rows=torch.as_tensor(rows,device='cuda')))
            offsets.append(offsets[-1]+len(records));weights.append(weight)
        data={k:torch.cat(v) for k,v in pieces.items()};roots=torch.cat(qq);eligible=torch.cat(ee)
        fresh_pairs=torch.as_tensor(np.concatenate(pairs),device='cuda')
        groups=np.concatenate(pairgroups);unique,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
        order=np.argsort(inverse,kind='stable');bounds=np.r_[0,np.cumsum(counts)]
        fresh_layout={k:torch.as_tensor(v,device='cuda') for k,v in dict(new_order=order,new_offsets=bounds,new_weights=np.ones(len(unique))/len(unique)).items()}
    if adaptive:
        assert fresh
        import adaptive_training
        data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout=adaptive_training.append(data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout)
    if screened:
        assert fresh
        import adaptive_training
        data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout=adaptive_training.append(data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout,prefix='screened')
    assert abs(sum(weights)-1)<1e-8
    return data,eligible,offsets,weights,roots,layouts,hard,hard_layout,fresh_pairs,fresh_layout


def validation(fresh=False):
    data=c.previous_training.validation_data(True,True)
    if fresh:
        for n in [9,10]:
            for suffix in ['', '_pairs']:
                name=f'fresh_S{n}_validation{suffix}'
                data[name]=dict(np.load(c.ART/f'{name}.npz'))
    return data
