"""Append adaptive examples with equal sampling across parent families."""
import json
import numpy as np
import torch
import session as s
from session import c
import compact


def append(data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout,prefix="neighbors"):
    report=json.loads((s.HERE/'NEIGHBOR_DATA.json').read_text());assert report['status']=='completed'
    prepared=json.loads((s.HERE/'NEIGHBOR_PREPARED.json').read_text());assert prepared['status']=='completed'
    plan=json.loads((s.HERE/'NEIGHBOR_PLAN.json').read_text());assert c.sha(s.HERE/'NEIGHBOR_PLAN.json')==report['plan_sha256']
    path=s.HERE/f'cache/data/{prefix}_train.bin';qpath=s.HERE/f'cache/data/{prefix}_Q.npy'
    assert c.sha(path)==report['data_sha256'] and c.sha(qpath)==report['root_sha256']
    raw=compact.load(path);base=offsets[-1]
    for key,value in data.items():
        additional=np.array(raw[key],copy=True)
        if key=='coefficients':additional=additional.astype(np.int32)
        data[key]=torch.cat([value,torch.as_tensor(additional,device='cuda')])
    roots=torch.cat([roots,torch.as_tensor(np.load(qpath).astype(np.int32),device='cuda')])
    eligible=torch.cat([eligible,torch.ones(len(raw),dtype=torch.bool,device='cuda')])
    by_parent=[[] for _ in plan.get('anchors',plan['pairs'])]
    for group,pair in enumerate(plan['pairs']):
        if report['offsets'][group+1]>report['offsets'][group]:by_parent[pair['parent']].append(group)
    by_parent=[groups for groups in by_parent if groups]
    bounds=np.r_[0,np.cumsum([len(groups) for groups in by_parent])]
    layouts.append(dict(parents=torch.as_tensor(bounds,device='cuda'),
        groups=torch.as_tensor([g for groups in by_parent for g in groups],device='cuda'),
        rows=torch.as_tensor(np.asarray(report['offsets'])+base,device='cuda')))
    offsets.append(base+len(raw));weights=[.8*w for w in weights]+[.2]
    path=s.ART/f'{prefix}_training_pairs.npy';assert c.sha(path)==prepared['pair_sha256']
    pairs=np.load(path);groups=raw['group'][pairs[:,0]];pairs=pairs.copy();pairs[:,:2]+=base
    unique,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    parents=np.asarray([p['parent'] for p in plan['pairs']])[unique]
    active,number=np.unique(parents,return_counts=True)
    parent_counts=dict(zip(active,number));group_weights=np.asarray([.5/(len(active)*parent_counts[p]) for p in parents])
    original_count=len(fresh_pairs);old=fresh_layout
    fresh_pairs=torch.cat([fresh_pairs,torch.as_tensor(pairs,device='cuda')])
    fresh_layout=dict(new_order=torch.cat([old['new_order'],torch.as_tensor(np.argsort(inverse,kind='stable')+original_count,device='cuda')]),
        new_offsets=torch.cat([old['new_offsets'],torch.as_tensor(np.cumsum(counts)+original_count,device='cuda')]),
        new_weights=torch.cat([old['new_weights']*.5,torch.as_tensor(group_weights,device='cuda')]))
    assert int(fresh_layout['new_offsets'][-1])==len(fresh_pairs)
    assert abs(float(fresh_layout['new_weights'].sum())-1)<1e-8
    return data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout


screened_offset=None

def load(fresh=False,adaptive=False,screened=False):
    global screened_offset
    import final_training_data as base
    values=base.load(fresh,adaptive,screened)
    data,eligible,offsets,weights,roots,layouts,hard,hard_layout,fresh_pairs,fresh_layout=values
    screened_offset=base.screened_offset
    data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout=append(
        data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout)
    assert len(data['n'])==offsets[-1]==len(roots)==len(eligible)
    assert len(layouts)==len(weights)==len(offsets)-1
    assert int(fresh_pairs[:,:2].max())<offsets[-1]
    return data,eligible,offsets,weights,roots,layouts,hard,hard_layout,fresh_pairs,fresh_layout


def validation(fresh=False):
    import final_training_data as base
    result=base.validation(fresh)
    prepared=json.loads((s.HERE/'PREPARED.json').read_text())
    assert prepared['status']=='completed'
    for name in prepared['development']:
        for suffix in ['', '_pairs']:
            key=f'new_{name}{suffix}'
            result[key]=dict(np.load(s.ART/(key+'.npz')))
    return result
