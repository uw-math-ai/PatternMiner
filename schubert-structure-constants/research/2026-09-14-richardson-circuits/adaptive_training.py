"""Append adaptive examples with equal sampling across parent families."""
import json
import numpy as np
import torch
import context as c
import compact


def append(data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout,prefix="adaptive"):
    report=json.loads((c.HERE/f'{prefix.upper()}_DATA.json').read_text());assert report['status']=='completed'
    prepared=json.loads((c.HERE/f'{prefix.upper()}_PREPARED.json').read_text());assert prepared['status']=='completed'
    plan=json.loads((c.HERE/f'{prefix.upper()}_PLAN.json').read_text());assert c.sha(c.HERE/f'{prefix.upper()}_PLAN.json')==report['plan_sha256']
    path=c.HERE/f'cache/data/{prefix}_train.bin';qpath=c.HERE/f'cache/data/{prefix}_Q.npy'
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
    path=c.ART/f'{prefix}_training_pairs.npy';assert c.sha(path)==prepared['pair_sha256']
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
