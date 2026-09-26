"""Training labels for ordinary and degree-one equivariant coefficients."""
import json
import numpy as np
import torch
import common as r


def load_roots(training_spec='TRAINING_DATA.json'):
    specification=json.loads((r.HERE/'ROOT_TARGETS.json').read_text())
    assert specification['status']=='completed'
    sources=json.loads((r.HERE/training_spec).read_text())['sources']
    pieces=[]
    for name,entry in sources.items():
        if 'root_targets' in entry:
            path=r.HERE/entry['root_targets']['path']
            assert r.c.a.sha(path)==entry['root_targets']['sha256']
            values=np.load(path)
            assert values.shape==(entry['rows'],16) and (values>=0).all()
        elif name not in specification['populations']:
            values=np.full((entry['rows'],16),-1,np.int16)
        else:
            path=r.HERE/f'cache/root_targets/{name}.npy'
            assert r.c.a.sha(path)==specification['populations'][name]['sha256']
            values=np.load(path)
            assert values.shape==(entry['rows'],16)
        # CUDA advanced indexing does not support uint16.
        pieces.append(torch.as_tensor(values.astype(np.int32),device='cuda'))
    return torch.cat(pieces),specification


def targets(data,roots,reverse,reflection):
    """Return log-coefficient targets and availability; these are never inputs."""
    counts=data['coefficients'].float().clone()
    counts[reverse]=counts[reverse][:,[1,0]]
    target=torch.cat([counts.transpose(1,2),roots.float().unsqueeze(-1)],-1)
    n=data['n'].long()
    index=torch.arange(16,device=n.device)
    if reflection.any():
        mirrored=(n[:,None]-2-index[None,:]).clamp(min=0)
        target[reflection]=target[reflection].gather(1,mirrored[reflection,:,None].expand(-1,-1,3))
    real=index[None,:]<n[:,None]-1
    available=real[:,:,None].expand(-1,-1,3)&(target>=0)
    return target.masked_fill(~available,0).log1p(),available


def loss(prediction,target,available):
    errors=(prediction-target).square()*available
    return errors.sum()/available.sum().clamp(min=1)
