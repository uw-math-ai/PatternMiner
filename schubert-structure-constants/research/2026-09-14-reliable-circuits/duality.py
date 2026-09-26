"""Poincaré duality for degree-one training triples; no inference oracle."""
import json
import numpy as np
import torch
import common as r


def transform(data,roots,anchor):
    """Fix factor anchor and replace the other factor by w0*w."""
    n=data['n'].long();p=data['permutations'];c=data['coefficients'].long()
    q=roots.long();index=torch.arange(16,device=n.device)
    real=index[None,:]<n[:,None]-1
    assert (q>=0).all()
    padded=torch.nn.functional.pad(q,(1,1))
    d=c.sum(1)-2*q+padded[:,:-2]+padded[:,2:]
    d=d*real
    assert (d>=0).all() and d.any(1).all()
    reverse_index=(n[:,None]-2-index).clamp(min=0)
    reversed_d=d.gather(1,reverse_index)*real
    reversed_q=q.gather(1,reverse_index)*real
    batch=torch.arange(len(n),device=n.device)
    fixed=c[batch,anchor]
    j=index[None,:,None]+1;k=index[None,None,:]+1
    kernel=torch.minimum(torch.minimum(j,k),torch.minimum(n[:,None,None]-j,n[:,None,None]-k))
    new_q=((kernel*fixed[:,None,:]).sum(-1)-reversed_q)*real
    assert (new_q>=0).all()
    positions=torch.arange(17,device=n.device)[None,:]<n[:,None]
    complement=lambda a:((n[:,None]+1-a.long())*positions).to(p.dtype)
    permutations=torch.stack([p[batch,anchor],complement(p[:,2]),complement(p[batch,1-anchor])],1)
    length=data['lengths'].long();top=n*(n-1)//2
    lengths=torch.stack([length[batch,anchor],top-length[:,2],top-length[batch,1-anchor]],1)
    assert (lengths[:,0]+lengths[:,1]-1==lengths[:,2]).all()
    return dict(permutations=permutations,n=data['n'],lengths=lengths.to(data['lengths'].dtype),
        coefficients=torch.stack([fixed,reversed_d],1).to(data['coefficients'].dtype)),new_q.to(roots.dtype)


def load_eligibility(training_spec='TRAINING_DATA.json'):
    spec=json.loads((r.HERE/'POINCARE_ELIGIBILITY.json').read_text())
    assert spec['status']=='completed'
    sources=json.loads((r.HERE/training_spec).read_text())['sources']
    pieces=[]
    for name,item in sources.items():
        if name not in spec['populations']:
            values=np.zeros((item['rows'],2),bool)
        else:
            path=r.HERE/f'cache/duality/{name}.npy'
            assert r.c.a.sha(path)==spec['populations'][name]['sha256']
            values=np.load(path)
            assert values.shape==(item['rows'],2)
        pieces.append(torch.as_tensor(values,device='cuda'))
    return torch.cat(pieces),spec
