"""Bruhat projections to consecutive Levi blocks of five through eight values."""
from functools import lru_cache
from itertools import permutations as all_permutations
import numpy as np
import torch
from richardson_features import permutations

CHANNELS=None


@lru_cache(None)
def tables(m,device):
    def length(p):return sum(a>b for i,a in enumerate(p) for b in p[i+1:])
    values=sorted(all_permutations(range(1,m+1)),key=lambda p:(length(p),p))
    p=torch.tensor(values,device=device)
    rank=((p[:,None,:]<=torch.arange(1,m,device=device)[None,:,None]).int().cumsum(-1))[:,:,:m-1]
    lengths=torch.tensor([length(p) for p in values],device=device)
    inverse=p.argsort(-1)
    desc=inverse[:,:-1]>inverse[:,1:]
    return p,rank,lengths,desc


def projections(x, sizes=(5,6,7,8)):
    p,n=permutations(x)
    inverse=p.masked_fill(p==0,99).argsort(-1)
    thresholds=torch.arange(1,18,device=x.device)
    ranks=((p[:,:,None,:]<=thresholds[None,None,:,None])&(p[:,:,None,:]>0)).int().cumsum(-1)
    ranks=torch.nn.functional.pad(ranks,(1,0,1,0))
    result={}
    for m in sizes:
        starts=torch.arange(1,min(19-m,int(n.max())-m+2),device=x.device)
        if not len(starts):continue
        values=starts[:,None]+torch.arange(m,device=x.device)[None,:]-1
        positions=inverse[:,2,values].sort(-1).values
        occupied=(positions[:,:,:,None]<torch.arange(18,device=x.device)[None,None,None,:]).sum(2)
        levels=starts[:,None]+torch.arange(m-1,device=x.device)[None,:]
        difference=ranks[:,:2,levels,:]-ranks[:,2,starts-1,:][:,None,:,None,:]
        bounds=torch.where(occupied[:,None,:,None,None,:]==torch.arange(1,m,device=x.device)[None,None,None,None,:,None],
                           difference[:,:,:,:,None,:],99).amin(-1)
        local,rp,lengths,desc=tables(m,str(x.device))
        allowed=(rp[None,None,None,:,:,:]<=bounds[:,:,:,None,:,:]).all(-1).all(-1)
        choice=allowed.int().argmax(-1)
        valid=starts[None,:]+m-1<=n[:,None]
        assert (allowed.any(-1)|~valid[:,None,:]).all()
        xp=p[:,2].gather(1,positions.flatten(1)).reshape(positions.shape)-starts[None,:,None]+1
        ellx=(xp[:,:,:,None]>xp[:,:,None,:]).triu(1).sum((-1,-2))
        result[m]=dict(starts=starts,valid=valid,permutations=local[choice],lengths=lengths[choice],
                       descents=desc[choice],x=xp,x_length=ellx,bounds=bounds)
    return result

