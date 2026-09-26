"""Bruhat projections to consecutive Levi blocks of two, three or four values."""
from functools import lru_cache
from itertools import permutations as all_permutations
import numpy as np
import torch
from richardson_features import permutations

CHANNELS=48


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


def projections(x):
    p,n=permutations(x)
    inverse=p.masked_fill(p==0,99).argsort(-1)
    thresholds=torch.arange(1,18,device=x.device)
    ranks=((p[:,:,None,:]<=thresholds[None,None,:,None])&(p[:,:,None,:]>0)).int().cumsum(-1)
    ranks=torch.nn.functional.pad(ranks,(1,0,1,0))
    result={}
    for m in [2,3,4]:
        starts=torch.arange(1,19-m,device=x.device)
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


def features(x):
    parts=[]
    for m,entry in projections(x).items():
        ell=entry['lengths'];ex=entry['x_length'];valid=entry['valid'];maximum=m*(m-1)/2
        for offset in range(m-1):
            du=entry['descents'][:,0,:,offset];dv=entry['descents'][:,1,:,offset]
            values=torch.stack([ell[:,0]/maximum,ell[:,1]/maximum,ex/maximum,
                (ell[:,0]+ell[:,1]-ex)/maximum,du.float(),dv.float(),
                (ell[:,0]+ell[:,1]-dv.int()-ex)/maximum,torch.ones_like(ex)],-1)
            block=x.new_zeros((len(x),16,8));block[:,offset:offset+values.shape[1]]=values*valid[:,:,None]
            parts.append(block)
    return torch.cat(parts,-1)
