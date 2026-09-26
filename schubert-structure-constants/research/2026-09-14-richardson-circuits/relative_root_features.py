"""Images of simple roots under x y^{-1}, for distinct x,y in {u,v,w}."""
import torch
from richardson_features import permutations

PAIRS=[(a,b) for a in range(3) for b in range(3) if a!=b]
CHANNELS=24


def features(x):
    p,n=permutations(x);j=torch.arange(1,17,device=x.device)[None,:]
    inverse=p.masked_fill(p==0,99).argsort(-1)
    result=[]
    for a,b in PAIRS:
        composition=p[:,a].gather(1,inverse[:,b])
        left=composition[:,:16];right=composition[:,1:17]
        coefficient=((left<=j)&(j<right)).float()-((right<=j)&(j<left)).float()
        result.extend([(left-j).float()/n[:,None],(right-j).float()/n[:,None],(right-left).sign().float(),coefficient])
    return torch.stack(result,-1)*(j<n[:,None])[:,:,None]
