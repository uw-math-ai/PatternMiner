"""Pure auxiliary root targets; no file writes and one exact product call."""
import hashlib,json
import numpy as np
import lrcalc
import session as s
from session import c

def root_targets(records,pair,group):
    begin,end=0,len(records)
    trim=c.prior.c.a.original.trim
    raw=lrcalc.schubmult(list(trim(tuple(pair['u']))),list(trim(tuple(pair['v']))))
    product={trim(tuple(w)):int(value) for w,value in raw.items()}
    roots=np.zeros((len(records),16),np.int16)
    for k,row in enumerate(records):
        n=int(row['n']);w=tuple(map(int,row['permutations'][2,:n]))+(n+1,);inverse=np.argsort(w)
        f=np.zeros(n,np.int64);f[:n-1]=-row['coefficients'][:,:n-1].astype(np.int64).sum(0)
        for i in range(n):
            if inverse[i]<inverse[i+1]:
                changed=list(w);changed[inverse[i]],changed[inverse[i+1]]=changed[inverse[i+1]],changed[inverse[i]]
                f[i]+=product.get(trim(tuple(changed)),0)
        assert np.dot(np.arange(1,n+1),f)==0
        q=np.array([np.dot(np.arange(1,n-j),f[j+1:]) for j in range(n-1)])
        assert (q>=0).all() and (q[None,:]>=row['coefficients'][:,:n-1]).all()
        padded=np.r_[0,q,0];assert np.array_equal(2*q-padded[:-2]-padded[2:],-f[:n-1])
        assert q.max(initial=0)<=32767;roots[k,:n-1]=q
    return roots,dict(group=group,begin=begin,end=end,product_terms=len(product),
        product_sha256=hashlib.sha256(json.dumps(sorted(product.items()),separators=(',',':')).encode()).hexdigest())

