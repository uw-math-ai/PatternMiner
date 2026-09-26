"""Input-derived reduced expressions and positive distinguished subexpressions."""
import torch

MAX_STEPS=136
CHANNELS=20


def permutations(x):
    n=(1/x[:,0,18]).round().long();rows=torch.arange(len(x),device=x.device)
    positions=torch.zeros((len(x),3,17),device=x.device,dtype=torch.long)
    positions[:,:,:16]=(x[:,:,[3,5,7]]*n[:,None,None]).round().long().transpose(1,2)
    positions[rows,:,n-1]=(x[rows,n-2][:,[4,6,8]]*n[:,None]).round().long()
    values=torch.arange(17,device=x.device)[None,None,:].expand_as(positions)
    active=values<n[:,None,None]
    locations=torch.where(active,positions-1,values)
    p=torch.zeros_like(positions).scatter(2,locations,torch.where(active,values+1,0))
    return p,n


def expressions(p,n):
    """Use the reverse of first-descent bubble sorting as a reduced expression."""
    size=torch.arange(17,device=p.device)
    current=torch.where(size[None,None,:]<n[:,None,None],p,size[None,None,:]+1).clone()
    lengths=((current[:,:,:,None]>current[:,:,None,:])&
             (size[:,None]<size[None,:])[None,None,:,:]).sum((-2,-1))
    m=int(lengths[:,2].max());rows=torch.arange(len(p),device=p.device)
    word=torch.zeros((len(p),m),device=p.device,dtype=torch.long)
    selected=torch.zeros((len(p),m,2),device=p.device,dtype=torch.bool)
    for t in range(m):
        active=t<lengths[:,2]
        i=(current[:,2,:-1]>current[:,2,1:]).long().argmax(1)
        slot=(lengths[:,2]-1-t).clamp(min=0)
        word[rows,slot]=torch.where(active,i,word[rows,slot])
        left=current.gather(2,i[:,None,None].expand(-1,3,1)).squeeze(2)
        right=current.gather(2,(i+1)[:,None,None].expand(-1,3,1)).squeeze(2)
        chosen=(left>right)&active[:,None]
        selected[rows,slot]=torch.where(active[:,None],chosen[:,:2],selected[rows,slot])
        current.scatter_(2,i[:,None,None].expand(-1,3,1),torch.where(chosen,right,left)[:,:,None])
        current.scatter_(2,(i+1)[:,None,None].expand(-1,3,1),torch.where(chosen,left,right)[:,:,None])
    return word,selected,lengths,current


def features(x,return_combinatorics=False):
    p,n=permutations(x);word,selected,lengths,terminal=expressions(p,n)
    batch=len(x);m=word.shape[1]
    result=torch.zeros((batch,16,MAX_STEPS,CHANNELS),device=x.device,dtype=torch.float32)
    prefix=torch.arange(1,18,device=x.device)[None,None,:].expand(batch,3,-1).clone()
    prefix_lengths=torch.zeros((batch,3),device=x.device,dtype=torch.long)
    j=torch.arange(1,17,device=x.device)[None,:]
    scale=n[:,None,None].float()
    for k in range(m):
        i=word[:,k];active=k<lengths[:,2]
        left=prefix.gather(2,i[:,None,None].expand(-1,3,1)).squeeze(2)
        right=prefix.gather(2,(i+1)[:,None,None].expand(-1,3,1)).squeeze(2)
        endpoints=torch.stack([left,right],-1).reshape(batch,6)
        interval=(((left[:,None,:]<=j[:,:,None])&(j[:,:,None]<right[:,None,:])).float()-
                  ((right[:,None,:]<=j[:,:,None])&(j[:,:,None]<left[:,None,:])).float())
        inverse=prefix[:,:2].argsort(-1)+1
        inverse=torch.stack([inverse[:,:,:16],inverse[:,:,1:17]],-1).permute(0,2,1,3).reshape(batch,16,4)
        channels=torch.cat([
            ((i[:,None]+1-j).float()/n[:,None])[:,:,None],
            selected[:,k,None,:].expand(-1,16,-1).float(),
            (endpoints[:,None,:]-j[:,:,None]).float()/scale,
            interval,
            inverse.float()/scale,
            prefix_lengths[:,None,:].expand(-1,16,-1).float()/scale.square(),
            ((k+1)/lengths[:,2].clamp(min=1).float())[:,None,None].expand(-1,16,1),
        ],-1)
        assert channels.shape[-1]==CHANNELS
        real=active[:,None]&(j<n[:,None])
        result[:,:,k]=channels*real[:,:,None]
        chosen=torch.cat([selected[:,k],active[:,None]],1)&active[:,None]
        prefix.scatter_(2,i[:,None,None].expand(-1,3,1),torch.where(chosen,right,left)[:,:,None])
        prefix.scatter_(2,(i+1)[:,None,None].expand(-1,3,1),torch.where(chosen,left,right)[:,:,None])
        prefix_lengths+=chosen
    flat=result.flatten(2)
    if return_combinatorics:return flat,dict(permutations=p,n=n,word=word,selected=selected,lengths=lengths,terminal=terminal,prefix=prefix)
    return flat
