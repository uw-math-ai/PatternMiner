"""Adjoin common fixed points on either side of a Schubert triple."""
import torch


def embed(data, roots, prefix, extra):
    n = data['n'].long()
    assert ((prefix >= 0) & (prefix <= extra) & (n+extra <= 17)).all()
    position = torch.arange(17, device=n.device)
    old_position = position[None, :] - prefix[:, None]
    inside = (old_position >= 0) & (old_position < n[:, None])
    gathered = data['permutations'].gather(2,old_position.clamp(0,16)[:,None,:].expand(-1,3,-1))
    moved = gathered.long() + prefix[:,None,None]
    identity = (position+1)[None,None,:].expand_as(moved)
    permutation = torch.where(inside[:,None,:],moved,identity)
    permutation = permutation.masked_fill(position[None,None,:] >= (n+extra)[:,None,None],0)
    index = torch.arange(16,device=n.device)[None,:] - prefix[:,None]
    real = (index >= 0) & (index < n[:,None]-1)
    counts = data['coefficients'].gather(2,index.clamp(0,15)[:,None,:].expand(-1,2,-1))
    counts = counts.masked_fill(~real[:,None,:],0)
    q = None if roots is None else roots.gather(1,index.clamp(0,15)).masked_fill(~real,0)
    return dict(permutations=permutation.to(data['permutations'].dtype),
                n=(n+extra).to(data['n'].dtype),lengths=data['lengths'],coefficients=counts),q


def sample(data, roots, fraction, generator, ordinary_and_error_count):
    n = data['n'].long()
    room = 17-n
    chosen = (torch.rand(len(n),device=n.device,generator=generator) < fraction) & (room > 0)
    extra = 1+(torch.rand(len(n),device=n.device,generator=generator)*room).long()
    extra = extra.masked_fill(~chosen,0)
    prefix = (torch.rand(len(n),device=n.device,generator=generator)*(extra+1)).long()
    # The two outputs in a hard pair use the same embedding.
    start = ordinary_and_error_count
    if start < len(n):
        assert torch.equal(n[start::2],n[start+1::2])
        extra[start:] = extra[start::2].clone().repeat_interleave(2)
        prefix[start:] = prefix[start::2].clone().repeat_interleave(2)
    transformed,q = embed(data,roots,prefix,extra)
    return transformed,q,prefix,extra
