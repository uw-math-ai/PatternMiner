"""Compact exact examples and a checked on-device permutation encoder."""
import numpy as np
import torch
from functools import lru_cache
import common as r

WIDTH = 17
DTYPE = np.dtype([('permutations', 'u1', (3, WIDTH)), ('n', 'u1'),
                  ('lengths', 'u1', (3,)), ('coefficients', '<u2', (2, WIDTH-1)),
                  ('group', '<u4')])


@lru_cache(maxsize=200000)
def length(p):
    return r.c.a.original.length(p)


def load(path):
    return np.memmap(path, dtype=DTYPE, mode='r')


def from_rows(rows, group=0):
    out = np.zeros(len(rows), dtype=DTYPE)
    for k, row in enumerate(rows):
        n = len(row['w'])
        out['n'][k] = n
        out['group'][k] = row.get('group', group)
        for j, key in enumerate(['u', 'v', 'w']):
            out['permutations'][k, j, :n] = row[key]
            out['lengths'][k, j] = length(tuple(row[key]))
        for j, key in enumerate(['A', 'B']):
            values = np.asarray(row[key][:n-1])
            assert values.min() >= 0 and values.max() <= np.iinfo(np.uint16).max
            out['coefficients'][k, j, :n-1] = values
    return out


def to_row(row, reverse=False):
    n = int(row['n'])
    pp = row['permutations'][:,:n]
    aa = row['coefficients']
    a, b = (1, 0) if reverse else (0, 1)
    return dict(u=pp[a].astype(int).tolist(), v=pp[b].astype(int).tolist(), w=pp[2].astype(int).tolist(),
                A=aa[a,:n-1].astype(int).tolist()+[0], B=aa[b,:n-1].astype(int).tolist()+[0])


def tensors(records, device='cuda'):
    # Copy structured-array fields to contiguous storage before device transfer.
    return {key: torch.as_tensor(np.array(records[key], copy=True), device=device)
            for key in ['permutations', 'n', 'lengths', 'coefficients']}


def encode(data, reverse=None, supplied=None):
    """Return the frozen 22 features, candidate mask, and both support arrays.

    The first valid supplied index reproduces the historical float64-to-float32
    encoder. Optional supplied indices use its recorded change_witness routine.
    Labels select the supplied certificate and targets; the feature map only
    receives permutations, lengths, and that single index.
    """
    p = data['permutations'].long()
    n = data['n'].long()
    coefficients = data['coefficients'].long()
    lengths = data['lengths'].long()
    if reverse is not None:
        p = p.clone()
        coefficients = coefficients.clone()
        lengths = lengths.clone()
        p[reverse] = p[reverse][:,[1,0,2]]
        coefficients[reverse] = coefficients[reverse][:,[1,0]]
        lengths[reverse] = lengths[reverse][:,[1,0,2]]
    cuts = torch.arange(WIDTH, device=p.device)
    inverse = torch.where(p > 0, p, WIDTH+1+cuts).argsort(-1)+1
    inverse = inverse*(cuts[None,None,:] < n[:,None,None])
    aa, y = (coefficients[:, k] > 0 for k in [0,1])
    i = aa.long().argmax(1)
    j = cuts[:-1][None,:]
    positions = inverse.double()
    scale = n.double()
    x = torch.zeros((len(p), WIDTH-1, 22), dtype=torch.float64, device=p.device)
    x[:,:,0] = (j+1)/scale[:,None]
    x[:,:,1] = (i[:,None]+1)/scale[:,None]
    x[:,:,2] = (j-i[:,None])/scale[:,None]
    x[:,:,3:9] = torch.stack([positions[:,:,:-1],positions[:,:,1:]],dim=-1).permute(0,2,1,3).reshape(len(p),WIDTH-1,6)/scale[:,None,None]
    selected = positions.gather(2, (i[:,None,None]+torch.arange(2,device=p.device)).expand(-1,3,-1))
    x[:,:,9:15] = selected.reshape(len(p),1,6)/scale[:,None,None]
    x[:,:,15] = inverse[:,0,:-1] > inverse[:,0,1:]
    x[:,:,16] = inverse[:,2,:-1] > inverse[:,2,1:]
    x[:,:,17] = j == i[:,None]
    x[:,:,18] = 1/scale[:,None]
    x[:,:,19:22] = lengths.double()[:,None,:]/scale[:,None,None].square()
    real = j < n[:,None]-1
    x = (x*real[:,:,None]).float()
    mask = (inverse[:,1,:-1] > inverse[:,1,1:])&real
    if supplied is not None:
        x = r.c.a.change_witness(x, supplied)
    return x, mask, y, aa
