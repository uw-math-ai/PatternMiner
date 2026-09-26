"""The Dynkin diagram involution on inputs and exact support labels."""
from functools import lru_cache
import numpy as np
import torch


def conjugate(p):
    n = len(p)
    return tuple(n+1-v for v in reversed(p))


@lru_cache(maxsize=30000)
def product_orbit(group):
    """Conservatively group reflection and common outer fixed-point shifts."""
    u, v = group
    n = max(len(u), len(v))
    u = tuple(u)+tuple(range(len(u)+1, n+1))
    v = tuple(v)+tuple(range(len(v)+1, n+1))
    while n and u[-1] == v[-1] == n:
        u, v, n = u[:-1], v[:-1], n-1
    while n and u[0] == v[0] == 1:
        u, v, n = tuple(x-1 for x in u[1:]), tuple(x-1 for x in v[1:]), n-1
    return min(tuple(sorted((u, v))), tuple(sorted((conjugate(u), conjugate(v)))))


def reflect(x, mask=None, y=None):
    """Return the exactly relabeled input under w -> w0 w w0.

    All inverse positions are recovered as integers before reflection. Labels
    and candidate masks are reversed only within the common ambient size.
    """
    width = x.shape[1]
    n = (1/x[:, 0, 18]).round().long()
    j = torch.arange(width, device=x.device)[None, :]
    active = j < n[:, None]-1
    rev = (n[:, None]-2-j).clamp(min=0)
    z = x.gather(1, rev[:, :, None].expand(-1, -1, x.shape[2])).clone()
    inverse = (z[:, :, 3:9]*n[:, None, None]).round().long().reshape(len(x), width, 3, 2)
    inverse = n[:, None, None, None]+1-inverse.flip(-1)
    z[:, :, 3:9] = inverse.reshape(len(x), width, 6).float()/n[:, None, None]
    z[:, :, 0] = (j+1)/n[:, None]
    old_i = (x[:, 0, 1]*n).round().long()-1
    new_i = n-2-old_i
    z = z*active[:, :, None]
    # Import lazily to avoid a module cycle when used inside campaign training.
    import accuracy as a
    z = a.change_witness(z, new_i)
    if mask is None:
        return z
    return z, mask.gather(1, rev)&active, y.gather(1, rev)&active


def aligned_scores(scores, x):
    n = (1/x[:, 0, 18]).round().long()
    j = torch.arange(x.shape[1], device=x.device)[None, :]
    rev = (n[:, None]-2-j).clamp(min=0)
    return scores.gather(1, rev)
