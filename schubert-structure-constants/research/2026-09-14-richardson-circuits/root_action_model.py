"""The action on the simple-root basis as additional attention comparisons."""
import torch
from torch import nn
import richardson_model as rm
from richardson_features import permutations
from relative_root_features import PAIRS
from supplied_independent_model import SuppliedIndependentSelector


def matrices(x):
    p,n = permutations(x)
    inverse = p.masked_fill(p == 0,99).argsort(-1)
    key = torch.arange(1,17,device=x.device)[None,None,:]
    result = []
    for a,b in PAIRS:
        composition = p[:,a].gather(1,inverse[:,b])
        left,right = composition[:,:16,None],composition[:,1:17,None]
        result.append(((left <= key)&(key < right)).float()-((right <= key)&(key < left)).float())
    real = torch.arange(1,17,device=x.device)[None,:] < n[:,None]
    return torch.stack(result,-1)*real[:,:,None,None]*real[:,None,:,None]


class RootActionMixin:
    @property
    def order_bias(self):
        return torch.cat([self.base_order_bias,self.root_action_order_bias],-1)

    def comparisons(self,x):
        return torch.cat([super().comparisons(x),matrices(x)],-1)


class RootActionSelector(RootActionMixin,rm.RichardsonSelector):
    pass


class RootActionIndependentSelector(RootActionMixin,SuppliedIndependentSelector):
    pass


def use(model):
    assert not getattr(model,'root_action_enabled',False)
    assert model.order_bias.shape[-1] == 72
    model.base_order_bias = model.order_bias
    model.root_action_order_bias = nn.Parameter(model.order_bias.new_zeros((*model.order_bias.shape[:2],6)))
    del model._parameters['order_bias']
    model.__class__ = RootActionIndependentSelector if getattr(model,'supplied_index_independent',False) else RootActionSelector
    model.root_action_enabled = True
    return model
