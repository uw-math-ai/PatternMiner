"""Add distinguished-subexpression features to the retained neural selector."""
from pathlib import Path
import torch
from torch import nn
import context as c
from models import AllPairOrderSelector
import richardson_features as rf


class SplitInput(nn.Module):
    def __init__(self,base,extra_features=rf.MAX_STEPS*rf.CHANNELS):
        super().__init__();self.base=base
        self.extra=nn.Linear(extra_features,base.out_features,bias=False,device=base.weight.device)
        nn.init.zeros_(self.extra.weight)
    def forward(self,features):
        if getattr(self,'rank_only',False):return self.base(features)
        return self.base(features[:,:,:124])+self.extra(features[:,:,124:])


class RichardsonSelector(AllPairOrderSelector):
    def feature_map(self,x):
        base=super().feature_map(x)
        if getattr(self,'rank_only',False):return base
        if getattr(self,'extra_kind','richardson')=='relative_roots':
            from relative_root_features import features
            return torch.cat([base,features(x)],-1)
        return torch.cat([base,rf.features(x)],-1)


def use_rank_only(model):
    assert not model.input.extra.weight.count_nonzero()
    model.input.extra.weight.requires_grad_(False)
    model.rank_only=True;model.input.rank_only=True
    return model


def use_relative_roots(model):
    from relative_root_features import CHANNELS
    assert not model.input.extra.weight.count_nonzero()
    model.input.extra=nn.Linear(CHANNELS,model.input.base.out_features,bias=False,device=model.input.base.weight.device)
    nn.init.zeros_(model.input.extra.weight)
    model.rank_only=False;model.input.rank_only=False;model.extra_kind='relative_roots'
    return model


def extend(base,kind='richardson'):
    assert not base.full_ranks
    base.__class__=RichardsonSelector
    from relative_root_features import CHANNELS
    base.input=SplitInput(base.input,CHANNELS if kind=='relative_roots' else rf.MAX_STEPS*rf.CHANNELS)
    base.extra_kind=kind
    return base


def load(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if 'richardson' not in saved:return extend(c.prior.model_from(path))
    state=saved['state_dict'];width=state['input.base.weight'].shape[0]
    root_action=None
    if saved.get('root_action_enabled',False):
        state=dict(state);root_action=state.pop('root_action_order_bias');state['order_bias']=state.pop('base_order_bias')
    depth=max(int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.'))+1
    base=AllPairOrderSelector(width//96,False,depth-2).cuda()
    if 'auxiliary_head.weight' in state:base.auxiliary_head=nn.Linear(width,3).cuda()
    model=extend(base,saved['richardson'].get('kind','richardson'));model.load_state_dict(state)
    if saved.get('rank_only',False):use_rank_only(model)
    if saved.get('supplied_index_independent',False):
        from supplied_independent_model import use
        use(model)
    if root_action is not None:
        from root_action_model import use
        use(model)
        with torch.no_grad():model.root_action_order_bias.copy_(root_action.to(model.root_action_order_bias.device))
    return model


def save(model,path,**metadata):
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    richardson=dict(kind=getattr(model,'extra_kind','richardson'),max_steps=rf.MAX_STEPS,channels=rf.CHANNELS,
                                    extra_features=model.input.extra.in_features),
                    rank_only=getattr(model,'rank_only',False),
                    supplied_index_independent=getattr(model,'supplied_index_independent',False),
                    root_action_enabled=getattr(model,'root_action_enabled',False),**metadata),path)
