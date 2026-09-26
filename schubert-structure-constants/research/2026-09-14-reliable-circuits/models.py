"""Local architecture extensions; frozen model implementations are unchanged."""
import torch
from torch import nn
import common as r
from order_attention import OrderSelector


class AllPairOrderSelector(OrderSelector):
    """Compare all query/key inverse positions across the three permutations."""
    comparison_mode='all_pairs'
    candidate_relation_channels=tuple(range(6))+tuple(range(12,42))
    supplied_relation_channels=tuple(range(6,12))+tuple(range(42,72))

    def __init__(self,multiplier=2,full_ranks=False,extra_depth=2):
        super().__init__(multiplier,0.,full_ranks,extra_depth)
        self.order_bias=nn.Parameter(torch.zeros(2+extra_depth,4*multiplier,72))

    @staticmethod
    def comparisons(x):
        original=OrderSelector.comparisons(x)
        n=(1/x[:,0,18]).round().long()
        endpoint=(x[:,:,3:9]*n[:,None,None]).round()
        supplied=(x[:,:,9:15]*n[:,None,None]).round()
        left,right=zip(*[(a,b) for a in range(6) for b in range(6) if a!=b])
        left,right=list(left),list(right)
        candidate=(endpoint[:,:,None,left]-endpoint[:,None,:,right]).sign()
        certificate=(supplied[:,:,None,left]-endpoint[:,None,:,right]).sign()
        real=x[:,:,18]!=0
        valid=real[:,:,None,None]&real[:,None,:,None]
        return torch.cat([original,candidate*valid,certificate*valid],-1)


def widen_state(state,factor):
    """Duplicate residual coordinates and attention heads, preserving head width."""
    if factor==1:return state
    result={}
    for name,value in state.items():
        if name=='input.weight':expanded=value.repeat(factor,1)
        elif name=='input.bias':expanded=value.repeat(factor)
        elif name in ['head.weight','auxiliary_head.weight']:expanded=value.repeat(1,factor)/factor
        elif name in ['head.bias','auxiliary_head.bias']:expanded=value.clone()
        elif name=='order_bias':expanded=value.repeat(1,factor,1)
        elif name.endswith('in_proj_weight'):
            width=value.shape[1]
            expanded=value.reshape(3,width,width).repeat(1,factor,factor).reshape(3*width*factor,width*factor)/factor
        elif name.endswith('in_proj_bias'):
            expanded=value.reshape(3,-1).repeat(1,factor).reshape(-1)
        elif name.startswith('encoder.layers.') and value.ndim==2:
            expanded=value.repeat(factor,factor)/factor
        elif name.startswith('encoder.layers.') and value.ndim==1:
            expanded=value.repeat(factor)
        else:raise ValueError(f'Unrecognized widening parameter: {name}')
        result[name]=expanded
    return result


def model_from(path,full_ranks=None,extra_depth=None,comparison_mode=None,multiplier=None,noise=0.):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['state_dict']
    current='all_pairs' if 'order_bias' in state and state['order_bias'].shape[-1]==72 else 'original'
    mode=current if comparison_mode is None else comparison_mode
    if mode=='original':
        if current!='original':raise ValueError('Dropping comparison channels requires an explicit ablation')
        if multiplier is not None or noise:raise ValueError('Use the local all_pairs model for explicit width/noise experiments')
        return r.c.model_from(path,full_ranks=full_ranks,extra_depth=extra_depth)
    if mode!='all_pairs':raise ValueError(mode)
    if 'order_bias' not in state:raise ValueError('This extension starts from a model with explicit order comparisons')
    saved_multiplier=state['input.weight'].shape[0]//96
    multiplier=saved_multiplier if multiplier is None else multiplier
    assert multiplier>=saved_multiplier and multiplier%saved_multiplier==0
    state=widen_state(state,multiplier//saved_multiplier)
    full_ranks=state['input.weight'].shape[1]==991 if full_ranks is None else full_ranks
    saved_depth=max(int(k.split('.')[2]) for k in state if k.startswith('encoder.layers.'))-1
    extra_depth=saved_depth if extra_depth is None else extra_depth
    assert extra_depth>=saved_depth
    model=AllPairOrderSelector(multiplier,full_ranks,extra_depth).cuda()
    if 'auxiliary_head.weight' in state:
        shape=state['auxiliary_head.weight'].shape
        assert tuple(shape)==(3,96*multiplier)
        model.auxiliary_head=nn.Linear(shape[1],shape[0]).cuda()
    if full_ranks and state['input.weight'].shape[1]==124:
        expanded=torch.zeros((96*multiplier,991),dtype=state['input.weight'].dtype)
        expanded[:,:124]=state['input.weight'];state['input.weight']=expanded
    bias=model.order_bias.detach().cpu().clone()
    bias[:len(state['order_bias']),:,:state['order_bias'].shape[-1]]=state['order_bias']
    state['order_bias']=bias
    for key,value in model.state_dict().items():
        if key.startswith('encoder.layers.') and int(key.split('.')[2])>=2+saved_depth:
            state[key]=value.detach().cpu()
    model.load_state_dict(state)
    if noise:
        assert noise>0
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim==2:parameter.add_(torch.randn_like(parameter)*noise)
    return model
