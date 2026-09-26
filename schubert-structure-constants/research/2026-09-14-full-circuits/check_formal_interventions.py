"""Check formal-study interfaces on development inputs, without opening its data."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from expanded_decomposition import bank
import richardson_model as rm
import circuit_forward as cf
from formal_confirm import patches_for
from formal_scan import gradient_scores,contrast
from full_resources import Budget


def main(args,budget):
    checkpoint=Path(args.checkpoint);model=rm.load(checkpoint).eval();data,_,_=bank()
    rng=np.random.default_rng(s.SEED+881);pairs=rng.choice(len(data['x'])//2,16,replace=False)
    take=np.stack([2*pairs,2*pairs+1],1).reshape(-1);data={key:value[take] for key,value in data.items()}
    x=torch.as_tensor(data['x'],device='cuda');y=torch.as_tensor(data['y'],device='cuda');partner=torch.arange(len(x),device='cuda')^1
    checks=dict(native=0.,identity=0.,restoration=0.,query_key_bias_pattern=0.)
    with torch.inference_mode():
        score,cache=cf.explicit(model,x,capture=True);source={k:v[partner] for k,v in cache.items()}
        checks['native']=float((score-model(x)).abs().max())
        for layer in range(len(model.encoder.layers)):
            for suffix in ['head_outputs','q','k','v','pattern','order_bias','mlp_hidden','mlp_output']:
                budget.check();site=f'L{layer}.{suffix}'
                checks['identity']=max(checks['identity'],float((cf.explicit(model,x,patches={site:cache[site]})-score).abs().max()))
        restored=cf.explicit(model,x,patches={k:source[k] for k in ['embedding','order_comparisons']})
        checks['restoration']=float((restored-score[partner]).abs().max())
        heads=['L1H0','L1H8','L2H0','L3H4']
        pattern=cf.explicit(model,x,patches=patches_for(cache,source,heads,'pattern'))
        qkb=cf.explicit(model,x,patches=patches_for(cache,source,heads,'query_key_bias'))
        checks['query_key_bias_pattern']=float((pattern-qkb).abs().max())
    parameters=[(name,value) for name,value in model.named_parameters() if value.requires_grad]
    native=model(x);ng=torch.autograd.grad(contrast(native,y[partner],y).sum(),[v for _,v in parameters],allow_unused=True)
    explicit=cf.explicit(model,x);eg=torch.autograd.grad(contrast(explicit,y[partner],y).sum(),[v for _,v in parameters],allow_unused=True)
    gradient={}
    for (name,_),a,b in zip(parameters,ng,eg):
        assert (a is None)==(b is None)
        if a is not None:gradient[name]=dict(maximum_absolute_error=float((a-b).abs().max()),relative_l2_error=float((a-b).norm()/a.norm().clamp(min=1e-8)))
    nomination=gradient_scores(model,data,budget)
    assert len(nomination)==len(model.encoder.layers)*(model.encoder.layers[0].self_attn.num_heads+1)
    assert checks['native']==checks['identity']==checks['restoration']==0 and checks['query_key_bias_pattern']<2e-5
    assert max(v['maximum_absolute_error'] for v in gradient.values())<1e-3
    assert max(v['relative_l2_error'] for v in gradient.values())<1e-3
    report=dict(status='passed',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),inputs=len(x),checks=checks,
        parameter_gradient_errors=gradient,nomination_components=len(nomination),
        maximum_gradient_absolute_error=max(v['maximum_absolute_error'] for v in gradient.values()),
        maximum_gradient_relative_l2_error=max(v['relative_l2_error'] for v in gradient.values()),
        role='Implementation verification on development inputs. No discovery or confirmation data are read.')
    s.write(args.name+'.json',report);print(json.dumps({k:v for k,v in report.items() if k!='parameter_gradient_errors'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True);args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget(args.name.lower(),120,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
