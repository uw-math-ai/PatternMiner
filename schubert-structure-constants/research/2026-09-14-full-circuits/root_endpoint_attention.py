"""Test attention to simple roots incident to the endpoints of w v^{-1}(alpha_j)."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from expanded_decomposition import bank
import richardson_model as rm
import richardson_features as rf
import circuit_forward as cf
from pair_circuit_diagnostic import contrast,summarize
from full_resources import Budget


def main(budget):
    comparison=json.loads((s.HERE/'ERROR_SAMPLING_COMMON_WINDOW_COMPARISON.json').read_text())
    checkpoint=s.ART/(comparison['winner']+'_common5000.pt');model=rm.load(checkpoint).eval()
    data,names,_=bank();heads=[0,8]
    plan=dict(status='fixed_before_interventions',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),
        inputs=len(data['x']),heads=heads,layer='Second attention layer',queries='Candidate simple roots, namely left descents of v',
        rules={'first_endpoint':'Simple roots incident to w(v^{-1}(j))',
               'second_endpoint':'Simple roots incident to w(v^{-1}(j+1))',
               'both_endpoints':'Simple roots with nonzero Cartan pairing against (w v^{-1})(alpha_j)',
               'root_support':'Simple roots with nonzero coefficient in (w v^{-1})(alpha_j)'},
        intervention='Replace both head patterns together at candidate queries, using either the uniform distribution on the prescribed set or the native pattern conditioned on that set.',
        control='Eight independent uniform random subsets with the same cardinality as the second-endpoint set, sampled before each intervention from a fixed generator.',
        role='Adaptive development test motivated by endpoint reception and the earlier unsuccessful single-key rule; no formal confirmation population opened')
    assert not (s.HERE/'ROOT_ENDPOINT_ATTENTION_PLAN.json').exists();s.write('ROOT_ENDPOINT_ATTENTION_PLAN.json',plan)
    effects={};correct={};mass={};hit={};size={};native_good=[];checks=dict(native=0.,identity=0.,cartan_support=True)
    generator=torch.Generator(device='cuda').manual_seed(s.SEED+861)
    with torch.inference_mode():
        for begin in range(0,len(data['x']),64):
            budget.check();end=min(begin+64,len(data['x']))
            x=torch.as_tensor(data['x'][begin:end],device='cuda');mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
            y=torch.as_tensor(data['y'][begin:end],device='cuda');rr=torch.arange(len(x),device='cuda');partner=rr^1;sy=y[partner]
            p,n=rf.permutations(x);inv=p[:,1].masked_fill(p[:,1]==0,99).argsort(-1)
            composition=p[:,2].gather(1,inv);a=composition[:,:16,None];b=composition[:,1:17,None]
            key=torch.arange(1,17,device='cuda')[None,None,:];real=key<n[:,None,None]
            first=((key==a-1)|(key==a))&real
            second=((key==b-1)|(key==b))&real
            cartan=(key==a).int()-(key+1==a).int()-(key==b).int()+(key+1==b).int()
            combined=first|second;assert torch.equal(combined,(cartan!=0)&real)
            support=((torch.minimum(a,b)<=key)&(key<torch.maximum(a,b)))&real
            rules=dict(first_endpoint=first,second_endpoint=second,both_endpoints=combined,root_support=support)
            scores,dc=cf.explicit(model,x,capture=True);pattern=dc['L1.pattern'];base=contrast(scores,sy,y)
            good=y[rr,scores.masked_fill(~mask,-1e9).argmax(1)];native_good.append(good.cpu().numpy())
            checks['native']=max(checks['native'],float((model(x)-scores).abs().max()))
            identity=cf.explicit(model,x,patches={'L1.pattern':pattern})
            checks['identity']=max(checks['identity'],float((identity-scores).abs().max()))
            for name,selection in rules.items():
                total=(pattern[:,heads]*selection[:,None]).sum(-1)
                maximum=pattern[:,heads].argmax(-1)
                hits=selection[:,None].expand(-1,len(heads),-1,-1).gather(-1,maximum[:,:,:,None]).squeeze(-1)
                mass.setdefault(name,[]).append(((total*mask[:,None]).sum((1,2))/(len(heads)*mask.sum(1))).cpu().numpy())
                hit.setdefault(name,[]).append(((hits*mask[:,None]).sum((1,2))/(len(heads)*mask.sum(1))).cpu().numpy())
                size.setdefault(name,[]).append((((selection.sum(-1)/(n[:,None]-1))*mask).sum(1)/mask.sum(1)).cpu().numpy())
            cardinality=second.sum(-1)
            for repeat in range(8):
                order=torch.rand(second.shape,device='cuda',generator=generator).masked_fill(~real.expand_as(second),2.).argsort(-1).argsort(-1)
                rules[f'random_second_{repeat}']=(order<cardinality[:,:,None])&real
            for name,selection in rules.items():
                for mode in ['uniform','conditioned']:
                    budget.check();replacement=pattern.clone()
                    if mode=='uniform':value=selection[:,None].float().expand(-1,len(heads),-1,-1)
                    else:value=pattern[:,heads]*selection[:,None]
                    value=value/value.sum(-1,keepdim=True).clamp(min=torch.finfo(value.dtype).tiny)
                    # Padding queries have empty sets, but their native pattern is retained.
                    replacement[:,heads]=torch.where(mask[:,None,:,None],value,pattern[:,heads])
                    result=cf.explicit(model,x,patches={'L1.pattern':replacement})
                    label=name+'_'+mode
                    effects.setdefault(label,[]).append((contrast(result,sy,y)-base).cpu().numpy())
                    correct.setdefault(label,[]).append(y[rr,result.masked_fill(~mask,-1e9).argmax(1)].cpu().numpy())
            if begin%1024==0:print(json.dumps(dict(inputs=end,total=len(data['x']))),flush=True)
    assert checks['native']==checks['identity']==0
    arrays={f'effect_{k}':np.concatenate(v) for k,v in effects.items()}
    arrays.update({f'correct_{k}':np.concatenate(v) for k,v in correct.items()})
    for label,values in [('mass',mass),('hit',hit),('uniform_mass',size)]:arrays.update({f'{label}_{k}':np.concatenate(v) for k,v in values.items()})
    arrays.update(native_correct=np.concatenate(native_good),group=data['group'],population=data['population'])
    rng=np.random.default_rng(s.SEED+862);summaries={};behavior={}
    for name in mass:
        summaries[name]={k:summarize(arrays[k+'_'+name],np.ones(len(data['x']),bool),data['group'],rng) for k in ['mass','hit','uniform_mass']}
    for name in correct:
        good=arrays['correct_'+name];paired=good[::2]&good[1::2]
        behavior[name]=dict(oriented_accuracy=float(good.mean()),pair_accuracy=float(paired.mean()),
            lost=int((arrays['native_correct']&~good).sum()),fixed=int((~arrays['native_correct']&good).sum()),
            by_population={pop:float(paired[data['population'][::2]==i].mean()) for i,pop in enumerate(names)})
    archive=s.ART/'ROOT_ENDPOINT_ATTENTION.npz';np.savez_compressed(archive,**arrays)
    report=dict(**{k:v for k,v in plan.items() if k!='status'},status='completed',checks=checks,
        native_pair_accuracy=float((arrays['native_correct'][::2]&arrays['native_correct'][1::2]).mean()),
        attention=summaries,behavior=behavior,artifact_sha256=s.sha(archive),
        limitation='A conditioned pattern retains learned weights within the proposed set. A uniform replacement is a stricter functional test. All results use development populations, with no claim of a recovered Schubert multiplication rule.')
    s.write('ROOT_ENDPOINT_ATTENTION.json',report);print(json.dumps(dict(status='completed',attention=summaries,behavior={k:v for k,v in behavior.items() if not k.startswith('random_')})),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('root_endpoint_attention',600,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
