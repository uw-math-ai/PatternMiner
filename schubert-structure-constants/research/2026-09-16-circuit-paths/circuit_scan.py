"""Head and MLP activation replacement on exact matched pairs.

Discovery nominates components. Confirmation requires a separately frozen
selection. A diagnostic scan is explicitly development evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import circuit_context as s
from circuit_context import c
r=c.prior
import richardson_model as rm
import circuit_forward as cf
import compact
from prepare import collisions, encoded
from circuit_resources import Budget


def pair_data(part, cap, budget):
    if part=='validation':
        data=dict(np.load(r.ART/'hard_validation.npz'))
        metadata=json.loads((r.HERE/'VALIDATION_DATA.json').read_text())['hard_pairs']
    else:
        report=json.loads((r.HERE/'DATA.json').read_text())['populations']
        rows=[]; reverse=[]; metadata=[]; offset=0
        for n in [9,10]:
            name=f'S{n}_{part}'
            data=compact.load(r.HERE/f'cache/data/{name}.bin')
            spec=report[name]
            pairs=collisions(data,spec['offsets'],budget)
            for left,right,rev in pairs:
                rr=np.array(data[[left,right]],copy=True)
                rr['group']+=offset
                rows.extend(rr);reverse.extend([bool(rev)]*2)
                metadata.append(dict(population=name,rows=[int(left),int(right)],reverse=bool(rev),group=int(rr['group'][0])))
            offset+=spec['completed_groups']
        data=encoded(np.asarray(rows,dtype=compact.DTYPE),np.asarray(reverse))
    # A fixed round-robin sample prevents one factor pair dominating discovery.
    rng=np.random.default_rng(r.SEED+191)
    groups={}
    for index,entry in enumerate(metadata):groups.setdefault(entry['group'],[]).append(index)
    for indices in groups.values():rng.shuffle(indices)
    selected=[]
    while len(selected)<min(cap,len(metadata)):
        for group in sorted(groups):
            if groups[group] and len(selected)<cap:selected.append(groups[group].pop())
    take=np.asarray([j for i in selected for j in [2*i,2*i+1]])
    return {k:v[take] for k,v in data.items()},[metadata[i] for i in selected]


def contrast(scores,source_y,dest_y):
    return scores.masked_fill(~source_y,-1e9).logsumexp(1)-scores.masked_fill(~dest_y,-1e9).logsumexp(1)


def scan(model,data,budget,sites):
    output={key:[] for key in ['source_contrast','dest_contrast','source_good','dest_good','candidate_features_equal']}
    effects={name:[] for name in sites}
    source_choices={name:[] for name in sites}
    checks=dict(native_max_error=0.,identity_max_error=0.,restoration_max_error=0.)
    model.eval()
    for begin in range(0,len(data['x']),128):
        budget.check()
        x=torch.as_tensor(data['x'][begin:begin+128],device='cuda')
        mask=torch.as_tensor(data['mask'][begin:begin+128],device='cuda')
        yy=torch.as_tensor(data['y'][begin:begin+128],device='cuda')
        partner=torch.arange(len(x),device='cuda')^1
        assert torch.equal(mask,mask[partner])
        assert not (yy&yy[partner]).any()
        with torch.inference_mode():
            dest,dc=cf.explicit(model,x,capture=True)
            native=model(x)
            checks['native_max_error']=max(checks['native_max_error'],float((native-dest).abs().max()))
            source=dest[partner]
            sc={name:value[partner] for name,value in dc.items()}
            same_features=((dc['features']==sc['features'])|~mask[:,:,None]).all(-1).all(-1)
            output['candidate_features_equal'].append(same_features.cpu().numpy())
            base=contrast(dest,yy[partner],yy)
            goal=contrast(source,yy[partner],yy)
            dp=dest.masked_fill(~mask,-1e9).argmax(1)
            sp=source.masked_fill(~mask,-1e9).argmax(1)
            output['source_contrast'].append(goal.cpu().numpy())
            output['dest_contrast'].append(base.cpu().numpy())
            output['source_good'].append(yy[partner,sp].cpu().numpy())
            output['dest_good'].append(yy[torch.arange(len(x),device='cuda'),dp].cpu().numpy())
            ident=cf.explicit(model,x,patches={'L0.head_outputs':dc['L0.head_outputs']})
            checks['identity_max_error']=max(checks['identity_max_error'],float((ident-dest).abs().max()))
            restore=cf.explicit(model,x,patches={'embedding':sc['embedding'],'order_comparisons':sc['order_comparisons']})
            checks['restoration_max_error']=max(checks['restoration_max_error'],float((restore-source).abs().max()))
            for name,(site,head) in sites.items():
                replacement=sc[site] if head is None else dc[site].clone()
                if head is not None:replacement[:,head]=sc[site][:,head]
                actual=cf.explicit(model,x,patches={site:replacement})
                effects[name].append((contrast(actual,yy[partner],yy)-base).cpu().numpy())
                pred=actual.masked_fill(~mask,-1e9).argmax(1)
                source_choices[name].append(yy[partner,pred].cpu().numpy())
    assert checks['native_max_error']<2e-5
    assert checks['identity_max_error']==0.
    assert checks['restoration_max_error']==0.
    arrays={key:np.concatenate(values) for key,values in output.items()}
    arrays.update({f'effect_{name}':np.concatenate(values) for name,values in effects.items()})
    arrays.update({f'source_choice_{name}':np.concatenate(values) for name,values in source_choices.items()})
    return arrays,checks


def summaries(arrays,data,sites):
    valid=arrays['source_good']&arrays['dest_good']
    groups=np.unique(data['group'])
    report={}
    for name in sites:
        report[name]={}
        for label,include in [('all',np.ones(len(valid),bool)),('both_correct',valid),
                              ('both_correct_same_candidate_features',valid&arrays['candidate_features_equal'])]:
            effect=arrays[f'effect_{name}'][include]
            denominator=(arrays['source_contrast']-arrays['dest_contrast'])[include].mean() if include.any() else float('nan')
            group_means=[arrays[f'effect_{name}'][include&(data['group']==group)].mean()
                         for group in groups if (include&(data['group']==group)).any()]
            report[name][label]=dict(inputs=int(include.sum()),raw_mean_change=float(effect.mean()) if len(effect) else None,
                    aggregate_restoration_fraction=float(effect.mean()/denominator) if abs(denominator)>1e-8 else None,
                    source_choice_fraction=float(arrays[f'source_choice_{name}'][include].mean()) if include.any() else None,
                    factor_group_mean_change=float(np.mean(group_means)) if group_means else None)
    return report



def gradient_scores(model,data,budget):
    result={}
    for begin in range(0,len(data['x']),32):
        budget.check();x=torch.as_tensor(data['x'][begin:begin+32],device='cuda')
        y=torch.as_tensor(data['y'][begin:begin+32],device='cuda');partner=torch.arange(len(x),device='cuda')^1
        scores,cache=cf.explicit(model,x,capture=True)
        names=[name for name in cache if name.endswith('.head_outputs') or name.endswith('.mlp_output')]
        values=[cache[name] for name in names]
        gradients=torch.autograd.grad(contrast(scores,y[partner],y).sum(),values)
        for name,value,gradient in zip(names,values,gradients):
            contribution=(value[partner]-value).detach()*gradient.detach()
            if name.endswith('.head_outputs'):
                contribution=contribution.sum((2,3))
                for head in range(contribution.shape[1]):result.setdefault(name.split('.')[0]+'H'+str(head),[]).append(contribution[:,head].cpu().numpy())
            else:result.setdefault(name.split('.')[0]+'MLP',[]).append(contribution.sum((1,2)).cpu().numpy())
    return {name:np.concatenate(values) for name,values in result.items()}


def main(args,budget):
    path=Path(args.checkpoint)
    output=s.HERE/f'{args.name}.json'
    if output.exists():raise RuntimeError('Use a new scan name')
    if args.part=='confirmation':raise RuntimeError('Confirmation requires frozen component selection and controls')
    assessment=json.loads(Path(args.assessment).read_text())
    s.verify_checkpoint(path, args.assessment)
    assert assessment['qualification']['checkpoint_sha256']==s.sha(path)
    assert args.part=='discovery'
    marker=s.HERE/'FORMAL_DISCOVERY_OPENED.json'
    with marker.open('x') as stream:json.dump(dict(checkpoint_sha256=s.sha(path),assessment_sha256=s.sha(Path(args.assessment)),source_sha256=s.sha(Path(__file__))),stream,indent=2)
    model=rm.load(path).eval()
    data,metadata=pair_data(args.part,args.pairs,budget)
    assert len(data['x'])>0
    s.write(args.name+'_MEMBERSHIP.json',dict(pairs=metadata,checkpoint_sha256=s.sha(path)))
    gradient=gradient_scores(model,data,budget)
    sites={f'L{layer}H{head}':(f'L{layer}.head_outputs',head)
           for layer,block in enumerate(model.encoder.layers) for head in range(block.self_attn.num_heads)}
    sites.update({f'L{layer}MLP':(f'L{layer}.mlp_output',None) for layer in range(len(model.encoder.layers))})
    arrays,checks=scan(model,data,budget,sites)
    np.savez_compressed(s.ART/f'{args.name}.npz',**arrays,**{'gradient_'+key:value for key,value in gradient.items()},**data)
    report=dict(configuration=vars(args),checkpoint_sha256=r.c.a.sha(path),
                source_sha256=r.c.a.sha(Path(__file__)),pairs=metadata,checks=checks,
                unmodified_both_pair_accuracy=float((arrays['source_good']&arrays['dest_good']).mean()),
                sites=summaries(arrays,data,sites),
                scope='Circuit discovery under the explicit September 16 protocol amendment; checkpoint accuracy is reported without an eligibility gate',
                gradient_nomination='Gradient times the paired activation difference is retained for every head and MLP. Direct replacements are measured for all components before selecting heads by factor-group mean effect.')
    s.write(output.name,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['pairs','sites']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--assessment',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--part',choices=['validation','discovery','confirmation'],required=True)
    parser.add_argument('--pairs',type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(1)
    budget=Budget(args.name.lower(),420,gpu=True)
    status='failed_or_partial'
    try:
        main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
