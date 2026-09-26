"""Development interventions between exact pairs with disjoint target supports."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import context as c
import richardson_model as rm
import richardson_features as rf
import circuit_forward as cf
import training_data
from resources import Budget


def orbit_keys(x):
    p,n=rf.permutations(torch.as_tensor(x,device='cpu'))
    return [c.prior.symmetry.product_orbit((tuple(map(int,v[0,:size])),tuple(map(int,v[1,:size])))) for v,size in zip(p.numpy(),n.numpy())]


def prepare(cap):
    populations={k:v for k,v in training_data.validation(True).items() if k=='hard_validation' or k.endswith('_pairs')}
    bank={key:np.concatenate([d[key] for d in populations.values()]) for key in ['x','mask','y','a']}
    keys=orbit_keys(bank['x']);unique={key:i for i,key in enumerate(dict.fromkeys(keys))};bank['group']=np.asarray([unique[key] for key in keys])
    pairs={};metadata=[];offset=0
    for name,data in populations.items():
        for j in range(0,len(data['x']),2):
            index=len(metadata);metadata.append(dict(population=name,rows=[j,j+1],bank_rows=[offset+j,offset+j+1],group=int(bank['group'][offset+j])))
            pairs.setdefault(metadata[-1]['group'],[]).append(index)
        offset+=len(data['x'])
    rng=np.random.default_rng(c.SEED+811)
    for values in pairs.values():rng.shuffle(values)
    selected=[]
    while len(selected)<min(cap,len(metadata)):
        for group in sorted(pairs):
            if pairs[group] and len(selected)<cap:selected.append(pairs[group].pop())
    meta=[metadata[i] for i in selected];take=np.asarray([j for item in meta for j in item['bank_rows']])
    data={k:v[take] for k,v in bank.items()}
    assert np.array_equal(data['mask'][::2],data['mask'][1::2]) and not (data['y'][::2]&data['y'][1::2]).any()
    assert np.array_equal(data['x'][::2][data['mask'][::2]],data['x'][1::2][data['mask'][1::2]])
    return data,meta,bank


def contrast(scores,source,destination):
    return scores.masked_fill(~source,-1e9).logsumexp(1)-scores.masked_fill(~destination,-1e9).logsumexp(1)


def summarize(effect,include,groups,rng):
    means=np.asarray([effect[include&(groups==g)].mean() for g in np.unique(groups[include])])
    if not len(means):return dict(inputs=0,groups=0)
    sampled=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
    return dict(inputs=int(include.sum()),groups=len(means),input_mean=float(effect[include].mean()),
                factor_group_mean=float(means.mean()),group_bootstrap_95_percent=np.quantile(sampled,[.025,.975]).tolist())


def main(args,budget):
    path=Path(args.checkpoint);model=rm.load(path).eval();assert getattr(model,'rank_only',False)
    data,metadata,bank=prepare(args.pairs);sites={}
    for label,columns in [('original',slice(0,22)),('rank_columns',slice(22,124)),('w_rank_columns',slice(90,124))]:
        sites['features_'+label]=('features',None,columns)
    sites['order_comparisons']=('order_comparisons',None,None)
    for layer,block in enumerate(model.encoder.layers):
        for suffix in ['pattern','q','k','v','head_outputs','mlp_output','resid_post']:
            sites[f'L{layer}.{suffix}']=(f'L{layer}.{suffix}',None,None)
        for head in range(block.self_attn.num_heads):sites[f'L{layer}.H{head}']=(f'L{layer}.head_outputs',head,None)
    pieces={key:[] for key in ['baseline_contrast','source_contrast','both_correct','same_full_candidate_features']}
    effects={key:[] for key in sites};choices={key:[] for key in sites};checks=dict(native=0.,identity=0.,complete_input_restoration=0.)
    with torch.inference_mode():
        for begin in range(0,len(data['x']),64):
            budget.check();xx=torch.as_tensor(data['x'][begin:begin+64],device='cuda')
            mask=torch.as_tensor(data['mask'][begin:begin+64],device='cuda');y=torch.as_tensor(data['y'][begin:begin+64],device='cuda')
            partner=torch.arange(len(xx),device='cuda')^1;rr=torch.arange(len(xx),device='cuda')
            scores,cache=cf.explicit(model,xx,capture=True);source={k:v[partner] for k,v in cache.items()}
            checks['native']=max(checks['native'],float((scores-model(xx)).abs().max()))
            identity=cf.explicit(model,xx,patches={'L0.head_outputs':cache['L0.head_outputs']})
            restore=cf.explicit(model,xx,patches={'embedding':source['embedding'],'order_comparisons':source['order_comparisons']})
            checks['identity']=max(checks['identity'],float((scores-identity).abs().max()))
            checks['complete_input_restoration']=max(checks['complete_input_restoration'],float((scores[partner]-restore).abs().max()))
            pred=scores.masked_fill(~mask,-1e9).argmax(1);good=y[rr,pred]
            base=contrast(scores,y[partner],y);goal=contrast(scores[partner],y[partner],y)
            for key,value in dict(baseline_contrast=base,source_contrast=goal,both_correct=good&good[partner],
                    same_full_candidate_features=((cache['features']==source['features'])|~mask[:,:,None]).all(-1).all(-1)).items():pieces[key].append(value.cpu().numpy())
            for name,(site,head,columns) in sites.items():
                if head is not None:
                    replacement=cache[site].clone();replacement[:,head]=source[site][:,head]
                elif columns is not None:
                    replacement=cache[site].clone();replacement[:,:,columns]=source[site][:,:,columns]
                else:replacement=source[site]
                actual=cf.explicit(model,xx,patches={site:replacement})
                effects[name].append((contrast(actual,y[partner],y)-base).cpu().numpy())
                pred=actual.masked_fill(~mask,-1e9).argmax(1);choices[name].append(y[partner,pred].cpu().numpy())
    assert max(checks.values())==0.,checks
    arrays={k:np.concatenate(v) for k,v in pieces.items()};arrays.update({k:np.asarray(v) for k,v in data.items()})
    arrays.update({'effect_'+k:np.concatenate(v) for k,v in effects.items()});arrays.update({'source_choice_'+k:np.concatenate(v) for k,v in choices.items()})
    rng=np.random.default_rng(c.SEED+812);summaries={}
    for name in sites:
        summaries[name]={label:summarize(arrays['effect_'+name],included,data['group'],rng) for label,included in [
            ('all',np.ones(len(data['x']),bool)),('both_correct',arrays['both_correct']),
            ('both_correct_equal_full_candidate_features',arrays['both_correct']&arrays['same_full_candidate_features'])]}
    report=dict(role='Development causal diagnostic; accuracy qualification remains unmet',checkpoint_sha256=c.sha(path),
        source_sha256=c.sha(Path(__file__)),forward_sha256=c.sha(Path(cf.__file__)),checks=checks,pairs=metadata,
        selection='Fixed seeded round robin over canonical factor orbits, before inspecting this checkpoint; recorded common supplied indices',
        baseline_pair_accuracy=float(arrays['both_correct'].mean()),sites=summaries,
        uncertainty='Percentile bootstrap over canonical factor orbits; exploratory intervals do not adjust for component selection')
    # Three preselected component types, with exact target-support matching and distinct factor orbits.
    control_sites=['L0.mlp_output','L1.pattern','L1.v'];eligible=[];donor_pools=[]
    n=np.rint(1/data['x'][:,0,18]).astype(int);ii=np.rint(data['x'][:,0,1]*n).astype(int)-1
    bn=np.rint(1/bank['x'][:,0,18]).astype(int);counts={}
    for row in np.flatnonzero(arrays['both_correct']):
        group=int(data['group'][row])
        if counts.get(group,0)>=4:continue
        source_y=data['y'][row^1]
        possible=np.flatnonzero((bn==n[row])&bank['a'][:,ii[row]]&np.all(bank['mask']==data['mask'][row],1)&np.all(bank['y']==source_y,1)&(bank['group']!=group))
        if not len(possible):continue
        possible=rng.choice(possible,min(len(possible),32),replace=False)
        dx=c.prior.c.a.change_witness(torch.as_tensor(bank['x'][possible],device='cuda'),torch.full((len(possible),),int(ii[row]),device='cuda'))
        with torch.inference_mode():pred=model(dx).masked_fill(~torch.as_tensor(bank['mask'][possible],device='cuda'),-1e9).argmax(1).cpu().numpy()
        possible=possible[bank['y'][possible,pred]]
        if not len(possible):continue
        eligible.append(int(row));donor_pools.append(possible);counts[group]=counts.get(group,0)+1
        if len(eligible)==64:break
    controls=dict(eligible_inputs=len(eligible),factor_groups=len(counts),repetitions=10,
        matching='Same n, supplied i, candidate mask, and source target support; positive supplied coefficient and correct donor prediction; distinct canonical factor orbit',sites={})
    arrays['control_eligible']=np.asarray(eligible)
    if eligible:
        xx=torch.as_tensor(data['x'][eligible],device='cuda');y=torch.as_tensor(data['y'][eligible],device='cuda')
        sy=torch.as_tensor(data['y'][np.asarray(eligible)^1],device='cuda');mask=torch.as_tensor(data['mask'][eligible],device='cuda')
        base=torch.as_tensor(arrays['baseline_contrast'][eligible],device='cuda');donors=[];changes={site:[] for site in control_sites}
        with torch.inference_mode():
            for repeat in range(10):
                budget.check();indices=np.asarray([rng.choice(pool) for pool in donor_pools]);donors.append(indices)
                dx=c.prior.c.a.change_witness(torch.as_tensor(bank['x'][indices],device='cuda'),torch.as_tensor(ii[eligible],device='cuda'))
                _,dc=cf.explicit(model,dx,capture=True)
                for site in control_sites:
                    actual=cf.explicit(model,xx,patches={site:dc[site]});changes[site].append((contrast(actual,sy,y)-base).cpu().numpy())
        arrays['donor_indices']=np.stack(donors)
        for site in control_sites:
            altered=np.stack(changes[site]);arrays['donor_effect_'+site]=altered
            difference=arrays['effect_'+site][eligible]-altered.mean(0)
            controls['sites'][site]=dict(paired_mean=float(arrays['effect_'+site][eligible].mean()),donor_mean=float(altered.mean()),
                paired_minus_donor=summarize(difference,np.ones(len(eligible),bool),data['group'][eligible],rng))
    report['matched_donor_controls']=controls
    archive=c.ART/(args.name+'.npz');np.savez_compressed(archive,**arrays);report['artifact_sha256']=c.sha(archive)
    report['limitation']='Whole-state replacement is a control. A positive component effect does not identify a Schubert multiplication rule, and these development-selected findings are not independent confirmation.'
    c.write(args.name+'.json',report)
    print(json.dumps(dict(checks=checks,pairs=len(metadata),baseline_pair_accuracy=report['baseline_pair_accuracy'],controls=controls,
        largest_component_effects=dict(sorted(summaries.items(),key=lambda item:-item[1]['both_correct'].get('factor_group_mean',float('-inf')))[:12]))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True);parser.add_argument('--pairs',type=int,default=256);args=parser.parse_args()
    assert not (c.HERE/(args.name+'.json')).exists();torch.set_num_threads(1)
    budget=Budget(args.name,360,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
