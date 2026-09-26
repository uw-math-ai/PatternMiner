"""Enumerate all receiver-head coalitions for a fixed first-MLP intervention."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import circuit_context as s
import richardson_model as rm
import circuit_forward as cf
from pair_circuit_diagnostic import contrast, orbit_keys, summarize
from circuit_resources import Budget
from trace_paths import token_patch, head_patch


def group_mean(value,groups):
    return np.stack([value[...,groups==g].mean(-1) for g in np.unique(groups)],-1).mean(-1)


def decomposition(values):
    """Exact Shapley values and Shapley interaction indices for eight groups.

    Pair interactions use |S|!(n-|S|-2)!/(n-1)! times the second difference.
    They are not an additive partition of the total effect.
    """
    n=8;phi=np.zeros((n,values.shape[1]),dtype=np.float64)
    interactions=np.zeros((n,n,values.shape[1]),dtype=np.float64)
    for i in range(n):
        for mask in range(256):
            if mask&(1<<i):continue
            k=mask.bit_count();weight=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            phi[i]+=weight*(values[mask|(1<<i)]-values[mask])
        for j in range(i+1,n):
            for mask in range(256):
                if mask&((1<<i)|(1<<j)):continue
                k=mask.bit_count();weight=math.factorial(k)*math.factorial(n-k-2)/math.factorial(n-1)
                delta=values[mask|(1<<i)|(1<<j)]-values[mask|(1<<i)]-values[mask|(1<<j)]+values[mask]
                interactions[i,j]+=weight*delta
            interactions[j,i]=interactions[i,j]
    return phi,interactions


def main(args,budget):
    part=args.part;input_path=s.ART/('DISCOVERY_SCAN.npz' if part=='discovery' else 'HEAD_CONFIRMATION.npz')
    data=dict(np.load(input_path));model=rm.load(s.CHECKPOINT).eval()
    ranking=np.load(s.PREVIOUS/'artifacts/MLP_PATH_ATTRIBUTION.npz')['ranking']
    keys=orbit_keys(data['x']);ids={k:i for i,k in enumerate(dict.fromkeys(keys))};groups=np.array([ids[k] for k in keys])
    plan=dict(role='Exploratory exhaustive head-coalition analysis after the first confirmation study',
        checkpoint_sha256=s.sha(s.CHECKPOINT),source_sha256=s.sha(Path(__file__)),part=part,
        source_data_sha256=s.sha(input_path),inputs=len(groups),factor_orbits=len(ids),
        intervention='The fixed top 32 first-MLP unit pairs are replaced at noncandidate positions. Replay changed L1 keys and/or values for a coalition of widening head pairs (h,h+8), leaving all other incoming activations native.',
        coalitions=256,modes=['keys','values','keys_and_values'],
        selection='Using discovery only, select the fewest head pairs with <=10% group-balanced mean absolute contrast-effect discrepancy and >=99% exact agreement with choices under the full key/value intervention. Break ties by discrepancy, then integer mask.',
        inference='The confirmation cohort was already examined when this experiment was designed. The fixed discovery-selected subset receives a separate-cohort replay, not a new blind confirmation claim.',
        uncertainty='Group-bootstrap intervals are descriptive and uncorrected for multiplicity.')
    name='COALITIONS_'+part.upper();assert not (s.HERE/(name+'_PLAN.json')).exists()
    if part=='confirmation':
        selection=json.loads((s.HERE/'COALITION_SELECTION.json').read_text())
        plan['selection_sha256']=s.sha(s.HERE/'COALITION_SELECTION.json')
    s.write(name+'_PLAN.json',plan)
    effects={mode:[] for mode in plan['modes']};choices={mode:[] for mode in plan['modes']};native_good=[]
    checks=dict(native=0.,identity=0.,full_path_replay=0.)
    with torch.inference_mode():
        selected=torch.zeros(768,device='cuda',dtype=torch.bool);selected[np.r_[ranking[:32],ranking[:32]+384]]=True
        for begin in range(0,len(groups),64):
            budget.check();end=min(begin+64,len(groups))
            x=torch.as_tensor(data['x'][begin:end],device='cuda');mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
            y=torch.as_tensor(data['y'][begin:end],device='cuda');rr=torch.arange(len(x),device='cuda');partner=rr^1;sy=y[partner]
            native,dc=cf.explicit(model,x,capture=True);base=contrast(native,sy,y)
            checks['native']=max(checks['native'],float((native-model(x)).abs().max()))
            pred=native.masked_fill(~mask,-1e9).argmax(1);native_good.append(y[rr,pred].cpu().numpy())
            hidden=torch.where(selected[None,None,:],dc['L0.mlp_hidden'][partner],dc['L0.mlp_hidden'])
            _,upstream=cf.explicit(model,x,patches={'L0.mlp_hidden':token_patch(hidden,(x[:,:,18]!=0)&~mask)},capture=True)
            for mode in plan['modes']:
                local=[];predictions=[]
                for coalition in range(256):
                    budget.check();heads=[h for j in range(8) if coalition&(1<<j) for h in [j,j+8]]
                    suffixes=['k','v'] if mode=='keys_and_values' else ['k' if mode=='keys' else 'v']
                    patches={site:value for suffix in suffixes for site,value in head_patch(upstream,1,suffix,heads).items()}
                    result=cf.explicit(model,x,patches=patches)
                    if coalition==0:checks['identity']=max(checks['identity'],float((result-native).abs().max()))
                    local.append((contrast(result,sy,y)-base).cpu().numpy())
                    predictions.append(result.masked_fill(~mask,-1e9).argmax(1).cpu().numpy())
                effects[mode].append(np.stack(local));choices[mode].append(np.stack(predictions))
            if part=='confirmation':
                full=np.load(s.ART/'MLP_CONFIRMATION.npz')['effect_top_32'][begin:end]
                checks['full_path_replay']=max(checks['full_path_replay'],float(np.abs(effects['keys_and_values'][-1][-1]-full).max()))
            print(json.dumps(dict(part=part,inputs=end,total=len(groups))),flush=True)
    assert max(checks.values())==0.,checks
    effects={k:np.concatenate(v,1) for k,v in effects.items()};choices={k:np.concatenate(v,1) for k,v in choices.items()}
    arrays=dict(group=groups,native_good=np.concatenate(native_good));report={};rng=np.random.default_rng(s.SEED+602)
    include=np.ones(len(groups),bool)
    for mode,values in effects.items():
        phi,interactions=decomposition(values.astype(np.float64))
        efficiency=float(np.max(np.abs(phi.sum(0)-(values[-1]-values[0]))))
        assert efficiency<1e-9
        arrays.update({mode+'_effects':values,mode+'_choices':choices[mode],mode+'_shapley':phi,mode+'_interactions':interactions})
        singleton_sum=sum(values[1<<j] for j in range(8))
        report[mode]=dict(full_effect=summarize(values[-1],include,groups,rng),
            full_minus_sum_of_singletons=summarize(values[-1]-singleton_sum,include,groups,rng),
            shapley_efficiency_max_error=efficiency,
            shapley={str(j):summarize(phi[j],include,groups,rng) for j in range(8)},
            interactions={f'{i},{j}':summarize(interactions[i,j],include,groups,rng) for i in range(8) for j in range(i+1,8)})
    values=effects['keys_and_values'];predictions=choices['keys_and_values']
    discrepancy=group_mean(np.abs(values-values[-1]),groups)/group_mean(np.abs(values[-1]),groups)
    agreement=(predictions==predictions[-1]).mean(1)
    order=sorted(range(256),key=lambda m:(m.bit_count(),float(discrepancy[m]),m))
    if part=='discovery':
        chosen=next(m for m in order if discrepancy[m]<=.1 and agreement[m]>=.99)
        selection=dict(mask=chosen,pairs=[j for j in range(8) if chosen&(1<<j)],
            discovery_discrepancy=float(discrepancy[chosen]),discovery_agreement=float(agreement[chosen]),
            source_sha256=s.sha(Path(__file__)),checkpoint_sha256=s.sha(s.CHECKPOINT))
        assert not (s.HERE/'COALITION_SELECTION.json').exists();s.write('COALITION_SELECTION.json',selection)
    chosen=selection['mask']
    report['selected_replay']=dict(selection=selection,relative_effect_discrepancy=float(discrepancy[chosen]),
        full_intervention_choice_agreement=float(agreement[chosen]),effect=summarize(values[chosen],include,groups,rng))
    report['best_by_number_of_pairs']={str(k):dict(mask=(m:=min([m for m in range(256) if m.bit_count()==k],key=lambda m:discrepancy[m])),
        relative_effect_discrepancy=float(discrepancy[m]),choice_agreement=float(agreement[m])) for k in range(9)}
    arrays.update(relative_effect_discrepancy=discrepancy,choice_agreement=agreement)
    out=s.ART/(name+'.npz');np.savez_compressed(out,**arrays)
    s.write(name+'.json',dict(**plan,status='completed',checks=checks,results=report,artifact_sha256=s.sha(out)))
    print(json.dumps(dict(status='completed',part=part,selected=report['selected_replay'],checks=checks)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['discovery','confirmation'],required=True);args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget('head_coalitions_'+args.part,420,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
