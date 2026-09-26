"""Compare opposite-support and same-support donors along a fixed MLP path."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
from expanded_decomposition import bank
import richardson_model as rm
import circuit_forward as cf
from mlp_path_attribution import path_forward
from pair_circuit_diagnostic import contrast,summarize
from full_resources import Budget


def main(args,budget):
    membership=json.loads((s.HERE/'SUPPORT_CONTROL_MEMBERSHIP.json').read_text())
    path=s.ART/'SUPPORT_CONTROL_MEMBERSHIP.npy';assert s.sha(path)==membership['artifact_sha256']
    entries=np.load(path);data,names,attribution=bank()
    assert len(data['x'])==membership['bank_inputs'] and names==membership['populations']
    checkpoint=Path(args.checkpoint);model=rm.load(checkpoint).eval()
    assert model.supplied_index_independent
    ranking=attribution['ranking'];random_sets=attribution['control_16_selections']
    subsets={f'top_{k}':ranking[:k] for k in [16,32,64,384]}
    subsets.update({f'random_16_{k}':v for k,v in enumerate(random_sets)})
    plan=dict(status='fixed_before_interventions',checkpoint_sha256=s.sha(checkpoint),
        source_sha256=s.sha(Path(__file__)),membership_sha256=s.sha(s.HERE/'SUPPORT_CONTROL_MEMBERSHIP.json'),
        inputs=len(entries),path='First-block MLP hidden units at non-candidate tokens, through second-block keys and values only',
        subsets='Earlier fixed neuron ranking and twenty fixed magnitude-matched random 16-pair subsets',
        endpoint='Change in log-sum-exp contrast of opposite paired target support against destination target support',
        role='Exploratory development comparison; no formal confirmation population opened')
    assert not (s.HERE/(args.name+'_PLAN.json')).exists();s.write(args.name+'_PLAN.json',plan)
    pieces={};checks=dict(native=0.,identity=0.,restoration=0.)
    def save(key,value):pieces.setdefault(key,[]).append(value.detach().cpu().numpy())
    with torch.inference_mode():
        for begin in range(0,len(entries),32):
            budget.check();take=entries[begin:begin+32];rows=take[:,0];same=take[:,1]
            x=torch.as_tensor(data['x'][rows],device='cuda');mask=torch.as_tensor(data['mask'][rows],device='cuda')
            y=torch.as_tensor(data['y'][rows],device='cuda');sy=torch.as_tensor(data['y'][rows^1],device='cuda')
            rr=torch.arange(len(x),device='cuda');other=(x[:,:,18]!=0)&~mask
            ss=torch.as_tensor(data['x'][same],device='cuda')
            n=torch.round(1/x[:,0,18]).long();ii=torch.round(x[:,0,1]*n).long()-1
            ss=c.prior.c.a.change_witness(ss,ii)
            native,dc=cf.explicit(model,x,capture=True)
            opposite,oc=cf.explicit(model,torch.as_tensor(data['x'][rows^1],device='cuda'),capture=True)
            same_scores,sc=cf.explicit(model,ss,capture=True)
            base=contrast(native,sy,y)
            good=y[rr,native.masked_fill(~mask,-1e9).argmax(1)]
            ogood=sy[rr,opposite.masked_fill(~mask,-1e9).argmax(1)]
            sgood=y[rr,same_scores.masked_fill(~mask,-1e9).argmax(1)]
            for key,value in dict(baseline_contrast=base,destination_correct=good,opposite_correct=ogood,same_correct=sgood,
                                  full_opposite_change=contrast(opposite,sy,y)-base,
                                  full_same_change=contrast(same_scores,sy,y)-base).items():save(key,value)
            checks['native']=max(checks['native'],float((model(x)-native).abs().max()))
            checks['identity']=max(checks['identity'],float((path_forward(model,x,dc['L0.mlp_hidden'],other)-native).abs().max()))
            for donor,cache,scores in [('opposite',oc,opposite),('same',sc,same_scores)]:
                restored=cf.explicit(model,x,patches={k:cache[k] for k in ['embedding','order_comparisons']})
                checks['restoration']=max(checks['restoration'],float((restored-scores).abs().max()))
                for name,units in subsets.items():
                    budget.check();selected=torch.zeros(768,device='cuda',dtype=torch.bool)
                    selected[units]=True;selected[units+384]=True
                    hidden=torch.where(selected[None,None,:],cache['L0.mlp_hidden'],dc['L0.mlp_hidden'])
                    result=path_forward(model,x,hidden,other)
                    save(f'{donor}_effect_{name}',contrast(result,sy,y)-base)
                    save(f'{donor}_choice_{name}',sy[rr,result.masked_fill(~mask,-1e9).argmax(1)])
            print(json.dumps(dict(inputs=begin+len(take),total=len(entries))),flush=True)
    assert max(checks.values())==0.,checks
    arrays={k:np.concatenate(v) for k,v in pieces.items()}
    arrays.update(membership=entries,group=data['group'][entries[:,0]])
    arrays['all_three_correct']=arrays['destination_correct']&arrays['opposite_correct']&arrays['same_correct']
    rng=np.random.default_rng(s.SEED+832);summaries={}
    for label,include in [('all',np.ones(len(entries),bool)),('all_three_correct',arrays['all_three_correct'])]:
        summaries[label]={}
        for name in ['top_16','top_32','top_64','top_384']:
            opp=arrays['opposite_effect_'+name];same=arrays['same_effect_'+name]
            summaries[label][name]={key:summarize(value,include,arrays['group'],rng) for key,value in
                                    [('opposite',opp),('same',same),('opposite_minus_same',opp-same)]}
        random=np.stack([arrays[f'opposite_effect_random_16_{k}']-arrays[f'same_effect_random_16_{k}'] for k in range(len(random_sets))]).mean(0)
        selected=arrays['opposite_effect_top_16']-arrays['same_effect_top_16']
        summaries[label]['top_16_specificity_minus_random']=summarize(selected-random,include,arrays['group'],rng)
    archive=s.ART/(args.name+'.npz');np.savez_compressed(archive,**arrays)
    report=dict(**{k:v for k,v in plan.items() if k!='status'},status='completed',checks=checks,
        factor_orbits=len(np.unique(arrays['group'])),all_three_correct=int(arrays['all_three_correct'].sum()),
        summaries=summaries,artifact_sha256=s.sha(archive),
        limitation='Targets and donor distances are matched on development data. Other statistics can differ; these tests do not identify a unique mathematical algorithm.')
    s.write(args.name+'.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True);args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget(args.name.lower(),900,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
