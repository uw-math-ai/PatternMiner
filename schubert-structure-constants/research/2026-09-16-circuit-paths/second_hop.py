"""Test a second transfer from noncandidate to candidate positions."""
import json
from pathlib import Path
import numpy as np
import torch
import circuit_context as s
import richardson_model as rm
import circuit_forward as cf
from pair_circuit_diagnostic import contrast,orbit_keys,summarize
from trace_paths import token_patch
from circuit_resources import Budget


def main(budget):
    model=rm.load(s.CHECKPOINT).eval()
    ranking=np.load(s.PREVIOUS/'artifacts/MLP_PATH_ATTRIBUTION.npz')['ranking']
    plan=dict(role='Exploratory second-hop study motivated by completed path experiments',
        checkpoint_sha256=s.sha(s.CHECKPOINT),source_sha256=s.sha(Path(__file__)),
        cohorts=['confirmation','matched_development'],
        path='L0 noncandidate hidden units -> L1 keys/values -> L1 attention outputs at noncandidate positions -> positionwise residual/MLP transformations -> Lk keys/values -> Lk attention outputs at candidate positions -> downstream decision, for k=2,3,4,5.',
        isolation='Restore intervening attention outputs to native destination values. Candidate residual states must remain exactly unchanged before the receiving attention layer. Replay only the receiving keys/values, then only its candidate attention outputs. All other incoming states are restored at each replay.',
        meaning='These are controlled path effects, not additive components of a mathematical algorithm.',
        uncertainty='Descriptive group-bootstrap intervals; only four orbits in the matched development cohort.')
    assert not (s.HERE/'SECOND_HOP_PLAN.json').exists();s.write('SECOND_HOP_PLAN.json',plan)
    results={}
    with torch.inference_mode():
        selected=torch.zeros(768,device='cuda',dtype=torch.bool);selected[np.r_[ranking[:32],ranking[:32]+384]]=True
        for cohort,path in [('confirmation',s.ART/'MLP_CONFIRMATION.npz'),
                            ('matched_development',s.FULL/'artifacts/AUGMENTED_FEATURE_OBSTRUCTION.npz')]:
            data=dict(np.load(path));keys=orbit_keys(data['x']);ids={k:i for i,k in enumerate(dict.fromkeys(keys))}
            groups=np.array([ids[k] for k in keys]);pieces={};checks=dict(candidate_state_before_receiver=0.,identity=0.)
            for begin in range(0,len(groups),64):
                budget.check();end=min(begin+64,len(groups))
                x=torch.as_tensor(data['x'][begin:end],device='cuda');mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
                y=torch.as_tensor(data['y'][begin:end],device='cuda');rr=torch.arange(len(x),device='cuda');partner=rr^1;sy=y[partner]
                other=(x[:,:,18]!=0)&~mask;native,dc=cf.explicit(model,x,capture=True);base=contrast(native,sy,y)
                def record(name,result):
                    pieces.setdefault('effect_'+name,[]).append((contrast(result,sy,y)-base).cpu().numpy())
                    pred=result.masked_fill(~mask,-1e9).argmax(1)
                    pieces.setdefault('source_good_'+name,[]).append(sy[rr,pred].cpu().numpy())
                for count in [32,384]:
                    hidden=torch.where(selected[None,None,:],dc['L0.mlp_hidden'][partner],dc['L0.mlp_hidden']) if count==32 else dc['L0.mlp_hidden'][partner]
                    _,upstream=cf.explicit(model,x,patches={'L0.mlp_hidden':token_patch(hidden,other)},capture=True)
                    all_first,first=cf.explicit(model,x,patches={k:upstream[k] for k in ['L1.k','L1.v']},capture=True)
                    record(f'all_first_{count}',all_first)
                    other_heads=torch.where(other[:,None,:,None],first['L1.head_outputs'],dc['L1.head_outputs'])
                    first_patch={'L1.head_outputs':other_heads}
                    record(f'other_first_{count}',cf.explicit(model,x,patches=first_patch))
                    for layer in [2,3,4,5]:
                        held={**first_patch,**{f'L{k}.head_outputs':dc[f'L{k}.head_outputs'] for k in range(2,layer)}}
                        _,transmitted=cf.explicit(model,x,patches=held,capture=True)
                        site=f'L{layer}.resid_pre'
                        checks['candidate_state_before_receiver']=max(checks['candidate_state_before_receiver'],float((transmitted[site]-dc[site])[mask].abs().max()))
                        _,receiver=cf.explicit(model,x,patches={f'L{layer}.{suffix}':transmitted[f'L{layer}.{suffix}'] for suffix in ['k','v']},capture=True)
                        site=f'L{layer}.head_outputs'
                        candidate_output=torch.where(mask[:,None,:,None],receiver[site],dc[site])
                        record(f'receiver_{layer}_{count}',cf.explicit(model,x,patches={site:candidate_output}))
                        identity=cf.explicit(model,x,patches={site:dc[site]})
                        checks['identity']=max(checks['identity'],float((identity-native).abs().max()))
                print(json.dumps(dict(cohort=cohort,inputs=end,total=len(groups))),flush=True)
            assert max(checks.values())==0.,checks
            arrays={k:np.concatenate(v) for k,v in pieces.items()};arrays['group']=groups
            rng=np.random.default_rng(s.SEED+910);include=np.ones(len(groups),bool)
            out=s.ART/('SECOND_HOP_'+cohort+'.npz');np.savez_compressed(out,**arrays)
            results[cohort]=dict(inputs=len(groups),factor_orbits=len(ids),checks=checks,
                effects={k:summarize(v,include,groups,rng) for k,v in arrays.items() if k.startswith('effect_')},
                source_choice_counts={k:int(v.sum()) for k,v in arrays.items() if k.startswith('source_good_')},
                source_data_sha256=s.sha(path),artifact_sha256=s.sha(out))
    s.write('SECOND_HOP.json',dict(**plan,status='completed',results=results))


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('second_attention_hop',180,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
