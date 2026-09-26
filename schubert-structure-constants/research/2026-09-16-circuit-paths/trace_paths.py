"""Isolate successive causal paths, and block downstream responses separately."""
import json
from pathlib import Path
import numpy as np
import torch
import circuit_context as s
import richardson_model as rm
import circuit_forward as cf
from pair_circuit_diagnostic import contrast, orbit_keys, summarize
from circuit_resources import Budget


def head_patch(cache, layer, suffix, heads):
    site=f'L{layer}.{suffix}'
    def apply(current):
        result=current.clone();result[:,heads]=cache[site][:,heads]
        return result
    return {site:apply}


def token_patch(value, positions):
    return lambda current:torch.where(positions[:,:,None],value,current)


def main(budget):
    archive=s.ART/'HEAD_CONFIRMATION.npz'
    data=dict(np.load(archive));model=rm.load(s.CHECKPOINT).eval()
    attribution=dict(np.load(s.PREVIOUS/'artifacts/MLP_PATH_ATTRIBUTION.npz'))
    ranking=attribution['ranking']
    keys=orbit_keys(data['x']);ids={k:i for i,k in enumerate(dict.fromkeys(keys))}
    groups=np.asarray([ids[k] for k in keys]);n=len(groups)
    plan=dict(status='fixed_before_interventions',checkpoint_sha256=s.sha(s.CHECKPOINT),
        source_sha256=s.sha(Path(__file__)),cohort_sha256=s.sha(archive),inputs=n,pairs=n//2,
        factor_orbits=len(ids),role='Exploratory multistage path analysis on the confirmation cohort; no model or neuron reselection',
        early_heads=[0,8],late_heads=[3,11],layers='Zero-based',
        source='The other member of an exact pair with the same candidates and disjoint positive coefficient supports',
        intervention='Change first-MLP hidden units at noncandidate positions, recompute second-layer keys and values, then replay only those keys and values on the unchanged destination. Successive isolated replays transmit only the named receiver activation; the baseline residual and other incoming activations are restored at each replay.',
        blocking='In the early-head replay, restore a downstream component to its native destination activation. This blocks its response to the intervention while retaining other paths. It differs from setting the component to zero.',
        effects='Change in logsumexp scores on donor-positive indices minus logsumexp scores on destination-positive indices. Effects of isolated paths and blocking are not assumed additive.',
        comparisons=['early full versus early selected heads','selected heads at candidate versus other query positions',
            'selected early attention versus transmission through the next MLP only',
            'later MLP response blocked separately and jointly',
            'serial path through L1 MLP, L3 selected heads and L3 MLP'],
        uncertainty='Descriptive 95% percentile bootstrap over factor orbits; exploratory comparisons are not adjusted for multiplicity.')
    assert not (s.HERE/'PATH_TRACE_PLAN.json').exists();s.write('PATH_TRACE_PLAN.json',plan)
    pieces={};checks=dict(native=0.,identity=0.,head_replay=0.,last_state_candidate_replay=0.)
    def add(key,value):pieces.setdefault(key,[]).append(value.detach().cpu().numpy())
    with torch.inference_mode():
        selected=torch.zeros(768,device='cuda',dtype=torch.bool)
        selected[np.r_[ranking[:32],ranking[:32]+384]]=True
        for begin in range(0,n,64):
            budget.check();end=min(begin+64,n)
            x=torch.as_tensor(data['x'][begin:end],device='cuda')
            mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
            y=torch.as_tensor(data['y'][begin:end],device='cuda')
            rr=torch.arange(len(x),device='cuda');partner=rr^1;sy=y[partner]
            other=(x[:,:,18]!=0)&~mask
            scores,dc=cf.explicit(model,x,capture=True);sc={k:v[partner] for k,v in dc.items()}
            base=contrast(scores,sy,y);goal=contrast(scores[partner],sy,y)
            pred=scores.masked_fill(~mask,-1e9).argmax(1);good=y[rr,pred]
            add('both_correct',good&good[partner]);add('baseline_choice',pred)
            add('baseline_contrast',base);add('source_contrast',goal)
            def record(name,result):
                choice=result.masked_fill(~mask,-1e9).argmax(1)
                add('effect_'+name,contrast(result,sy,y)-base)
                add('source_good_'+name,sy[rr,choice]);add('dest_good_'+name,y[rr,choice])
                add('choice_'+name,choice)
            def run(name,patches,capture=False):
                budget.check();result=cf.explicit(model,x,patches=patches,capture=capture)
                record(name,result[0] if capture else result)
                return result
            checks['native']=max(checks['native'],float((scores-model(x)).abs().max()))
            identity=cf.explicit(model,x,patches={'L0.mlp_hidden':dc['L0.mlp_hidden']})
            checks['identity']=max(checks['identity'],float((identity-scores).abs().max()))
            for count in [32,384]:
                hidden=torch.where(selected[None,None,:],sc['L0.mlp_hidden'],dc['L0.mlp_hidden']) if count==32 else sc['L0.mlp_hidden']
                _,upstream=run(f'noncandidate_mlp0_{count}',{'L0.mlp_hidden':token_patch(hidden,other)},True)
                modes={'all':list(range(16)),'selected':[0,8],'complement':[h for h in range(16) if h not in [0,8]]}
                for mode,heads in modes.items():
                    patch={**head_patch(upstream,1,'k',heads),**head_patch(upstream,1,'v',heads)}
                    result,changed=run(f'kv1_{mode}_{count}',patch,True)
                    if mode!='selected' or count!=32:continue
                    early_patch=patch;early_cache=changed;early_scores=result
                    for suffix in ['k','v']:
                        run(f'{suffix}1_selected_32',head_patch(upstream,1,suffix,heads))
            # Hold the complete early-attention effect fixed for every downstream test.
            replay=cf.explicit(model,x,patches=head_patch(early_cache,1,'head_outputs',[0,8]))
            checks['head_replay']=max(checks['head_replay'],float((replay-early_scores).abs().max()))
            head_site='L1.head_outputs'
            candidate_heads=torch.where(mask[:,None,:,None],early_cache[head_site],dc[head_site])
            other_heads=torch.where(other[:,None,:,None],early_cache[head_site],dc[head_site])
            _,candidate_cache=run('early_heads_candidate_queries',{head_site:candidate_heads},True)
            run('early_heads_other_queries',{head_site:other_heads})
            for position,positions in [('all',x[:,:,18]!=0),('candidate',mask),('other',other)]:
                run('early_to_mlp1_'+position,{'L1.mlp_output':token_patch(early_cache['L1.mlp_output'],positions)})
            # The short proposed serial path, with independent destination replays at each hop.
            _,first_mlp_cache=run('candidate_heads_to_candidate_mlp1',
                {'L1.mlp_output':token_patch(candidate_cache['L1.mlp_output'],mask)},True)
            for layer in [2,3,4,5]:
                run(f'mlp1_to_mlp{layer}_candidate',
                    {f'L{layer}.mlp_output':token_patch(first_mlp_cache[f'L{layer}.mlp_output'],mask)})
            late_site='L3.head_outputs'
            late=dc[late_site].clone()
            late[:,[3,11]]=torch.where(mask[:,None,:,None],first_mlp_cache[late_site][:,[3,11]],late[:,[3,11]])
            _,late_cache=run('mlp1_to_selected_heads3_candidate',{late_site:late},True)
            run('serial_mlp1_heads3_mlp3',{'L3.mlp_output':token_patch(late_cache['L3.mlp_output'],mask)})
            # Blocking measures whether a response matters in its original computational context.
            for layer in [1,2,3,4,5]:
                site=f'L{layer}.mlp_output'
                run(f'block_mlp{layer}',{**early_patch,site:dc[site]})
                run(f'block_candidate_mlp{layer}',{**early_patch,site:token_patch(dc[site],mask)})
            run('block_all_later_mlps',{**early_patch,**{f'L{k}.mlp_output':dc[f'L{k}.mlp_output'] for k in range(1,6)}})
            run('block_mlp1_and_mlp3',{**early_patch,**{f'L{k}.mlp_output':dc[f'L{k}.mlp_output'] for k in [1,3]}})
            for layer in [2,3,4,5]:
                run(f'block_attention{layer}',{**early_patch,f'L{layer}.head_outputs':dc[f'L{layer}.head_outputs']})
            run('block_selected_heads3',{**early_patch,**head_patch(dc,3,'head_outputs',[3,11])})
            run('block_all_later_attention',{**early_patch,**{f'L{k}.head_outputs':dc[f'L{k}.head_outputs'] for k in range(2,6)}})
            run('block_all_later_attention_and_mlp',{**early_patch,
                **{f'L{k}.head_outputs':dc[f'L{k}.head_outputs'] for k in range(2,6)},
                **{f'L{k}.mlp_output':dc[f'L{k}.mlp_output'] for k in range(1,6)}})
            # Candidate-state transmission across successive layers is a larger-path control.
            for layer in range(1,6):
                result=run(f'early_to_resid{layer}_candidate',
                    {f'L{layer}.resid_post':token_patch(early_cache[f'L{layer}.resid_post'],mask)})
                if layer==5:
                    checks['last_state_candidate_replay']=max(checks['last_state_candidate_replay'],
                        float((result-early_scores).masked_select(mask).abs().max()))
            print(json.dumps(dict(inputs=end,total=n,checks=checks)),flush=True)
    assert max(checks.values())==0.,checks
    arrays={k:np.concatenate(v) for k,v in pieces.items()};arrays['group']=groups
    comparisons={
        'all_minus_selected_early_heads':('kv1_all_32','kv1_selected_32'),
        'selected_minus_complement_early_heads':('kv1_selected_32','kv1_complement_32'),
        'candidate_minus_other_queries':('early_heads_candidate_queries','early_heads_other_queries'),
        'response_of_mlp1':('kv1_selected_32','block_mlp1'),
        'response_of_mlp3':('kv1_selected_32','block_mlp3'),
        'response_of_late_selected_heads':('kv1_selected_32','block_selected_heads3'),
        'response_of_all_later_mlps':('kv1_selected_32','block_all_later_mlps'),
        'response_of_all_later_attention':('kv1_selected_32','block_all_later_attention')}
    for name,(a,b) in comparisons.items():arrays['difference_'+name]=arrays['effect_'+a]-arrays['effect_'+b]
    rng=np.random.default_rng(s.SEED+20260916)
    summaries={subset:{name:summarize(value,include,groups,rng)
        for name,value in arrays.items() if name.startswith(('effect_','difference_'))}
        for subset,include in [('all',np.ones(n,bool)),('both_correct',arrays['both_correct'])]}
    choices={name.removeprefix('effect_'):dict(
        source_choice_fraction=float(arrays['source_good_'+name.removeprefix('effect_')].mean()),
        destination_pair_accuracy=float(arrays['dest_good_'+name.removeprefix('effect_')].reshape(-1,2).all(1).mean()))
        for name in arrays if name.startswith('effect_')}
    output=s.ART/'PATH_TRACE.npz';np.savez_compressed(output,**arrays)
    s.write('PATH_TRACE.json',dict(**{k:v for k,v in plan.items() if k!='status'},status='completed',checks=checks,
        native_pair_accuracy=float(arrays['both_correct'].mean()),summaries=summaries,choices=choices,artifact_sha256=s.sha(output)))
    print(json.dumps(dict(status='completed',inputs=n,pairs=n//2,checks=checks)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('multistage_path_trace',300,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
