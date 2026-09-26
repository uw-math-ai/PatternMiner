"""Confirm a frozen attention-head selection with causal controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import circuit_context as s
from circuit_context import c
r=c.prior
import richardson_model as rm
import circuit_forward as cf
from circuit_scan import pair_data, contrast
from circuit_resources import Budget


def freeze(discovery, k, output):
    report=json.loads(discovery.read_text())
    assert report['configuration']['part']=='discovery'
    heads=[name for name in report['sites'] if 'H' in name]
    assert k == 4
    pairs=[(layer,head) for layer in range(6) for head in range(8)]
    def effect(pair):
        layer,head=pair
        return sum(report['sites'][f'L{layer}H{h}']['both_correct']['factor_group_mean_change'] for h in [head,head+8])
    pairs.sort(key=lambda pair:(-effect(pair),pair))
    selected=[f'L{layer}H{h}' for layer,head in pairs[:2] for h in [head,head+8]]
    rng=np.random.default_rng(r.SEED+773)
    per_layer=Counter(layer for layer,head in pairs[:2])
    controls=[]
    for _ in range(20):
        controls.append([f'L{layer}H{int(h)}' for layer,count in sorted(per_layer.items())
                         for head in rng.choice(8,count,replace=False) for h in [head,head+8]])
    result=dict(discovery=str(discovery),discovery_sha256=r.c.a.sha(discovery),
                checkpoint_sha256=report['checkpoint_sha256'],selected=selected,random_controls=controls,
                criterion='Two widening pairs with largest sum of individual factor-group mean contrast effects among originally correct pairs',
                control_design='20 random sets matching the layer counts and widening-pair topology')
    s.write(output.name,result)
    return result


def patches_for(cache, source, heads, mode):
    selected={}
    kinds={'output':['head_outputs'],'pattern':['pattern'],'values':['v'],
           'query':['q'],'key':['k'],'bias':['order_bias'],
           'query_key_bias':['q','k','order_bias'],'zero':['head_outputs']}
    for name in heads:
        layer,head=name.split('H');head=int(head)
        for suffix in kinds[mode]:
            site=f'{layer}.{suffix}'
            selected.setdefault(site,[]).append(head)
    def replacement(site,indices):
        def apply(current):
            result=current.clone()
            result[:,indices]=0 if mode=='zero' else source[site][:,indices]
            return result
        return apply
    return {site:replacement(site,indices) for site,indices in selected.items()}


def main(args,budget):
    output=s.HERE/f'{args.name}.json'
    selection_path=s.HERE/f'{args.name}_SELECTION.json'
    if output.exists() or selection_path.exists():raise RuntimeError('Use a fresh confirmation name')
    checkpoint=Path(args.checkpoint)
    assessment=json.loads(Path(args.assessment).read_text())
    s.verify_checkpoint(checkpoint, args.assessment)
    selection=freeze(Path(args.discovery),args.heads,selection_path)
    assert selection['checkpoint_sha256']==r.c.a.sha(checkpoint)
    model=rm.load(checkpoint).eval()
    marker=s.HERE/'FORMAL_CONFIRMATION_OPENED.json'
    with marker.open('x') as stream:json.dump(dict(checkpoint_sha256=s.sha(checkpoint),selection_sha256=s.sha(selection_path)),stream,indent=2)
    data,metadata=pair_data('confirmation',args.pairs,budget)
    # Membership and selection are written before any confirmation prediction.
    s.write(f'{args.name}_MEMBERSHIP.json',dict(pairs=metadata,selection_sha256=r.c.a.sha(selection_path)))
    all_heads=[f'L{layer}H{head}' for layer,block in enumerate(model.encoder.layers)
               for head in range(block.self_attn.num_heads)]
    selected=selection['selected']
    conditions={f'selected_{mode}':(selected,mode) for mode in
                ['output','pattern','values','query','key','bias','query_key_bias','zero']}
    conditions['retain_selected']=([h for h in all_heads if h not in selected],'zero')
    for index,heads in enumerate(selection['random_controls']):
        conditions[f'random_{index:02d}_output']=(heads,'output')
        conditions[f'random_{index:02d}_retain']=([h for h in all_heads if h not in heads],'zero')
    arrays={k:[] for k in ['baseline_contrast','source_contrast','baseline_good','source_good','shuffle_eligible','candidate_features_equal']}
    for name in [*conditions,'shuffled_selected_output']:
        for metric in ['change','source_choice','dest_choice']:arrays[f'{name}_{metric}']=[]
    checks=dict(native_max_error=0.,identity_max_error=0.,restoration_max_error=0.,query_key_pattern_max_error=0.)
    # Independently sample compatible sources from different factor groups.
    # This excludes both members of the destination's own pair and avoids
    # counting a two-member bucket's identity replacement as a shuffled control.
    rng=np.random.default_rng(r.SEED+991)
    partners=np.arange(len(data['x']))^1
    shuffled=partners.copy()
    shuffle_eligible=np.zeros(len(partners),bool)
    buckets={}
    for i,x in enumerate(data['x']):
        key=(round(1/float(x[0,18])),tuple(data['mask'][i].tolist()))
        buckets.setdefault(key,[]).append(i)
    for indices in buckets.values():
        indices=np.asarray(indices,dtype=np.int64)
        for index in indices:
            eligible=indices[data['group'][indices]!=data['group'][index]]
            if len(eligible):
                shuffled[index]=partners[int(rng.choice(eligible))]
                shuffle_eligible[index]=True
    with torch.inference_mode():
        for begin in range(0,len(partners),128):
            budget.check()
            end=min(begin+128,len(partners))
            x=torch.as_tensor(data['x'][begin:end],device='cuda')
            yy=torch.as_tensor(data['y'][begin:end],device='cuda')
            mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
            partner=torch.arange(len(x),device='cuda')^1
            arange=torch.arange(len(x),device='cuda')
            dest,dc=cf.explicit(model,x,capture=True)
            source=dest[partner]
            sc={name:value[partner] for name,value in dc.items()}
            _,shuffled_cache=cf.explicit(model,torch.as_tensor(data['x'][shuffled[begin:end]],device='cuda'),capture=True)
            base=contrast(dest,yy[partner],yy)
            goal=contrast(source,yy[partner],yy)
            pred=dest.masked_fill(~mask,-1e9).argmax(1)
            source_pred=source.masked_fill(~mask,-1e9).argmax(1)
            for key,value in [('baseline_contrast',base),('source_contrast',goal),
                              ('baseline_good',yy[arange,pred]),('source_good',yy[partner,source_pred])]:
                arrays[key].append(value.cpu().numpy())
            arrays['shuffle_eligible'].append(shuffle_eligible[begin:end])
            arrays['candidate_features_equal'].append(((dc['features']==sc['features'])|~mask[:,:,None]).all(-1).all(-1).cpu().numpy())
            checks['native_max_error']=max(checks['native_max_error'],float((model(x)-dest).abs().max()))
            identity=cf.explicit(model,x,patches={'L0.head_outputs':dc['L0.head_outputs']})
            checks['identity_max_error']=max(checks['identity_max_error'],float((identity-dest).abs().max()))
            restore=cf.explicit(model,x,patches={'embedding':sc['embedding'],'order_comparisons':sc['order_comparisons']})
            checks['restoration_max_error']=max(checks['restoration_max_error'],float((restore-source).abs().max()))
            actual_pattern=None
            for name in [*conditions,'shuffled_selected_output']:
                heads,mode=(selected,'output') if name=='shuffled_selected_output' else conditions[name]
                cache=shuffled_cache if name=='shuffled_selected_output' else sc
                actual=cf.explicit(model,x,patches=patches_for(dc,cache,heads,mode))
                if name=='selected_pattern':actual_pattern=actual
                if name=='selected_query_key_bias':
                    checks['query_key_pattern_max_error']=max(checks['query_key_pattern_max_error'],float((actual-actual_pattern).abs().max()))
                pred=actual.masked_fill(~mask,-1e9).argmax(1)
                for metric,value in [('change',contrast(actual,yy[partner],yy)-base),
                                     ('source_choice',yy[partner,pred]),('dest_choice',yy[arange,pred])]:
                    arrays[f'{name}_{metric}'].append(value.cpu().numpy())
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    arrays['group']=data['group']
    assert checks['native_max_error']<2e-5 and checks['query_key_pattern_max_error']<2e-5
    assert checks['identity_max_error']==checks['restoration_max_error']==0.
    report={}
    both=arrays['baseline_good']&arrays['source_good']
    for name in [*conditions,'shuffled_selected_output']:
        report[name]={}
        for label,include in [('all',np.ones(len(both),bool)),('both_correct',both),
                              ('both_correct_same_candidate_features',both&arrays['candidate_features_equal'])]:
            if name=='shuffled_selected_output':include=include&arrays['shuffle_eligible']
            denominator=(arrays['source_contrast']-arrays['baseline_contrast'])[include].mean() if include.any() else 0.
            effect=arrays[f'{name}_change'][include]
            group_means=[arrays[f'{name}_change'][include&(data['group']==group)].mean()
                         for group in np.unique(data['group']) if (include&(data['group']==group)).any()]
            report[name][label]=dict(inputs=int(include.sum()),raw_mean_change=float(effect.mean()) if len(effect) else None,
                aggregate_restoration_fraction=float(effect.mean()/denominator) if abs(denominator)>1e-8 else None,
                source_choice_fraction=float(arrays[f'{name}_source_choice'][include].mean()) if include.any() else None,
                dest_choice_fraction=float(arrays[f'{name}_dest_choice'][include].mean()) if include.any() else None,
                factor_groups=len(group_means),factor_group_mean_change=float(np.mean(group_means)) if group_means else None)
    np.savez_compressed(s.ART/f'{args.name}.npz',**{**arrays,**data})
    result=dict(configuration=vars(args),checkpoint_sha256=r.c.a.sha(checkpoint),checks=checks,
                source_sha256=r.c.a.sha(Path(__file__)),forward_sha256=r.c.a.sha(Path(cf.__file__)),selection=selection,conditions=report,
                unmodified_both_pair_accuracy=float(both.mean()),
                ablation_scope='Head outputs are set to zero. Unselected components are recomputed; embeddings, residual paths, and all MLPs remain active.',
                shuffled_source_scope='Uniform draws with replacement from the same ambient size and candidate mask, excluding the destination factor group.')
    s.write(output.name,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['conditions','selection']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--assessment',required=True)
    parser.add_argument('--discovery',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--heads',type=int,default=4)
    parser.add_argument('--pairs',type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(1)
    budget=Budget(args.name.lower(),420,gpu=True)
    status='failed_or_partial'
    try:
        main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
