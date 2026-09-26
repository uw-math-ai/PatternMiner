"""Test a fixed MLP path and its input contributions across development pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import session as s
from session import c
import richardson_model as rm
import circuit_forward as cf
import training_data
from exposed_assessment import append_validation
from pair_circuit_diagnostic import contrast,orbit_keys,summarize
from mlp_path_attribution import path_forward
from full_resources import Budget


def bank():
    populations=append_validation(training_data.validation(True))
    if (s.HERE/'PREPARED.json').exists():
        prepared=json.loads((s.HERE/'PREPARED.json').read_text())
        assert prepared['status']=='completed'
        for name in prepared['development']:
            path=s.ART/f'new_{name}_pairs.npz'
            if path.exists():populations[f'new_{name}_pairs']=dict(np.load(path))
    populations={name:data for name,data in populations.items() if name=='hard_validation' or name.endswith('_pairs')}
    data={key:np.concatenate([v[key] for v in populations.values()]) for key in ['x','mask','y','a']}
    labels=np.concatenate([np.full(len(v['x']),i) for i,v in enumerate(populations.values())])
    keys=orbit_keys(data['x']);unique={key:i for i,key in enumerate(dict.fromkeys(keys))}
    data['group']=np.asarray([unique[key] for key in keys]);data['population']=labels
    prior=dict(np.load(s.PREVIOUS/'artifacts/REFINEMENT_PAIR_CIRCUITS.npz'))
    attribution=dict(np.load(s.PREVIOUS/'artifacts/MLP_PATH_ATTRIBUTION.npz'))
    used=set(orbit_keys(prior['x'][attribution['selected']]))
    data['previously_used_neuron_orbit']=np.asarray([key in used for key in keys])
    assert np.array_equal(data['mask'][::2],data['mask'][1::2])
    assert not (data['y'][::2]&data['y'][1::2]).any()
    assert np.array_equal(data['x'][::2][data['mask'][::2]],data['x'][1::2][data['mask'][1::2]])
    return data,list(populations),attribution


def main(args,budget):
    checkpoint=Path(args.checkpoint);model=rm.load(checkpoint).eval()
    assert model.extra_kind=='relative_roots' and model.supplied_index_independent
    data,names,attribution=bank()
    if args.preflight:data={key:value[:64] for key,value in data.items()}
    count=len(data['x'])
    ranking=attribution['ranking'];units=np.r_[ranking[:32],ranking[:32]+384]
    random_sets=attribution['control_16_selections']
    plan=dict(status='fixed_before_interventions',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),
        inputs=count,pairs=count//2,factor_orbits=len(np.unique(data['group'])),populations=names,
        selection='Every recorded difficult pair in the declared development populations; no correctness filter. Results are separated by correctness and prior use in neuron analysis.',
        units='The earlier fixed ranking, with duplicated pairs at offsets k and k+384; no reselection',
        random_controls='All twenty fixed earlier magnitude-matched 16-pair subsets; no rematching',
        role='Implementation preflight on the first 64 inputs' if args.preflight else 'Expanded development analysis; no formal discovery or confirmation population is opened')
    assert not (s.HERE/(args.name+'_PLAN.json')).exists()
    s.write(args.name+'_PLAN.json',plan)
    effects={};choices={};pieces={key:[] for key in ['both_correct','equal_candidate_features','full_pair_effect']}
    checks={key:0. for key in ['native','identity','complete_input_restoration','destination_reconstruction','source_reconstruction','difference_decomposition','all_terms_effect_error']}
    layer=model.encoder.layers[0];assert not layer.norm_first
    with torch.inference_mode():
        V=layer.linear1.weight*layer.norm1.weight[None,:];V=V-V.mean(1,keepdim=True)
        W=torch.cat([model.input.base.weight,model.input.extra.weight],1)
        E=V@W;offset=V@model.input.base.bias;shift=layer.linear1.weight@layer.norm1.bias+layer.linear1.bias
        selected=torch.zeros(768,device='cuda',dtype=torch.bool);selected[units]=True
        columns={'w_original':[7,8,13,14,16,21],'w_ranks':list(range(90,124)),
                 'relative_roots':[124+i for i in list(range(4,8))+list(range(12,24))]}
        for begin in range(0,count,64):
            budget.check();end=min(begin+64,count)
            x=torch.as_tensor(data['x'][begin:end],device='cuda');mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
            y=torch.as_tensor(data['y'][begin:end],device='cuda');rr=torch.arange(len(x),device='cuda');partner=rr^1;sy=y[partner]
            other=(x[:,:,18]!=0)&~mask
            scores,dc=cf.explicit(model,x,capture=True);sc={key:value[partner] for key,value in dc.items()}
            base=contrast(scores,sy,y);good=y[rr,scores.masked_fill(~mask,-1e9).argmax(1)]
            pieces['both_correct'].append((good&good[partner]).cpu().numpy())
            pieces['equal_candidate_features'].append(((dc['features']==sc['features'])|~mask[:,:,None]).all(-1).all(-1).cpu().numpy())
            pieces['full_pair_effect'].append((contrast(scores[partner],sy,y)-base).cpu().numpy())
            native=model(x);identity=path_forward(model,x,dc['L0.mlp_hidden'],other)
            restored=cf.explicit(model,x,patches={'embedding':sc['embedding'],'order_comparisons':sc['order_comparisons']})
            for key,value in [('native',(native-scores).abs().max()),('identity',(identity-scores).abs().max()),('complete_input_restoration',(restored-scores[partner]).abs().max())]:
                checks[key]=max(checks[key],float(value))
            pd=layer.linear1(dc['L0.mlp_input']);ps=layer.linear1(sc['L0.mlp_input'])
            def terms(cache):
                total=cache['embedding']+cache['L0.attention_output']
                sigma=(total.var(-1,unbiased=False,keepdim=True)+layer.norm1.eps).sqrt()
                linear=F.linear(cache['features'],E,offset)+F.linear(cache['L0.attention_output'],V)
                return sigma,linear
            sd,ad=terms(dc);ss,ass=terms(sc);delta=sc['features']-dc['features']
            allowed=sum(columns.values(),[]);complement=[k for k in range(delta.shape[-1]) if k not in allowed]
            assert not delta[:,:,complement].count_nonzero()
            parts={name:F.linear(delta[:,:,cols],E[:,cols])/ss for name,cols in columns.items()}
            parts.update(first_attention=F.linear(sc['L0.attention_output']-dc['L0.attention_output'],V)/ss,
                         normalization=ad*(1/ss-1/sd))
            for key,value in [('destination_reconstruction',(ad/sd+shift-pd).abs().max()),('source_reconstruction',(ass/ss+shift-ps).abs().max()),('difference_decomposition',(sum(parts.values())-(ps-pd)).abs().max())]:
                checks[key]=max(checks[key],float(value))
            hidden={name:F.gelu(pd+change*selected[None,None,:],approximate='tanh') for name,change in parts.items()}
            hidden['all_input_terms']=F.gelu(pd+sum(parts[name] for name in columns)*selected[None,None,:],approximate='tanh')
            hidden['all_terms']=F.gelu(pd+sum(parts.values())*selected[None,None,:],approximate='tanh')
            hybrid=layer.linear1(layer.norm1(sc['embedding']+dc['L0.attention_output']))
            hidden['source_embedding_destination_attention']=torch.where(selected[None,None,:],F.gelu(hybrid,approximate='tanh'),dc['L0.mlp_hidden'])
            for number in [16,32,64,384]:
                include=torch.zeros(768,device='cuda',dtype=torch.bool);include[ranking[:number]]=True;include[ranking[:number]+384]=True
                hidden[f'top_{number}']=torch.where(include[None,None,:],sc['L0.mlp_hidden'],dc['L0.mlp_hidden'])
            hidden['without_32']=torch.where(selected[None,None,:],dc['L0.mlp_hidden'],sc['L0.mlp_hidden'])
            for index,chosen in enumerate(random_sets):
                include=torch.zeros(768,device='cuda',dtype=torch.bool);include[chosen]=True;include[chosen+384]=True
                hidden[f'random_16_{index}']=torch.where(include[None,None,:],sc['L0.mlp_hidden'],dc['L0.mlp_hidden'])
            local={}
            for name,value in hidden.items():
                budget.check();result=path_forward(model,x,value,other);change=(contrast(result,sy,y)-base).cpu().numpy()
                effects.setdefault(name,[]).append(change);local[name]=change
                choices.setdefault(name,[]).append(sy[rr,result.masked_fill(~mask,-1e9).argmax(1)].cpu().numpy())
            checks['all_terms_effect_error']=max(checks['all_terms_effect_error'],float(np.abs(local['all_terms']-local['top_32']).max()))
            if begin%1024==0:print(json.dumps(dict(inputs=end,total=count,checks=checks)),flush=True)
    assert max(checks[k] for k in ['native','identity','complete_input_restoration'])==0
    assert max(checks[k] for k in ['destination_reconstruction','source_reconstruction','difference_decomposition'])<1e-3
    assert checks['all_terms_effect_error']<1e-2
    arrays={key:np.concatenate(value) for key,value in pieces.items()}
    arrays.update(data);arrays.update({'effect_'+key:np.concatenate(value) for key,value in effects.items()})
    arrays.update({'source_correct_'+key:np.concatenate(value) for key,value in choices.items()})
    arrays.update(ranking=ranking,effective_input_coefficients=E[units].cpu().numpy(),random_16_selections=random_sets)
    random=np.stack([arrays[f'effect_random_16_{i}'] for i in range(len(random_sets))]).mean(0)
    arrays['selected_minus_random_16']=arrays['effect_top_16']-random
    arrays['removal_32']=arrays['effect_top_384']-arrays['effect_without_32']
    rng=np.random.default_rng(s.SEED+811)
    subsets={'all':np.ones(count,bool),'both_correct':arrays['both_correct'],
             'other_neuron_orbits':~arrays['previously_used_neuron_orbit'],
             'other_neuron_orbits_both_correct':~arrays['previously_used_neuron_orbit']&arrays['both_correct'],
             'equal_candidate_features':arrays['equal_candidate_features']}
    summaries={}
    for subset,include in subsets.items():
        summaries[subset]={name:summarize(arrays['effect_'+name],include,arrays['group'],rng) for name in effects if not name.startswith('random_')}
        summaries[subset]['selected_minus_random_16']=summarize(arrays['selected_minus_random_16'],include,arrays['group'],rng)
        summaries[subset]['removal_32']=summarize(arrays['removal_32'],include,arrays['group'],rng)
    archive=s.ART/(args.name+'.npz');np.savez_compressed(archive,**arrays)
    report=dict(**{key:value for key,value in plan.items() if key!='status'},status='completed',checks=checks,
        both_correct=int(arrays['both_correct'].sum()),other_neuron_orbit_inputs=int((~arrays['previously_used_neuron_orbit']).sum()),
        summaries=summaries,artifact_sha256=s.sha(archive),forward_sha256=s.sha(Path(cf.__file__)),
        preactivation_identity='p=(E f+V attention+b)/sigma+c; split the source-minus-destination difference using the source normalization scale',
        limitations='All populations are development data. Components are artificial interventions. Bootstrap intervals are exploratory and uncorrected for multiple comparisons. A concentrated effect is not a Schubert multiplication rule.')
    s.write(args.name+'.json',report)
    print(json.dumps(dict(status='completed',inputs=count,checks=checks,summaries=summaries['other_neuron_orbits'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget(args.name.lower(),1200,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
