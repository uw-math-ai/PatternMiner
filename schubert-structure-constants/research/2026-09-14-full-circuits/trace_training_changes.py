"""Replace parameter groups to locate the new gain on exposed difficult pairs."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_model as rm
from full_resources import Budget


def groups(state):
    result={}
    for key in state:
        if key.startswith('encoder.layers.'):
            parts=key.split('.');layer=parts[2];component=parts[3]
            kind='attention' if component=='self_attn' else 'mlp' if component in ['linear1','linear2'] else 'normalization'
            assert component in ['self_attn','linear1','linear2','norm1','norm2']
            name=f'L{layer}_{kind}'
        elif key.startswith('input.'):name='.'.join(key.split('.')[:2])
        elif key.startswith('auxiliary_head.'):name='unused_auxiliary_readout'
        elif key.startswith('head.'):name='selector_readout'
        else:
            assert key in ['base_order_bias','root_action_order_bias'];name=key
        result.setdefault(name,[]).append(key)
    return result


def main(budget):
    source=s.ART/'second_neighbor_refinement_compact.pt';snapshot=s.ART/'second_refinement_weight_study.pt'
    assert not snapshot.exists();before_hash=s.sha(source);shutil.copyfile(source,snapshot);assert before_hash==s.sha(source)==s.sha(snapshot)
    saved=torch.load(snapshot,map_location='cpu',weights_only=True)
    prior=s.ART/'teacher_stability.pt';initial=torch.load(prior,map_location='cpu',weights_only=True)
    state0=initial['state_dict'];state1=saved['state_dict'];assert set(state0)==set(state1)
    partition=groups(state0)
    assert sum(map(len,partition.values()))==len(state0)
    names=['first_exposed_S10_pairs','new_S10_validation_pairs']
    populations={name:dict(np.load(s.ART/(name+'.npz'))) for name in names}
    plan=dict(status='fixed_before_parameter_interventions',initial_sha256=s.sha(prior),updated_sha256=s.sha(snapshot),updated_step=saved['step'],
        source_sha256=s.sha(Path(__file__)),parameter_groups=partition,populations={name:s.sha(s.ART/(name+'.npz')) for name in names},
        directions=['Insert the updated parameter group into the earlier model','Restore the earlier parameter group in the updated model'],
        role='Development analysis of an adaptive training change. It does not isolate the effects of individual training ingredients, and no protected predictions are read.')
    assert not (s.HERE/'TRAINING_WEIGHT_REPLACEMENT_PLAN.json').exists();s.write('TRAINING_WEIGHT_REPLACEMENT_PLAN.json',plan)
    model=rm.load(prior).eval();assert model.supplied_index_independent
    assert set(model.state_dict())==set(state0)
    metrics={name:{} for name in names};arrays={name:{} for name in names}
    def evaluate(condition,state):
        model.load_state_dict(state)
        with torch.inference_mode():
            for name,data in populations.items():
                scores=[]
                for start in range(0,len(data['x']),512):
                    budget.check();scores.append(model(torch.as_tensor(data['x'][start:start+512],device='cuda')).cpu().numpy())
                score=np.concatenate(scores);pred=np.where(data['mask'],score,-1e9).argmax(1)
                good=data['y'][np.arange(len(pred)),pred];pair=good[::2]&good[1::2]
                arrays[name][condition+'_scores']=score;arrays[name][condition+'_good']=good
                metrics[name][condition]=dict(pairs=len(pair),correct_pairs=int(pair.sum()),pair_accuracy=float(pair.mean()))
    evaluate('initial',state0);evaluate('updated',state1)
    for name,keys in partition.items():
        forward=dict(state0);forward.update({key:state1[key] for key in keys})
        reverse=dict(state1);reverse.update({key:state0[key] for key in keys})
        evaluate(name+'_insert',forward);evaluate(name+'_restore',reverse)
    checks={}
    for name in names:
        assert np.array_equal(arrays[name]['initial_scores'],arrays[name]['unused_auxiliary_readout_insert_scores'])
        assert np.array_equal(arrays[name]['updated_scores'],arrays[name]['unused_auxiliary_readout_restore_scores'])
        for condition,result in metrics[name].items():
            good=arrays[name][condition+'_good'];pair=good[::2]&good[1::2]
            for base in ['initial','updated']:
                reference=arrays[name][base+'_good'];reference=reference[::2]&reference[1::2]
                result['fixed_vs_'+base]=int((~reference&pair).sum());result['lost_vs_'+base]=int((reference&~pair).sum())
        path=s.ART/('TRAINING_WEIGHT_REPLACEMENT_'+name+'.npz');np.savez_compressed(path,**arrays[name]);checks[name]=s.sha(path)
    parameter_changes={name:dict(parameters=sum(state0[key].numel() for key in keys),maximum_change=max(float((state1[key]-state0[key]).abs().max()) for key in keys)) for name,keys in partition.items()}
    report=dict(**{k:v for k,v in plan.items() if k!='status'},status='completed',metrics=metrics,parameter_changes=parameter_changes,artifacts=checks,
        checks=dict(unused_auxiliary_head_has_exactly_zero_effect=True),
        limitation='Parameter hybrids are artificial models. Individual group effects need not add, and these development measurements do not constitute formal circuit confirmation.')
    s.write('TRAINING_WEIGHT_REPLACEMENT.json',report)
    print(json.dumps(dict(status='completed',step=saved['step'],metrics=metrics)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('training_weight_replacement',300,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
