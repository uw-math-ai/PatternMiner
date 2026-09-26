"""Separate exploratory pruning that preserves validity of the selected witnesses."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_model as rm
import circuit_forward as cf
import compact
from prepare import encoded
from final_development_decomposition import bank
from full_resources import Budget


def zeroed_heads(removed):
    by_layer={}
    for layer,head in removed:by_layer.setdefault(layer,[]).extend([head,head+8])
    patches={}
    for layer,indices in by_layer.items():
        def replace(value,indices=indices):
            value=value.clone();value[:,indices]=0;return value
        patches[f'L{layer}.head_outputs']=replace
    return patches


def predict(model,data,removed,budget,native_check=False):
    choices=[];maximum=0.;patches=zeroed_heads(removed)
    with torch.inference_mode():
        for begin in range(0,len(data['x']),256):
            budget.check();x=torch.as_tensor(data['x'][begin:begin+256],device='cuda')
            mask=torch.as_tensor(data['mask'][begin:begin+256],device='cuda')
            scores=cf.explicit(model,x,patches=patches)
            if native_check:maximum=max(maximum,float((scores-model(x)).abs().max()))
            choices.append(scores.masked_fill(~mask,-1e9).argmax(1).cpu().numpy())
    if native_check:assert maximum==0
    return np.concatenate(choices),maximum


def metrics(choices,reference,data):
    agreement=choices==reference;pair_agreement=agreement[::2]&agreement[1::2]
    correct=data['y'][np.arange(len(choices)),choices];pair_correct=correct[::2]&correct[1::2]
    populations={str(int(p)):float(pair_agreement[data['population'][::2]==p].mean()) for p in np.unique(data['population'])}
    accuracies={str(int(p)):float(pair_correct[data['population'][::2]==p].mean()) for p in np.unique(data['population'])}
    return dict(input_prediction_agreement=float(agreement.mean()),pair_prediction_agreement=float(pair_agreement.mean()),
        minimum_population_pair_agreement=min(populations.values()),population_pair_agreement=populations,
        minimum_population_pair_accuracy=min(accuracies.values()),population_pair_accuracy=accuracies,
        correct_pairs=int(pair_correct.sum()),pair_accuracy=float(pair_correct.mean()))


def exposed_pairs(budget):
    assessment=json.loads((s.HERE/'SECOND_ASSESSMENT.json').read_text());assert not assessment['passes']
    pieces=[];membership=[];offset=0
    for number,n in enumerate([7,8,9,10]):
        budget.check();name=f'second_S{n}_assessment'
        with np.load(s.ART/f'second_assessment_{name}.npz') as a:pairs=a['pairs'].copy()
        records=compact.load(s.HERE/f'cache/data/{name}.bin')
        take=pairs[:,:2].reshape(-1);values=encoded(records[take],np.repeat(pairs[:,2].astype(bool),2))
        values['group']+=offset;offset=int(values['group'].max())+1
        values['population']=np.full(len(take),number);pieces.append(values)
        membership.extend([dict(population=name,rows=pair[:2].tolist(),reverse=int(pair[2])) for pair in pairs])
    return {key:np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]},membership


def main(budget):
    plan=json.loads((s.HERE/'TASK_HEAD_PRUNING_PLAN.json').read_text());checkpoint=s.HERE/plan['checkpoint']
    assert s.sha(checkpoint)==plan['checkpoint_sha256']
    model=rm.load(checkpoint).eval();data,names,_=bank();reference,error=predict(model,data,[],budget,True)
    groups=[(layer,head) for layer in range(6) for head in range(8)]
    with np.load(s.ART/'FINAL_COMPONENT_SCAN.npz') as archive:
        importance={pair:float(np.abs(archive[f'effect_L{pair[0]}H{pair[1]}']+archive[f'effect_L{pair[0]}H{pair[1]+8}']).mean()) for pair in groups}
    groups.sort(key=lambda pair:(importance[pair],pair))
    removed=[];history=[];predictions=reference
    baseline=metrics(reference,reference,data);minimum_correct=baseline['correct_pairs']-5
    s.write('TASK_HEAD_PRUNING_ORDER.json',dict(order=groups,importance=[importance[pair] for pair in groups],plan_sha256=s.sha(s.HERE/'TASK_HEAD_PRUNING_PLAN.json')))
    for pair in groups:
        trial=removed+[pair];choices,_=predict(model,data,trial,budget);values=metrics(choices,reference,data)
        accepted=values['correct_pairs']>=minimum_correct and values['minimum_population_pair_accuracy']>=.99
        if accepted:removed=trial;predictions=choices
        history.append(dict(pair=pair,accepted=accepted,removed_pairs=len(removed),metrics=values))
        print(json.dumps(history[-1]),flush=True)
    kept=[pair for pair in groups if pair not in removed]
    # Freeze pruning before loading the replay cohort. No later revision.
    s.write('TASK_HEAD_PRUNING_SELECTION.json',dict(status='frozen_before_external_replay',removed=removed,retained=kept,
        checkpoint_sha256=s.sha(checkpoint),development=metrics(predictions,reference,data),
        role='Development selection; every MLP, residual path and embedding remains. This mask does not define a complete algebraic algorithm.'))
    other,membership=exposed_pairs(budget);other_reference,check=predict(model,other,[],budget,True)
    other_predictions,_=predict(model,other,removed,budget)
    assessment=json.loads((s.HERE/'SECOND_ASSESSMENT.json').read_text())
    for pop,name in enumerate(assessment['populations']):
        keep=other['population']==pop;good=other['y'][np.arange(len(other_reference)),other_reference][keep]
        assert int((good[::2]&good[1::2]).sum())==assessment['populations'][name]['difficult_pairs_correct']
    arrays=dict(development_reference=reference,development_pruned=predictions,development_population=data['population'],
        development_group=data['group'],development_y=data['y'],replay_reference=other_reference,replay_pruned=other_predictions,
        replay_population=other['population'],replay_group=other['group'],replay_y=other['y'])
    archive=s.ART/'TASK_HEAD_PRUNING.npz';np.savez_compressed(archive,**arrays)
    s.write('TASK_HEAD_PRUNING_REPLAY_MEMBERSHIP.json',dict(pairs=membership,role='Already exposed assessment; not used for pruning decisions, not formal confirmation'))
    s.write('TASK_HEAD_PRUNING.json',dict(status='completed',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),
        plan_sha256=s.sha(s.HERE/'TASK_HEAD_PRUNING_PLAN.json'),artifact_sha256=s.sha(archive),
        retained_head_pairs=kept,removed_head_pairs=removed,retained_heads=2*len(kept),history=history,
        development=metrics(predictions,reference,data),development_populations=names,
        exposed_replay=metrics(other_predictions,other_reference,other),native_checks=[error,check],
        scope='Task-validity pruning on all development pairs and a fixed replay on previously exposed assessment pairs. Different valid witnesses are allowed. No formal hypotheses or qualification claims change.'))
    assert s.sha(checkpoint)==plan['checkpoint_sha256']


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('development_task_head_pruning',600,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
