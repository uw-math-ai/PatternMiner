"""Use disclosed errors to choose new, disjoint training factor pairs."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import training_data
from exposed_assessment import append_validation
import richardson_features as rf
import plan_neighborhoods as pn
from full_resources import Budget


def main(budget):
    assert not (s.HERE/'NEIGHBOR_PLAN.json').exists()
    with gzip.open(s.ART/'forbidden_factor_orbits.json.gz','rt') as stream:
        forbidden={tuple(tuple(p) for p in pair) for pair in json.load(stream)}
    fresh=json.loads((s.HERE/'DATA_PLAN.json').read_text())
    for spec in fresh['populations'].values():forbidden.update(pn.orbit(pair) for pair in spec['pairs'])
    previous=json.loads((s.PREVIOUS/'ROOT_ATTENTION_ABLATION.json').read_text())
    archive=s.PREVIOUS/'artifacts/ROOT_ATTENTION_ABLATION.npz'
    assert s.sha(archive)==previous['artifact_sha256']
    assert previous['checkpoint_sha256']==json.loads((s.HERE/'ERROR_SAMPLING_PLAN.json').read_text())['initial_checkpoint_sha256']
    saved=dict(np.load(archive));names=list(previous['populations'])
    datasets=append_validation(training_data.validation(True));anchors={}
    for index,(population,row,i) in enumerate(saved['examples']):
        name=names[population];data=datasets[name]
        scores=np.where(data['mask'][row],saved['full_scores'][index],-1e9)
        prediction=int(scores.argmax())
        if data['y'][row,prediction]:continue
        budget.check()
        p,n=rf.permutations(torch.as_tensor(data['x'][row:row+1]))
        N=int(n[0]);p=p[0,:,:N].numpy()
        pair=pn.normalized(dict(u=p[0].tolist(),v=p[1].tolist()))
        assert 2<=len(pair['u'])<=10
        key=pn.orbit(pair)
        item=anchors.setdefault(key,dict(pair=pair,origin='previously_exposed_development_failure',errors=[]))
        item['errors'].append(dict(population=name,row=int(row),supplied_index=int(i),predicted_index=prediction,w=p[2].tolist(),valid_indices=np.flatnonzero(data['y'][row]).tolist()))
    anchors=sorted(anchors.values(),key=lambda item:(-len(item['errors']),item['pair']['u'],item['pair']['v']))
    pools=[pn.candidates(item['pair'],forbidden,s.SEED+700+index) for index,item in enumerate(anchors)]
    chosen=set();selected=[];cost=0;cursor=[0]*len(pools)
    while cost<9000:
        progress=False
        for parent,pool in enumerate(pools):
            while cursor[parent]<len(pool):
                item=pool[cursor[parent]];cursor[parent]+=1;key=pn.orbit(item)
                if key in chosen or cost+item['exact_query_cost']>9000:continue
                chosen.add(key);cost+=item['exact_query_cost'];selected.append(dict(parent=parent,**item));progress=True;break
        if not progress:break
    assert selected and not chosen&forbidden
    report=dict(status='frozen_before_labels',anchors=anchors,pairs=selected,groups=len(selected),exact_query_cost=cost,
        forbidden_orbits=len(forbidden),source_sha256=s.sha(Path(__file__)),
        initial_checkpoint_sha256=previous['checkpoint_sha256'],prediction_archive_sha256=s.sha(archive),
        candidate_generator_sha256=s.sha(Path(pn.__file__)),fresh_plan_sha256=s.sha(s.HERE/'DATA_PLAN.json'),
        policy='Training only. Round robin over failure-factor neighborhoods, excluding all prior and freshly reserved factor orbits. Complete exact products supply labels; no assessment predictions are read. Not included in the initial matched sampler comparison.')
    s.write('NEIGHBOR_PLAN.json',report)
    print(json.dumps(dict(anchors=len(anchors),incorrect_supplied_inputs=sum(len(a['errors']) for a in anchors),groups=len(selected),exact_queries=cost)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('plan_current_failure_neighborhoods',180);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
