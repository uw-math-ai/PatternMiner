"""Directly replace every attention head and MLP on all development pairs."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import session as s
import richardson_model as rm
from final_development_decomposition import bank
from formal_scan import gradient_scores,scan,summaries
from full_resources import Budget


def main(budget):
    plan=json.loads((s.HERE/'FINAL_COMPONENT_SCAN_PLAN.json').read_text())
    checkpoint=s.HERE/plan['checkpoint'];digest=s.sha(checkpoint)
    model=rm.load(checkpoint).eval();data,names,_=bank()
    assert not (s.HERE/'FINAL_COMPONENT_SCAN.json').exists()
    gradient=gradient_scores(model,data,budget)
    sites={f'L{layer}H{head}':(f'L{layer}.head_outputs',head)
           for layer,block in enumerate(model.encoder.layers) for head in range(block.self_attn.num_heads)}
    sites.update({f'L{layer}MLP':(f'L{layer}.mlp_output',None) for layer in range(len(model.encoder.layers))})
    arrays,checks=scan(model,data,budget,sites)
    path=s.ART/'FINAL_COMPONENT_SCAN.npz'
    np.savez_compressed(path,**arrays,**{'gradient_'+key:value for key,value in gradient.items()},
                        **{'data_'+key:value for key,value in data.items()})
    report=dict(status='completed',checkpoint_sha256=digest,source_sha256=s.sha(Path(__file__)),
        plan_sha256=s.sha(s.HERE/'FINAL_COMPONENT_SCAN_PLAN.json'),artifact_sha256=s.sha(path),
        inputs=len(data['x']),pairs=len(data['x'])//2,factor_orbits=len(np.unique(data['group'])),
        populations=names,checks=checks,unmodified_both_pair_accuracy=float((arrays['source_good']&arrays['dest_good']).mean()),
        sites=summaries(arrays,data,sites),role='Development: all recorded difficult pairs, with no correctness selection.',
        limitation='All components are measured. Their ranking is exploratory; individual interventions are not additive and do not identify a complete algorithm.')
    assert s.sha(checkpoint)==digest;s.write('FINAL_COMPONENT_SCAN.json',report)
    ranked=sorted(report['sites'],key=lambda k:-report['sites'][k]['all']['factor_group_mean_change'])
    print(json.dumps(dict(status='completed',pairs=report['pairs'],checks=checks,
        largest_effects={k:report['sites'][k]['all'] for k in ranked[:12]})),flush=True)


if __name__=='__main__':
    while not (s.HERE/'FINAL_DEVELOPMENT_REPLAY_STATUS.json').exists():time.sleep(2)
    assert json.loads((s.HERE/'FINAL_DEVELOPMENT_REPLAY_STATUS.json').read_text())['status']=='completed'
    torch.set_num_threads(1);budget=Budget('final_component_scan',900,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
