"""Separate inverse-position midpoint, difference, and left descent in the fixed path."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import session as s
from expanded_decomposition import bank
import richardson_model as rm
import circuit_forward as cf
from mlp_path_attribution import path_forward
from pair_circuit_diagnostic import contrast,summarize
from full_resources import Budget


def main(budget):
    prior=json.loads((s.HERE/'EXPANDED_DECOMPOSITION.json').read_text());assert prior['status']=='completed'
    comparison=json.loads((s.HERE/'ERROR_SAMPLING_COMMON_WINDOW_COMPARISON.json').read_text())
    checkpoint=s.ART/(comparison['winner']+'_common5000.pt');assert s.sha(checkpoint)==prior['checkpoint_sha256']
    data,names,attribution=bank();model=rm.load(checkpoint).eval();layer=model.encoder.layers[0]
    units=np.r_[attribution['ranking'][:32],attribution['ranking'][:32]+384]
    old=dict(np.load(s.ART/'EXPANDED_DECOMPOSITION.npz'));assert np.array_equal(old['x'],data['x'])
    plan=dict(status='fixed_before_interventions',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),
        parent_analysis_sha256=s.sha(s.HERE/'EXPANDED_DECOMPOSITION.json'),inputs=len(data['x']),
        hypothesis='The inverse-position contribution can be separated into the midpoint, signed difference, and left-descent indicator, without fitting a probe.',
        formulas=dict(midpoint='(w^{-1}(j)+w^{-1}(j+1))/(2n)',gap='(w^{-1}(j+1)-w^{-1}(j))/n',descent='1[w^{-1}(j)>w^{-1}(j+1)]'),
        role='Adaptive follow-up on the same development populations; not independent confirmation')
    assert not (s.HERE/'POSITION_DECOMPOSITION_PLAN.json').exists();s.write('POSITION_DECOMPOSITION_PLAN.json',plan)
    effects={};choices={};checks=dict(position_identity=0.,retained_full_path_replay=0.)
    with torch.inference_mode():
        V=layer.linear1.weight*layer.norm1.weight[None,:];V=V-V.mean(1,keepdim=True)
        E=V@torch.cat([model.input.base.weight,model.input.extra.weight],1)
        selected=torch.zeros(768,device='cuda',dtype=torch.bool);selected[units]=True
        for begin in range(0,len(data['x']),64):
            budget.check();end=min(begin+64,len(data['x']))
            x=torch.as_tensor(data['x'][begin:end],device='cuda');mask=torch.as_tensor(data['mask'][begin:end],device='cuda')
            y=torch.as_tensor(data['y'][begin:end],device='cuda');rr=torch.arange(len(x),device='cuda');partner=rr^1;sy=y[partner]
            other=(x[:,:,18]!=0)&~mask;scores,dc=cf.explicit(model,x,capture=True)
            sc={key:value[partner] for key,value in dc.items()};base=contrast(scores,sy,y)
            sigma=((sc['embedding']+sc['L0.attention_output']).var(-1,unbiased=False,keepdim=True)+layer.norm1.eps).sqrt()
            delta=sc['features']-dc['features'];assert not delta[:,:,[13,14,21]].count_nonzero()
            da,db=delta[:,:,7:8],delta[:,:,8:9]
            parts=dict(midpoint=(da+db)/2*(E[:,7]+E[:,8])/sigma,
                gap=(db-da)/2*(E[:,8]-E[:,7])/sigma,
                descent=delta[:,:,16:17]*E[:,16]/sigma)
            direct=F.linear(delta[:,:,[7,8,16]],E[:,[7,8,16]])/sigma
            checks['position_identity']=max(checks['position_identity'],float((sum(parts.values())-direct).abs().max()))
            changes={**parts,'positions':parts['midpoint']+parts['gap'],'positions_and_descent':sum(parts.values()),
                     'gap_and_descent':parts['gap']+parts['descent'],'midpoint_and_descent':parts['midpoint']+parts['descent']}
            pd=layer.linear1(dc['L0.mlp_input'])
            for name,change in changes.items():
                hidden=F.gelu(pd+change*selected[None,None,:],approximate='tanh')
                result=path_forward(model,x,hidden,other)
                effects.setdefault(name,[]).append((contrast(result,sy,y)-base).cpu().numpy())
                choices.setdefault(name,[]).append(sy[rr,result.masked_fill(~mask,-1e9).argmax(1)].cpu().numpy())
            changed=torch.where(selected[None,None,:],sc['L0.mlp_hidden'],dc['L0.mlp_hidden'])
            full=(contrast(path_forward(model,x,changed,other),sy,y)-base).cpu().numpy()
            checks['retained_full_path_replay']=max(checks['retained_full_path_replay'],float(np.abs(full-old['effect_top_32'][begin:end]).max()))
    assert checks['position_identity']<1e-4 and checks['retained_full_path_replay']==0.,checks
    arrays={f'effect_{k}':np.concatenate(v) for k,v in effects.items()}
    arrays.update({f'source_choice_{k}':np.concatenate(v) for k,v in choices.items()})
    arrays.update(group=data['group'],previously_used_neuron_orbit=data['previously_used_neuron_orbit'],both_correct=old['both_correct'],units=units,
        midpoint_coefficients=(E[units,7]+E[units,8]).cpu().numpy(),gap_coefficients=((E[units,8]-E[units,7])/2).cpu().numpy(),descent_coefficients=E[units,16].cpu().numpy())
    rng=np.random.default_rng(s.SEED+850);summaries={}
    for label,include in [('all',np.ones(len(data['x']),bool)),('other_neuron_orbits',~data['previously_used_neuron_orbit']),
                          ('other_neuron_orbits_both_correct',~data['previously_used_neuron_orbit']&old['both_correct'])]:
        summaries[label]={k:summarize(arrays['effect_'+k],include,data['group'],rng) for k in effects}
    archive=s.ART/'POSITION_DECOMPOSITION.npz';np.savez_compressed(archive,**arrays)
    report=dict(**{k:v for k,v in plan.items() if k!='status'},status='completed',checks=checks,summaries=summaries,
        artifact_sha256=s.sha(archive),limitation='Effects after GELU and subsequent layers are not additive. The source normalization scale is fixed in the preactivation decomposition. No claim that the model computes a Bruhat projection is made.')
    s.write('POSITION_DECOMPOSITION.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('position_decomposition',420,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
