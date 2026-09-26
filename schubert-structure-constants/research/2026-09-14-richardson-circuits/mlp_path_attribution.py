"""Integrated gradients for an isolated MLP-to-attention path, then direct tests."""
import json
from pathlib import Path
import numpy as np
import torch
import context as c
import richardson_model as rm
import circuit_forward as cf
from pair_circuit_diagnostic import contrast,summarize
from resources import Budget


def path_forward(model,x,hidden,other):
    _,changed=cf.explicit(model,x,patches={'L0.mlp_hidden':lambda current:torch.where(other[:,:,None],hidden,current)},capture=True)
    return cf.explicit(model,x,patches={site:changed[site] for site in ['L1.k','L1.v']})


def main(budget):
    checkpoint=c.ART/'screened_refinement_early.pt';model=rm.load(checkpoint).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    saved=dict(np.load(c.ART/'REFINEMENT_PAIR_CIRCUITS.npz'))
    selected=np.flatnonzero(saved['both_correct']&saved['same_full_candidate_features']);groups=saved['group'][selected]
    rng=np.random.default_rng(c.SEED+851);order=rng.permutation(np.unique(groups));discovery=order[:2*len(order)//3];reserved=order[2*len(order)//3:]
    membership=dict(discovery_groups=discovery.tolist(),reserved_groups=reserved.tolist(),seed=c.SEED+851,
        role='The reserved development groups are excluded from neuron selection. They are not independent accuracy assessment or the frozen formal circuit confirmation.')
    assert not (c.HERE/'MLP_PATH_MEMBERSHIP.json').exists();c.write('MLP_PATH_MEMBERSHIP.json',membership)
    array=dict(selected=selected,groups=groups);contributions=[];magnitudes=[];differences=[];completeness=[]
    discovery_rows=np.flatnonzero(np.isin(groups,discovery));reserved_rows=np.flatnonzero(np.isin(groups,reserved))
    def tensors(indices):
        raw=selected[indices];x=torch.as_tensor(saved['x'][raw],device='cuda');sx=torch.as_tensor(saved['x'][raw^1],device='cuda')
        y=torch.as_tensor(saved['y'][raw],device='cuda');sy=torch.as_tensor(saved['y'][raw^1],device='cuda')
        other=(x[:,:,18]!=0)&~torch.as_tensor(saved['mask'][raw],device='cuda')
        with torch.no_grad():
            native,dc=cf.explicit(model,x,capture=True);_,sc=cf.explicit(model,sx,capture=True)
        return x,y,sy,other,native,dc['L0.mlp_hidden'],sc['L0.mlp_hidden']
    for begin in range(0,len(discovery_rows),32):
        budget.check();indices=discovery_rows[begin:begin+32];x,y,sy,other,native,dest,source=tensors(indices)
        delta=(source-dest)*other[:,:,None]
        with torch.no_grad():
            base=contrast(native,sy,y);full=contrast(path_forward(model,x,source,other),sy,y)-base
            identity=path_forward(model,x,dest,other);assert float((identity-native).abs().max())==0.
        for points in [16,32,64]:
            budget.check();nodes,weights=np.polynomial.legendre.leggauss(points);total=torch.zeros_like(dest)
            for alpha,weight in zip((nodes+1)/2,weights/2):
                hidden=(dest+float(alpha)*delta).detach().requires_grad_(True)
                scores=path_forward(model,x,hidden,other);objective=contrast(scores,sy,y).sum()
                gradient,=torch.autograd.grad(objective,hidden);total+=float(weight)*gradient.detach()
            attribution=(delta*total).sum(1);residual=attribution.sum(1)-full
            relative=float(residual.abs().sum()/full.abs().sum().clamp(min=1e-8))
            if relative<=.01:break
        assert relative<=.02,('Integrated-gradient completeness failed',relative)
        contributions.append(attribution.cpu().numpy());magnitudes.append(delta.square().sum(1).cpu().numpy());differences.append(full.cpu().numpy())
        completeness.append(dict(inputs=len(indices),quadrature_points=points,relative_absolute_completeness_error=relative,maximum_absolute_completeness_error=float(residual.abs().max())))
    contribution=np.concatenate(contributions);magnitude=np.concatenate(magnitudes);full_effect=np.concatenate(differences)
    neuron_pairs=contribution[:,:384]+contribution[:,384:]
    means=np.stack([neuron_pairs[groups[discovery_rows]==g].mean(0) for g in discovery])
    ranking=np.argsort(-means.mean(0),kind='stable')
    amplitude=np.sqrt((magnitude[:,:384]+magnitude[:,384:]).mean(0));amplitude_order=np.argsort(amplitude)
    bins=np.empty(384,int);bins[amplitude_order]=np.arange(384)//24
    array.update(discovery_rows=discovery_rows,reserved_rows=reserved_rows,attributions=contribution,
                 squared_activation_changes=magnitude,full_discovery_path_effect=full_effect,ranking=ranking,amplitude_bins=bins)
    x,y,sy,other,native,dest,source=tensors(reserved_rows);base=contrast(native,sy,y)
    with torch.no_grad():full=contrast(path_forward(model,x,source,other),sy,y)-base
    array['full_reserved_path_effect']=full.cpu().numpy();results={};reserved_groups=groups[reserved_rows]
    for count in [1,2,4,8,16,32,64,128,384]:
        budget.check();chosen=ranking[:count];include=torch.zeros(768,device='cuda',dtype=torch.bool);include[chosen]=True;include[chosen+384]=True
        with torch.no_grad():
            sparse=torch.where(include[None,None,:],source,dest)
            effect=(contrast(path_forward(model,x,sparse,other),sy,y)-base).cpu().numpy()
        array[f'effect_top_{count}_pairs']=effect
        info=summarize(effect,np.ones(len(effect),bool),reserved_groups,rng)
        info['aggregate_full_path_fraction']=float(effect.mean()/array['full_reserved_path_effect'].mean());results[str(count)]=info
        if count in [16,64]:
            with torch.no_grad():
                removal=torch.where(include[None,None,:],dest,source)
                removed=(contrast(path_forward(model,x,removal,other),sy,y)-base).cpu().numpy()
            array[f'effect_without_{count}_pairs']=removed;controls=[];selections=[]
            for repeat in range(20):
                budget.check();control=np.concatenate([rng.choice(np.flatnonzero(bins==b),int((bins[chosen]==b).sum()),replace=False) for b in range(16)])
                selections.append(control);indicator=torch.zeros(768,device='cuda',dtype=torch.bool);indicator[control]=True;indicator[control+384]=True
                with torch.no_grad():
                    changed=torch.where(indicator[None,None,:],source,dest)
                    controls.append((contrast(path_forward(model,x,changed,other),sy,y)-base).cpu().numpy())
            controls=np.stack(controls);array[f'control_{count}_pairs']=controls;array[f'control_{count}_selections']=np.stack(selections)
            info['random_control_mean']=float(controls.mean())
            info['selected_minus_random']=summarize(effect-controls.mean(0),np.ones(len(effect),bool),reserved_groups,rng)
            info['removal_effect']=summarize(array['full_reserved_path_effect']-removed,np.ones(len(effect),bool),reserved_groups,rng)
    archive=c.ART/'MLP_PATH_ATTRIBUTION.npz';np.savez_compressed(archive,**array)
    report=dict(role='Exploratory neuron analysis with reserved development factor groups',checkpoint_sha256=c.sha(checkpoint),
        source_sha256=c.sha(Path(__file__)),membership_sha256=c.sha(c.HERE/'MLP_PATH_MEMBERSHIP.json'),
        discovery_inputs=len(discovery_rows),reserved_inputs=len(reserved_rows),discovery_groups=len(discovery),reserved_groups=len(reserved),
        path='First-block non-candidate MLP hidden units, through the second-block keys and values only; upstream destination states are restored in the receiver replay',
        integration='Gauss-Legendre quadrature along the straight hidden-activation segment; 16, 32, then 64 points if needed',
        completeness=completeness,selection='Rank sums for duplicated neuron pairs by their signed mean attribution, with equal weight per discovery factor group',
        control='Twenty random subsets of the same size, matching 16 bins of activation-change magnitude; both duplicated neuron indices are selected together',
        top_16_neuron_pairs=[[int(i),int(i+384)] for i in ranking[:16]],reserved_results=results,artifact_sha256=c.sha(archive),
        limitation='The interpolation and replacements are interventions in the network. They need not be realizable by Schubert triples. A concentrated path does not identify the statistic computed by its neurons.')
    c.write('MLP_PATH_ATTRIBUTION.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('mlp_path_integrated_gradients',300,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
