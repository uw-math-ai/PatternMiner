"""Fit explicit inverse-position comparisons to attention logits and replace them."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from expanded_decomposition import bank
import richardson_model as rm
import circuit_forward as cf
from full_resources import Budget


def feature_matrix(data):
    rows,query=np.nonzero(data['mask']);x=data['x'];q=x[rows,query,3:9];k=x[rows,:,3:9]
    difference=q[:,None,:,None]-k[:,:,None,:]
    parts=[];names=[];coordinates=['u^-1(j)','u^-1(j+1)','v^-1(j)','v^-1(j+1)','w^-1(j)','w^-1(j+1)']
    for name,value in [('negative_absolute_difference',-abs(difference)),('negative_squared_difference',-difference**2),('comparison_sign',np.sign(difference)),('position_equality',(difference==0).astype(np.float32))]:
        parts.append(value.reshape(len(rows),16,36))
        names.extend([f'{name}: query {a}, key {b}' for a in coordinates for b in coordinates])
    parts.extend([k,k*k,x[rows,:,0:1],x[rows,:,0:1]**2])
    names.extend(['key '+a for a in coordinates]+['squared key '+a for a in coordinates]+['key index/n','squared key index/n'])
    parts.append(np.stack([x[rows,:,15],data['mask'][rows].astype(np.float32),x[rows,:,16]],-1));names.extend(['left descent of u at key','left descent of v at key','left descent of w at key'])
    distance=x[rows,query,0,None]-x[rows,:,0]
    parts.append(np.stack([-abs(distance),-distance**2],-1));names.extend(['negative absolute query-key index difference/n','negative squared query-key index difference/n'])
    values=np.concatenate(parts,-1).astype(np.float32);real=x[rows,:,18]!=0
    values-=((values*real[:,:,None]).sum(1)/real.sum(1)[:,None])[:,None,:]
    assert values.shape[-1]==len(names)
    return values,real,rows,query,names


def main(budget):
    comparison=json.loads((s.HERE/'ERROR_SAMPLING_COMMON_WINDOW_COMPARISON.json').read_text())
    checkpoint=s.ART/(comparison['winner']+'_common5000.pt');data,populations,_=bank();heads=[0,8]
    groups=np.unique(data['group'][~data['previously_used_neuron_orbit']]);rng=np.random.default_rng(s.SEED+870);rng.shuffle(groups)
    a=int(.6*len(groups));b=int(.8*len(groups));membership=dict(fit=groups[:a].tolist(),selection=groups[a:b].tolist(),reserved=groups[b:].tolist())
    plan=dict(status='fixed_before_logit_fitting',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),membership=membership,
        heads=heads,features='All 36 query/key inverse-position differences: absolute value, square, sign and equality; individual key positions, squares, key index, left descents, and query-key index differences. Positions are divided by n before continuous comparisons.',
        target='Attention logits in the second layer, centered across real keys for each candidate query',
        regularization=[1e-6,1e-4,1e-2,1.],sizes=[4,8,16,32,64,'all'],
        selection='Fit ridge regressions with factor-orbit weights. Rank coefficients by their standardized norm across the two heads, refit each listed size, then choose the smallest model within 5% of the minimum selection-set logit MSE.',
        role='Development approximation and functional replacement. Reserved factor orbits are excluded from fitting and tuning, but were used in previous head analysis; this is not formal confirmation.')
    assert not (s.HERE/'ATTENTION_APPROXIMATION_PLAN.json').exists();s.write('ATTENTION_APPROXIMATION_PLAN.json',plan)
    model=rm.load(checkpoint).eval();logits=[];patterns=[];native=[];predictions=[]
    with torch.inference_mode():
        for begin in range(0,len(data['x']),64):
            budget.check();x=torch.as_tensor(data['x'][begin:begin+64],device='cuda');score,cache=cf.explicit(model,x,capture=True)
            logits.append(cache['L1.attention_logits'][:,heads].cpu().numpy());patterns.append(cache['L1.pattern'][:,heads].cpu().numpy())
            mask=torch.as_tensor(data['mask'][begin:begin+64],device='cuda');y=torch.as_tensor(data['y'][begin:begin+64],device='cuda')
            pred=score.masked_fill(~mask,-1e9).argmax(1);predictions.append(pred.cpu().numpy());native.append(y[torch.arange(len(x),device='cuda'),pred].cpu().numpy())
    logits=np.concatenate(logits);patterns=np.concatenate(patterns);native=np.concatenate(native);predictions=np.concatenate(predictions)
    F,real,rows,query,names=feature_matrix(data)
    target=logits[rows,:,query,:].transpose(0,2,1)
    target-=((target*real[:,:,None]).sum(1)/real.sum(1)[:,None])[:,None,:]
    X=F[real].astype(np.float64);Y=target[real].astype(np.float64)
    edge_groups=np.broadcast_to(data['group'][rows,None],real.shape)[real]
    unique,counts=np.unique(edge_groups,return_counts=True);number=dict(zip(unique,counts));weight=np.array([1/number[g] for g in edge_groups])
    selections={key:np.isin(edge_groups,value) for key,value in membership.items()}
    fit=selections['fit'];ww=weight[fit];xx=X[fit];yy=Y[fit];total=ww.sum()
    scale=np.sqrt((xx**2*ww[:,None]).sum(0)/total);scale[scale<1e-8]=1.
    xx/=scale
    G=xx.T@(xx*ww[:,None])/total;D=xx.T@(yy*ww[:,None])/total
    def mse(coef,part):
        take=selections[part];error=Y[take]-X[take]@coef
        return float((error**2*weight[take,None]).sum()/(2*weight[take].sum()))
    def variance(part):
        take=selections[part];return float((Y[take]**2*weight[take,None]).sum()/(2*weight[take].sum()))
    fits=[]
    for ridge in plan['regularization']:
        budget.check();dense=np.linalg.solve(G+ridge*np.eye(len(G)),D);ranking=np.argsort(-np.linalg.norm(dense,axis=1),kind='stable')
        for size in plan['sizes']:
            selected=ranking if size=='all' else ranking[:size]
            coef=np.zeros_like(D);coef[selected]=np.linalg.solve(G[np.ix_(selected,selected)]+ridge*np.eye(len(selected)),D[selected]);coef/=scale[:,None]
            fits.append(dict(ridge=ridge,size=len(selected),selected=selected,coefficients=coef,selection_mse=mse(coef,'selection')))
    best=min(f['selection_mse'] for f in fits)
    chosen=min((f for f in fits if f['selection_mse']<=1.05*best),key=lambda f:(f['size'],f['selection_mse']))
    coef=chosen['coefficients'];fit_metrics={part:dict(mse=mse(coef,part),centered_logit_r_squared=1-mse(coef,part)/variance(part)) for part in membership}
    predicted=(F@coef).astype(np.float32);replacement=np.zeros_like(logits);replacement[rows,:,query,:]=predicted.transpose(0,2,1)
    modified={key:[] for key in ['fitted','uniform']};agreement={key:[] for key in modified};checks=dict(identity=0.)
    with torch.inference_mode():
        for begin in range(0,len(data['x']),64):
            budget.check();end=min(begin+64,len(data['x']));x=torch.as_tensor(data['x'][begin:end],device='cuda')
            score,cache=cf.explicit(model,x,capture=True);mask=torch.as_tensor(data['mask'][begin:end],device='cuda');y=torch.as_tensor(data['y'][begin:end],device='cuda')
            identity=cf.explicit(model,x,patches={'L1.pattern':cache['L1.pattern']});checks['identity']=max(checks['identity'],float((identity-score).abs().max()))
            valid=x[:,:,18]!=0
            for kind in modified:
                raw=torch.as_tensor(replacement[begin:end],device='cuda') if kind=='fitted' else torch.zeros((len(x),2,16,16),device='cuda')
                pattern=raw.masked_fill(~valid[:,None,None,:],-torch.inf).softmax(-1)
                patch=cache['L1.pattern'].clone();patch[:,heads]=torch.where(mask[:,None,:,None],pattern,patch[:,heads])
                output=cf.explicit(model,x,patches={'L1.pattern':patch});pred=output.masked_fill(~mask,-1e9).argmax(1)
                modified[kind].append(y[torch.arange(len(x),device='cuda'),pred].cpu().numpy());agreement[kind].append(pred.cpu().numpy()==predictions[begin:end])
    assert checks['identity']==0
    arrays=dict(logits=logits,patterns=patterns,coefficients=coef,scale=scale,selected=chosen['selected'],native_correct=native,group=data['group'],population=data['population'])
    arrays.update({key+'_correct':np.concatenate(value) for key,value in modified.items()});arrays.update({key+'_agreement':np.concatenate(value) for key,value in agreement.items()})
    behavior={}
    for part,subset in [('all',np.ones(len(native),bool))]+[(k,np.isin(data['group'],v)) for k,v in membership.items()]:
        behavior[part]=dict(inputs=int(subset.sum()),native_pair_accuracy=float((native[::2]&native[1::2])[subset[::2]].mean()))
        for kind in modified:
            good=arrays[kind+'_correct'];behavior[part][kind]=dict(pair_accuracy=float((good[::2]&good[1::2])[subset[::2]].mean()),choice_agreement=float(arrays[kind+'_agreement'][subset].mean()),lost=int((native&~good&subset).sum()),fixed=int((~native&good&subset).sum()))
    archive=s.ART/'ATTENTION_APPROXIMATION.npz';np.savez_compressed(archive,**arrays)
    report=dict(status='completed',plan_sha256=s.sha(s.HERE/'ATTENTION_APPROXIMATION_PLAN.json'),checks=checks,
        fits=[{k:v for k,v in f.items() if k not in ['coefficients','selected']} for f in fits],
        selected_ridge=chosen['ridge'],selected_features=[dict(name=names[i],coefficients=coef[i].tolist()) for i in chosen['selected']],
        fit_metrics=fit_metrics,behavior=behavior,artifact_sha256=s.sha(archive),
        limitation='A fitted approximation is not by itself a circuit explanation. Its replacement behavior and reserved-factor errors are part of the result. Neither a general multiplication rule nor independent confirmation is claimed.')
    s.write('ATTENTION_APPROXIMATION.json',report);print(json.dumps(dict(status='completed',features=chosen['size'],fit_metrics=fit_metrics,behavior=behavior)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('attention_approximation',600,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
