"""Compare fixed linear combinations of the selector and its coefficient predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import richardson_model as rm
import second_training_data as td
from exposed_assessment import append_validation
from full_resources import Budget


def main(args,budget):
    checkpoint=Path(args.checkpoint);model=rm.load(checkpoint).eval()
    assert model.supplied_index_independent
    populations=append_validation(td.validation(True));assert len(populations)==27
    grid=[(b,q) for b in [0.,2.,4.,8.] for q in [0.,1.,2.,4.]]
    plan=dict(status='fixed_before_predictions',checkpoint_sha256=s.sha(checkpoint),source_sha256=s.sha(Path(__file__)),
        grid=grid,readout='Selector score + b times predicted log(1+B_j) + q times predicted log(1+Q_j)',
        selection='Minimum of twenty-seven development accuracies, then their mean; initialization is included at lambda=0',
        role='Development readout comparison; no exact labels are inputs and no assessment is opened',
        all_supplied='The model structurally discards the supplied index; each oriented triple has the same prediction for every positive supplied coefficient.')
    assert not (s.HERE/(args.name+'_PLAN.json')).exists();s.write(args.name+'_PLAN.json',plan)
    captured={};hook=model.head.register_forward_pre_hook(lambda module,inputs:captured.update(hidden=inputs[0]))
    metrics={str(k):{} for k in grid};artifacts={}
    with torch.inference_mode():
        for name,data in populations.items():
            scores=[];auxiliary=[]
            for begin in range(0,len(data['x']),512):
                budget.check();x=torch.as_tensor(data['x'][begin:begin+512],device='cuda')
                scores.append(model(x).cpu().numpy());auxiliary.append(model.auxiliary_head(captured.pop('hidden'))[:,:,1:].cpu().numpy())
            score=np.concatenate(scores);aux=np.concatenate(auxiliary);good=[];predictions=[]
            for k in grid:
                pred=np.where(data['mask'],score+k[0]*aux[:,:,0]+k[1]*aux[:,:,1],-1e9).argmax(1)
                correct=data['y'][np.arange(len(pred)),pred];good.append(correct);predictions.append(pred)
                criterion=(correct[::2]&correct[1::2]).mean() if name=='hard_validation' or name.endswith('_pairs') else correct.mean()
                metrics[str(k)][name]=float(criterion)
            path=s.ART/f'{args.name}_{name}.npz'
            np.savez_compressed(path,score=score,auxiliary_B_Q=aux,good=np.stack(good),predictions=np.stack(predictions).astype(np.int8))
            digest=hashlib.sha256()
            for key in ['x','mask','y','a']:digest.update(data[key].tobytes())
            artifacts[name]=dict(sha256=s.sha(path),development_tensor_sha256=digest.hexdigest(),oriented_examples=len(score))
    hook.remove()
    criteria={key:[min(v.values()),float(np.mean(list(v.values())))] for key,v in metrics.items()}
    winner=max(criteria,key=lambda key:criteria[key]);coefficient=next(k for k in grid if str(k)==winner)
    output=s.ART/(args.name+'.pt')
    with torch.no_grad():
        model.head.weight.add_(model.auxiliary_head.weight[1:2],alpha=coefficient[0])
        model.head.bias.add_(model.auxiliary_head.bias[1:2],alpha=coefficient[0])
        model.head.weight.add_(model.auxiliary_head.weight[2:3],alpha=coefficient[1])
        model.head.bias.add_(model.auxiliary_head.bias[2:3],alpha=coefficient[1])
    rm.save(model,output,step=0,criterion=criteria[winner],readout_coefficient=coefficient,initial_sha256=s.sha(checkpoint))
    report=dict(**{k:v for k,v in plan.items() if k!='status'},status='completed',winner=coefficient,
        criteria=criteria,populations=metrics,artifacts=artifacts,selected_checkpoint_sha256=s.sha(output),
        limitation='Finite development selection of a linear readout, not independent evidence. The saved merged head must be evaluated natively before qualification; rounding can differ from adding cached scores.')
    s.write(args.name+'.json',report);print(json.dumps(dict(status='completed',winner=coefficient,criteria=criteria)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True);args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget(args.name.lower(),300,gpu=True);status='failed_or_partial'
    try:main(args,budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
