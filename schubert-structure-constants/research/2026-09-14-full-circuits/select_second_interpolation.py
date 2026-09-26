"""Compare the five registered single-network parameter interpolations."""
import json
import shutil
from pathlib import Path
import torch
import session as s
import richardson_model as rm
import second_training_data as td
from exposed_assessment import append_validation
from train_neighbors import evaluate
from full_resources import Budget


def main(budget):
    plan_path=s.HERE/'SECOND_INTERPOLATION_PLAN.json'
    plan=json.loads(plan_path.read_text())
    assert not (s.HERE/'SECOND_ASSESSMENT_STARTED.json').exists()
    assert not (s.HERE/'SECOND_INTERPOLATION.json').exists()
    early=s.HERE/plan['early_checkpoint'];final=s.HERE/plan['final_checkpoint']
    assert s.sha(early)==plan['early_sha256']
    hashes=dict(early=s.sha(early),final=s.sha(final))
    a=torch.load(early,map_location='cpu',weights_only=True)['state_dict']
    b=torch.load(final,map_location='cpu',weights_only=True)['state_dict']
    assert a.keys()==b.keys()
    assert all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a)
    identical=all(torch.equal(a[k],b[k]) for k in a)
    weights=[0] if identical else plan['early_weights']
    model=rm.load(final).eval();data=append_validation(td.validation(True));assert len(data)==27
    candidates=[];best=None;winner=None
    for weight in weights:
        budget.check()
        state={}
        for k in a:
            if a[k].is_floating_point():
                state[k]=b[k] if weight==0 else a[k] if weight==1 else a[k]*weight+b[k]*(1-weight)
            else:
                assert torch.equal(a[k],b[k]);state[k]=b[k]
        model.load_state_dict(state)
        metrics,criterion=evaluate(model,data)
        path=s.ART/f'second_interpolation_{round(weight*100):03d}.pt'
        assert not path.exists()
        rm.save(model,path,step=0,criterion=criterion,early_weight=weight,endpoint_sha256=hashes,
                plan_sha256=s.sha(plan_path),selection_role='27 development criteria')
        candidate=dict(early_weight=weight,criterion=criterion,validation=metrics,
                       checkpoint=str(path.relative_to(s.HERE)),checkpoint_sha256=s.sha(path))
        candidates.append(candidate)
        if best is None or criterion>best:best=criterion;winner=candidate
        print(json.dumps(dict(early_weight=weight,criterion=criterion)),flush=True)
    output=s.ART/'SECOND_INTERPOLATION_SELECTED.pt';assert not output.exists()
    shutil.copy2(s.HERE/winner['checkpoint'],output)
    assert s.sha(early)==hashes['early'] and s.sha(final)==hashes['final']
    s.write('SECOND_INTERPOLATION.json',dict(status='completed',endpoint_sha256=hashes,
        endpoints_identical=identical,candidates=candidates,winner=winner,
        selected_checkpoint=str(output.relative_to(s.HERE)),selected_sha256=s.sha(output),
        plan_sha256=s.sha(plan_path),source_sha256=s.sha(Path(__file__))))


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('second_parameter_interpolation',600,gpu=True);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
