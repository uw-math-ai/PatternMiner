"""One protected assessment, after all nineteen development populations pass."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import compact
import richardson_model as rm
import mine_all_certificates
from assess import evaluate
from full_resources import Budget


def gate(report_path):
    training=json.loads(report_path.read_text())
    assert training['status'] in ['completed','intentional_time_stop','qualified_checkpoint_frozen']
    checkpoint=s.ART/(training['configuration']['arm']+'.pt')
    assert c.sha(checkpoint)==training['checkpoint_sha256']
    step=training['best_step']
    values=training['initial_validation'] if step==0 else next(x['validation'] for x in training['history'] if x['step']==step)
    assert len(values)==19 and all(v['selection_accuracy']>=.99 for v in values.values()), 'Development accuracy requirement is not met; assessment remains protected'
    return checkpoint,dict(training_report_sha256=c.sha(report_path),selected_step=step,
                           checkpoint_sha256=c.sha(checkpoint),development=values)


def init_worker():torch.set_num_threads(1)


def main(checkpoint,qualification,budget):
    output=s.HERE/'FRESH_ASSESSMENT.json';started=s.HERE/'FRESH_ASSESSMENT_STARTED.json'
    assert not output.exists() and not started.exists()
    metadata=json.loads((s.HERE/'DATA.json').read_text());assert metadata['status']=='completed'
    report=dict(status='running',role='independent_assessment',requirement=.99,qualification=qualification,
        source_sha256=c.sha(Path(__file__)),evaluation_source_sha256=c.sha(Path(evaluate.__code__.co_filename)),
        mining_source_sha256=c.sha(Path(mine_all_certificates.__file__)),data_metadata_sha256=c.sha(s.HERE/'DATA.json'),
        plan_sha256=c.sha(s.HERE/'DATA_PLAN.json'),populations={},
        difficult_pair_policy='Mine every positive supplied index; deduplicate by the two record indices and orientation; require both records correct at every positive supplied index')
    with started.open('x') as stream:json.dump(report,stream,indent=2)
    s.write(output.name,report)
    model=rm.load(checkpoint).eval()
    with ProcessPoolExecutor(max_workers=2,initializer=init_worker) as pool:
        for n in [7,8,9,10]:
            name=f'S{n}_assessment';spec=metadata['populations'][name]
            path=s.HERE/f'cache/data/{name}.bin';assert c.sha(path)==spec['sha256']
            records=compact.load(path);assert len(records)==spec['rows']
            count,errors,choices=evaluate(model,records,budget)
            jobs=[(str(path),spec['offsets'][begin:begin+33],0) for begin in range(0,len(spec['offsets'])-1,32)]
            pieces=[]
            for pairs,_ in pool.map(mine_all_certificates.worker,jobs):budget.check();pieces.append(pairs)
            pairs=np.concatenate(pieces)
            if len(pairs):
                _,keep=np.unique(pairs[:,:3],axis=0,return_index=True);pairs=pairs[np.sort(keep)]
                left,right,reverse=pairs[:,:3].T
                pair_good=(errors[2*left+reverse]==0)&(errors[2*right+reverse]==0)
            else:pair_good=np.empty(0,bool)
            group=np.repeat(records['group'],2);gc=np.bincount(group,weights=count);ge=np.bincount(group,weights=errors);active=gc>0
            metrics=dict(triples=len(records),oriented_triples=len(count),supplied_inputs=int(count.sum()),
                supplied_errors=int(errors.sum()),supplied_accuracy=float(1-errors.sum()/count.sum()),
                all_supplied_correct_accuracy=float((errors==0).mean()),
                factor_group_mean=float((1-ge[active]/gc[active]).mean()),difficult_pairs=len(pairs),
                difficult_pairs_correct=int(pair_good.sum()),
                difficult_pair_all_supplied_accuracy=float(pair_good.mean()) if len(pairs) else None)
            scores=[metrics['supplied_accuracy'],metrics['all_supplied_correct_accuracy']]
            if len(pairs):scores.append(metrics['difficult_pair_all_supplied_accuracy'])
            metrics['passes']=all(s>=.99 for s in scores)
            archive=s.ART/f'fresh_assessment_{name}.npz'
            np.savez_compressed(archive,count=count,errors=errors,choices=choices,pairs=pairs,pair_good=pair_good)
            metrics['artifact_sha256']=c.sha(archive);report['populations'][name]=metrics
            s.write(output.name,report);print(json.dumps(dict(population=name,**metrics)),flush=True)
    report.update(status='completed',passes=all(m['passes'] for m in report['populations'].values()))
    s.write(output.name,report)
    if not report['passes']:
        s.write('FRESH_ASSESSMENT_EXPOSURE.json',dict(role_after_failure='development',assessment_sha256=c.sha(output),
            reason='First independent assessment failed; any subsequent adaptation must not reuse these populations as independent evidence'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-report',required=True);args=parser.parse_args()
    if (s.HERE/'READOUT_SELECTION_PLAN.json').exists() and not (s.HERE/'FINAL_MODEL_SELECTION.json').exists():
        s.write('ASSESSMENT_DEFERRED.json',dict(status='deferred_before_predictions',reason='Complete the registered readout comparisons and final model selection before opening assessment.'))
        raise SystemExit(0)
    checkpoint,qualification=gate(Path(args.training_report))
    selection=json.loads((s.HERE/'FINAL_MODEL_SELECTION.json').read_text())
    assert selection['status']=='selected' and selection['checkpoint_sha256']==s.sha(checkpoint)
    torch.set_num_threads(1);budget=Budget('new_independent_assessment',1800,gpu=True);status='failed_or_partial'
    try:main(checkpoint,qualification,budget);status='completed'
    finally:
        if status!='completed' and (s.HERE/'FRESH_ASSESSMENT_STARTED.json').exists():
            s.write('FRESH_ASSESSMENT_EXPOSURE.json',dict(role_after_failure='development',reason='Interrupted assessment; preserve all results and do not restart as a fresh assessment'))
        print(json.dumps(budget.finish(status)),flush=True)
