"""Distinguish changed valid witnesses from new errors in the pruning study."""
import json
from pathlib import Path
import numpy as np
import session as s
from full_resources import Budget


def main():
    strict=json.loads((s.HERE/'HEAD_PRUNING.json').read_text())
    task=json.loads((s.HERE/'TASK_HEAD_PRUNING.json').read_text())
    native=json.loads((s.HERE/'TASK_HEAD_PRUNING_NATIVE_VALIDATION.json').read_text())
    assert strict['status']==task['status']==native['status']=='completed'
    populations={}
    with np.load(s.ART/'TASK_HEAD_PRUNING.npz') as archive:
        for name in ['development','replay']:
            before=archive[name+'_reference'];after=archive[name+'_pruned'];y=archive[name+'_y'];rr=np.arange(len(y))
            good_before=y[rr,before];good_after=y[rr,after];changed=before!=after
            pair_before=good_before[::2]&good_before[1::2];pair_after=good_after[::2]&good_after[1::2]
            populations[name]=dict(inputs=len(y),changed_choices=int(changed.sum()),
                changed_choices_with_both_indices_valid=int((changed&good_before&good_after).sum()),
                changed_choices_from_valid_to_invalid=int((changed&good_before&~good_after).sum()),
                changed_choices_from_invalid_to_valid=int((changed&~good_before&good_after).sum()),
                pairs_fixed=int((~pair_before&pair_after).sum()),pairs_lost=int((pair_before&~pair_after).sum()))
    result=dict(status='completed',checkpoint_sha256=task['checkpoint_sha256'],strict_retained_heads=strict['retained_heads'],
        single_pair_ablations_with_at_least_99_percent_task_accuracy=sum(item['metrics']['pair_accuracy']>=.99 for item in strict['history']),
        task_retained_heads=task['retained_heads'],task_development=task['development'],task_exposed_replay=task['exposed_replay'],
        all_27_development_criterion=native['criterion'],choice_changes=populations,
        source_sha256=s.sha(Path(__file__)),input_sha256={name:s.sha(s.HERE/(name+'.json')) for name in ['HEAD_PRUNING','TASK_HEAD_PRUNING','TASK_HEAD_PRUNING_NATIVE_VALIDATION']},
        conclusion='Exact choice fidelity and validity of a witness are different objectives. The second mask preserves high development task accuracy but worsens exposed S10 accuracy. Neither study establishes a minimal circuit or independent qualification.')
    assert not (s.HERE/'PRUNING_SUMMARY.json').exists();s.write('PRUNING_SUMMARY.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    budget=Budget('pruning_summary',60);status='failed_or_partial'
    try:main();status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
