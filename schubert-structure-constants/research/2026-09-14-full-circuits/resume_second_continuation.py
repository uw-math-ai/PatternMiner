"""Finish registered selection after the exact label-storage correction."""
import json,time
import session as s
from run_second_continuation import run


def main():
    while True:
        path=s.HERE/'second_neighbor_refinement_compact.json'
        active=json.loads((s.HERE/'RESOURCE_RESERVATIONS.json').read_text())
        if path.exists():
            training=json.loads(path.read_text())
            if training['status'] in ['completed','intentional_time_stop'] and 'second_neighbor_refinement_compact_training' not in active:break
        time.sleep(2)
    run('select_second_interpolation')
    run('compare_second_readouts',['--checkpoint',str(s.ART/'SECOND_INTERPOLATION_SELECTED.pt'),'--name','SECOND_REFINEMENT_READOUTS'])
    checkpoint=s.ART/'SECOND_REFINEMENT_READOUTS.pt'
    run('evaluate_second_checkpoint',['--checkpoint',str(checkpoint),'--name','SECOND_NATIVE_VALIDATION'])
    report=json.loads((s.HERE/'SECOND_NATIVE_VALIDATION.json').read_text())
    if report['best_criterion'][0]<.99:
        s.write('SECOND_CONTINUATION_STATUS.json',dict(status='development_requirement_failed',criterion=report['best_criterion'],assessment_opened=False))
        print('Second continuation failed development qualification; assessment remains protected.',flush=True);return
    readout=json.loads((s.HERE/'SECOND_REFINEMENT_READOUTS.json').read_text())
    selection=dict(status='selected',checkpoint='artifacts/SECOND_REFINEMENT_READOUTS.pt',checkpoint_sha256=s.sha(checkpoint),
        criterion=report['best_criterion'],native_validation_sha256=s.sha(s.HERE/'SECOND_NATIVE_VALIDATION.json'),
        readout=readout['winner'],plan_sha256=s.sha(s.HERE/'SECOND_CONTINUATION_PLAN.json'),
        storage_correction_sha256=s.sha(s.HERE/'SECOND_MEMORY_CORRECTION.json'),
        interpolation_sha256=s.sha(s.HERE/'SECOND_INTERPOLATION.json'),
        interpolation_plan_sha256=s.sha(s.HERE/'SECOND_INTERPOLATION_PLAN.json'),
        policy='Final training, registered parameter interpolation and readout selection on twenty-seven development criteria, before opening the replacement assessment.')
    with (s.HERE/'SECOND_MODEL_SELECTION.json').open('x') as stream:json.dump(selection,stream,indent=2)
    run('assess_second',['--training-report',str(s.HERE/'SECOND_NATIVE_VALIDATION.json')])
    run('run_second_formal_study')
    assessment=json.loads((s.HERE/'SECOND_ASSESSMENT.json').read_text())
    s.write('SECOND_CONTINUATION_STATUS.json',dict(status='completed',independent_assessment_passes=assessment['passes'],checkpoint_sha256=s.sha(checkpoint)))
    print('Second continuation and its conditional formal study completed.',flush=True)


if __name__=='__main__':main()
