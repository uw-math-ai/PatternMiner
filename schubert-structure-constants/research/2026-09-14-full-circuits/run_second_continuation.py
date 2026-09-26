"""Carry out the registered second continuation and its one-shot assessment."""
import json,subprocess,sys,time
import session as s


def run(name,arguments=()):
    with (s.HERE/(name+'_run.log')).open('x') as stream:
        result=subprocess.run([sys.executable,str(s.HERE/(name+'.py')),*arguments],stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,name


def main():
    while True:
        path=s.HERE/'teacher_stability.json';prepared=s.HERE/'SECOND_NEIGHBOR_PREPARED.json'
        active=json.loads((s.HERE/'RESOURCE_RESERVATIONS.json').read_text())
        if path.exists() and prepared.exists():
            training=json.loads(path.read_text())
            if training['status'] in ['completed','intentional_time_stop'] and 'teacher_stability_training' not in active:break
        time.sleep(2)
    run('audit_second_training')
    arguments=['--arm','second_neighbor_refinement','--initial',str(s.ART/'teacher_stability.pt'),
        '--teacher',str(s.ART/'pinned_error_control_common5000.pt'),'--teacher-weight','.2',
        '--steps','9000','--evaluate-every','1500','--wall-limit','3300',
        '--base-lr','5e-6','--extra-lr','1.5e-5','--root-action-lr','1e-4',
        '--fresh','--adaptive','--screened','--exposed-assessment',
        '--error-refresh-every','3000','--mining-samples','350000',
        '--paired-views','128','--hard-supplied','mixed','--pinned-errors']
    run('train_second_continuation',arguments)
    run('compare_second_readouts',['--checkpoint',str(s.ART/'second_neighbor_refinement.pt'),'--name','SECOND_REFINEMENT_READOUTS'])
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
        policy='Final training and registered readout selection on twenty-seven development criteria, before opening the replacement assessment.')
    with (s.HERE/'SECOND_MODEL_SELECTION.json').open('x') as stream:json.dump(selection,stream,indent=2)
    run('assess_second',['--training-report',str(s.HERE/'SECOND_NATIVE_VALIDATION.json')])
    run('run_second_formal_study')
    assessment=json.loads((s.HERE/'SECOND_ASSESSMENT.json').read_text())
    s.write('SECOND_CONTINUATION_STATUS.json',dict(status='completed',independent_assessment_passes=assessment['passes'],checkpoint_sha256=s.sha(checkpoint)))
    print('Second continuation and its conditional formal study completed.',flush=True)


if __name__=='__main__':main()
