"""Retain a common evaluation window before either timed trainer reaches it."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import session as s


def snapshot(name):
    target=s.HERE/(name+'_common5000.json')
    assert not target.exists()
    while True:
        path=s.HERE/(name+'.json')
        if path.exists():
            report=json.loads(path.read_text())
            if any(item['step']==5000 for item in report['history']):break
            assert report['status']=='running', report['status']
        time.sleep(.5)
    assert max(item['step'] for item in report['history'])==5000
    assert report['best_step']<=5000
    source=s.ART/(name+'.pt');before=s.sha(source)
    saved=torch.load(source,map_location='cpu',weights_only=True)
    assert saved['step']==report['best_step'] and s.sha(source)==before
    output=s.ART/(name+'_common5000.pt');shutil.copyfile(source,output)
    assert s.sha(output)==before
    report.update(status='completed_common_window',checkpoint_sha256=before,parent_arm=name,
        selected_window_updates=5000,role='Completed scheduled evaluations from initialization through update 5000; parent training may continue under its timed limit')
    report['configuration']=dict(report['configuration'],arm=name+'_common5000')
    s.write(target.name,report)
    return report


def main():
    assert not (s.HERE/'ERROR_SAMPLING_WINDOW_PLAN.json').exists()
    current=json.loads((s.HERE/'pinned_error_control.json').read_text())
    assert max([0]+[item['step'] for item in current['history']])<5000
    s.write('ERROR_SAMPLING_WINDOW_PLAN.json',dict(status='fixed_before_window_completion',updates=5000,
        parent_plan_sha256=s.sha(s.HERE/'ERROR_SAMPLING_PLAN.json'),source_sha256=s.sha(Path(__file__)),
        reason='Measured evaluation and mining cost projects the 6000-update schedules beyond their conservative timed stopping point. Compare both arms on the identical first 5000 updates and preserve those checkpoints before further updates.',
        control_completed_evaluations_when_fixed=[item['step'] for item in current['history']],
        comparison='Same six scheduled evaluations including initialization, same 6000-step learning-rate schedule restricted to the first 5000 updates; later updates are outside this comparison'))
    control=snapshot('pinned_error_control')
    while True:
        report=json.loads((s.HERE/'pinned_error_control.json').read_text())
        if report['status']!='running':break
        time.sleep(1)
    process=None;stream=None
    if report['status']=='intentional_time_stop':
        # The original runner requires all 6000 updates and exits at this point.
        # The common-window plan permits continuation of the second arm.
        assert not (s.HERE/'refreshed_errors.log').exists()
        command=[sys.executable,str(s.HERE/'train_selector.py'),'--arm','refreshed_errors',
            '--initial',str(s.PREVIOUS/'artifacts/root_action_training.pt'),'--steps','6000',
            '--evaluate-every','1000','--wall-limit','1500','--base-lr','2e-5','--extra-lr','6e-5',
            '--root-action-lr','5e-4','--fresh','--adaptive','--screened','--exposed-assessment',
            '--error-refresh-every','1000','--mining-samples','250000','--paired-views','128','--hard-supplied','mixed']
        stream=(s.HERE/'refreshed_errors.log').open('x')
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
    else:assert report['status']=='completed'
    treatment=snapshot('refreshed_errors')
    if process is not None:
        assert process.wait()==0;stream.close()
    else:
        while json.loads((s.HERE/'refreshed_errors.json').read_text())['status']=='running':time.sleep(1)
    reports={name:{key:report[key] for key in ['best_criterion','best_step','best_variant','checkpoint_sha256']} for name,report in [('pinned_error_control',control),('refreshed_errors',treatment)]}
    winner=max(reports,key=lambda name:reports[name]['best_criterion'])
    result=dict(status='completed',winner=winner,arms=reports,updates=5000,
        plan_sha256=s.sha(s.HERE/'ERROR_SAMPLING_WINDOW_PLAN.json'),
        role='Matched comparison on the common predeclared 5000-update window')
    s.write('ERROR_SAMPLING_COMMON_WINDOW_COMPARISON.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
