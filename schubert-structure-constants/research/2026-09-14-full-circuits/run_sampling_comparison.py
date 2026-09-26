"""Execute the fixed matched error-sampling comparison sequentially."""
import json
import subprocess
import sys
import session as s


def main():
    plan=json.loads((s.HERE/'ERROR_SAMPLING_PLAN.json').read_text())
    assert plan['status']=='fixed_before_training'
    initial=s.PREVIOUS/'artifacts/root_action_training.pt'
    assert s.sha(initial)==plan['initial_checkpoint_sha256']
    reports={}
    for name in plan['arms']:
        command=[sys.executable,str(s.HERE/'train_selector.py'),'--arm',name,'--initial',str(initial),
            '--steps','6000','--evaluate-every','1000','--wall-limit','1500',
            '--base-lr','2e-5','--extra-lr','6e-5','--root-action-lr','5e-4',
            '--fresh','--adaptive','--screened','--exposed-assessment',
            '--error-refresh-every','1000','--mining-samples','250000',
            '--paired-views','128','--hard-supplied','mixed']
        if name=='pinned_error_control':command.append('--pinned-errors')
        with (s.HERE/(name+'.log')).open('x') as out:
            subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
        report=json.loads((s.HERE/(name+'.json')).read_text())
        assert report['status']=='completed' and report['completed_updates']==6000
        reports[name]={key:report[key] for key in ['best_criterion','best_step','best_variant','checkpoint_sha256']}
        reports[name]['report_sha256']=s.sha(s.HERE/(name+'.json'))
    winner=max(reports,key=lambda name:reports[name]['best_criterion'])
    result=dict(status='completed',winner=winner,arms=reports,plan_sha256=s.sha(s.HERE/'ERROR_SAMPLING_PLAN.json'),source_sha256=s.sha(__file__),role='Matched development comparison')
    s.write('ERROR_SAMPLING_COMPARISON.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
