"""Execute the registered circuit study only after independent qualification."""
import json
from pathlib import Path
import subprocess
import sys
import time
import session as s


def run(script,arguments):
    with (s.HERE/(script.removesuffix('.py')+'_qualified.log')).open('x') as stream:
        process=subprocess.run([sys.executable,str(s.HERE/script),*arguments],stdout=stream,stderr=subprocess.STDOUT)
    assert process.returncode==0,(script,process.returncode)


def main():
    assessment=s.HERE/'SECOND_ASSESSMENT.json'
    while True:
        if assessment.exists():
            report=json.loads(assessment.read_text())
            active=json.loads((s.HERE/'RESOURCE_RESERVATIONS.json').read_text())
            if report['status']=='completed' and 'second_independent_assessment' not in active:break
            if (s.HERE/'SECOND_ASSESSMENT_EXPOSURE.json').exists() and 'second_independent_assessment' not in active:
                s.write('SECOND_FORMAL_STUDY_STATUS.json',dict(status='not_started',reason='Independent assessment was interrupted or failed.'))
                return
        time.sleep(2)
    if not report['passes']:
        s.write('SECOND_FORMAL_STUDY_STATUS.json',dict(status='not_started',reason='Independent accuracy requirement was not met; discovery and confirmation remain protected.',assessment_sha256=s.sha(assessment)))
        print('Independent assessment failed; formal populations remain protected.',flush=True);return
    selection=json.loads((s.HERE/'SECOND_MODEL_SELECTION.json').read_text())
    checkpoint=s.HERE/selection['checkpoint'];digest=s.sha(checkpoint)
    assert digest==selection['checkpoint_sha256']==report['qualification']['checkpoint_sha256']
    shared=['--checkpoint',str(checkpoint),'--assessment',str(assessment)]
    s.write('SECOND_FORMAL_STUDY_STATUS.json',dict(status='running',checkpoint_sha256=digest,plan_sha256=s.sha(s.HERE/'FORMAL_CONTINUATION_PLAN.md')))
    run('check_formal_interventions.py',['--checkpoint',str(checkpoint),'--name','QUALIFIED_FORMAL_INTERFACE_CHECKS'])
    run('formal_scan.py',shared+['--name','FORMAL_DISCOVERY','--part','discovery','--pairs','512'])
    run('formal_confirm.py',shared+['--name','FORMAL_CONFIRMATION','--discovery',str(s.HERE/'FORMAL_DISCOVERY.json'),'--heads','4','--pairs','512'])
    run('formal_path.py',shared+['--name','FORMAL_MLP_PATH'])
    assert s.sha(checkpoint)==digest
    s.write('SECOND_FORMAL_STUDY_STATUS.json',dict(status='completed',checkpoint_sha256=digest,plan_sha256=s.sha(s.HERE/'FORMAL_CONTINUATION_PLAN.md'),assessment_sha256=s.sha(assessment),source_sha256=s.sha(Path(__file__))))
    print('The registered formal study completed.',flush=True)


if __name__=='__main__':main()
