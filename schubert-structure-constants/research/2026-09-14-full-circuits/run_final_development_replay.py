"""Replay the fixed development interventions on the final selected network."""
import json
import time
import session as s
from run_second_continuation import run


def main():
    plan=json.loads((s.HERE/'FINAL_DEVELOPMENT_REPLAY_PLAN.json').read_text())
    while True:
        path=s.HERE/'SECOND_CONTINUATION_STATUS.json'
        if path.exists() and json.loads(path.read_text())['status'] in ['completed','development_requirement_failed']:
            break
        time.sleep(2)
    checkpoint=s.HERE/plan['checkpoint'];digest=s.sha(checkpoint)
    for study in plan['studies']:
        run(study['script'].removesuffix('.py'),['--checkpoint',str(checkpoint),'--name',study['name']])
        report=json.loads((s.HERE/(study['name']+'.json')).read_text())
        assert report['status']=='completed' and report['checkpoint_sha256']==digest
    assert s.sha(checkpoint)==digest
    s.write('FINAL_DEVELOPMENT_REPLAY_STATUS.json',dict(status='completed',checkpoint_sha256=digest,
        plan_sha256=s.sha(s.HERE/'FINAL_DEVELOPMENT_REPLAY_PLAN.json')))


if __name__=='__main__':main()
