"""Read frozen helpers without redirecting their files or ledgers."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-09-14-richardson-circuits'
sys.path.insert(0,str(PREVIOUS))
import context as c
import richardson_model as rm
import final_training_data
import exposed_assessment
FULL=HERE.parent/'2026-09-14-full-circuits'
sys.path.insert(0,str(FULL))
sys.path.insert(0,str(HERE))
ART=HERE/'artifacts'
sha=c.sha
SEED=2026091451
def write(name,value):c.prior.c.write(HERE/name,value)

CHECKPOINT=FULL/'artifacts/SECOND_REFINEMENT_READOUTS.pt'
ASSESSMENT=FULL/'SECOND_ASSESSMENT.json'
def verify_checkpoint(path, assessment):
    import json
    report=json.loads(Path(assessment).read_text())
    assert report['status']=='completed'
    assert report['qualification']['checkpoint_sha256']==sha(Path(path))
    assert json.loads((HERE/'PROTOCOL_AMENDMENT.json').read_text())['proceed_with_circuit_analysis']
