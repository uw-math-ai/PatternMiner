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
sys.path.insert(0,str(HERE))
ART=HERE/'artifacts'
sha=c.sha
SEED=2026091451
def write(name,value):c.prior.c.write(HERE/name,value)
