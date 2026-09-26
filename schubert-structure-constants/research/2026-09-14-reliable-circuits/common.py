"""Read frozen models and data while keeping this campaign's records separate."""
from pathlib import Path
import sys
from budget import Budget

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'2026-09-13-circuit-readiness'
sys.path.insert(0, str(PREVIOUS))
import campaign as c
import symmetry
import circuits
sys.path.insert(0, str(HERE))

ART = HERE/'artifacts'
INITIAL = PREVIOUS/'artifacts/deeper_attention.pt'
VALIDATION = ['S7_validation', 'S8_validation', 'fresh_S9_validation', 'fresh_S10_validation']
POOLS = ['S7_train', 'S8_train', 'fresh_train']
SEED = 2026091401


def model_from(path,**kwargs):
    from models import model_from as load_model
    return load_model(path,**kwargs)


def write(name, data):
    c.write(HERE/name, data)
