"""Read the frozen predecessor without redirecting its paths or resource ledger."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'2026-09-14-reliable-circuits'
sys.path.insert(0,str(OLD))
import common as prior
import compact
import train_expanded as previous_training
import generate as previous_generation
sys.path.insert(0,str(HERE))
ART=HERE/'artifacts'
SEED=2026091421
sha=prior.c.a.sha

def write(name,data):prior.c.write(HERE/name,data)
