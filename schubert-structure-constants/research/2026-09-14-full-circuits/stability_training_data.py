"""Keep the existing training corpus and append the exposed assessment for development."""
import neighbor_training_data as base
from second_training_data import validation
screened_offset=None

def load(fresh=False,adaptive=False,screened=False):
    global screened_offset
    result=base.load(fresh,adaptive,screened)
    screened_offset=base.screened_offset
    return result
