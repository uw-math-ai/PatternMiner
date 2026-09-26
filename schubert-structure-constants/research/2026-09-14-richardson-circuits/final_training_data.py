"""Append the last disjoint neighborhoods only after the matched comparisons."""
import json
import context as c
import training_data as base
import adaptive_training

screened_offset=None


def load(fresh=False,adaptive=False,screened=False):
    global screened_offset
    assert fresh and adaptive and screened
    values=base.load(fresh,adaptive,screened)
    data,eligible,offsets,weights,roots,layouts,hard,hard_layout,fresh_pairs,fresh_layout=values
    assert len(offsets)==12
    screened_offset=offsets[-2];old_groups=len(fresh_layout['new_weights'])
    data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout=adaptive_training.append(
        data,eligible,offsets,weights,roots,layouts,fresh_pairs,fresh_layout,prefix='late')
    # The generic append uses .2 of ordinary draws and .5 of this hard pool.
    # The final refinement fixes those shares at .1 and .25, respectively.
    weights=[w*(.9/.8) for w in weights[:-1]]+[.1]
    fresh_layout['new_weights'][:old_groups]*=1.5
    fresh_layout['new_weights'][old_groups:]*=.5
    assert abs(sum(weights)-1)<1e-8 and abs(float(fresh_layout['new_weights'].sum())-1)<1e-8
    return data,eligible,offsets,weights,roots,layouts,hard,hard_layout,fresh_pairs,fresh_layout


validation=base.validation
