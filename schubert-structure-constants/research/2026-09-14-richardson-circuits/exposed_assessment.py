"""Retain failed assessment pairs as development data, never as training rows."""
import json
from pathlib import Path
import numpy as np
import torch
import context as c
import compact
from prepare_fresh import encode_cpu
from resources import Budget


def prepare(budget):
    assessment = json.loads((c.HERE/'FRESH_ASSESSMENT.json').read_text())
    assert assessment['status'] == 'completed' and not assessment['passes']
    assert (c.HERE/'FRESH_ASSESSMENT_EXPOSURE.json').exists()
    report = dict(status='completed', assessment_sha256=c.sha(c.HERE/'FRESH_ASSESSMENT.json'),
        source_sha256=c.sha(Path(__file__)), role='Development after failed assessment; excluded from training', populations={})
    for n in [7, 8, 9, 10]:
        name = f'S{n}_assessment'
        archive = c.ART/f'fresh_assessment_{name}.npz'
        assert c.sha(archive) == assessment['populations'][name]['artifact_sha256']
        pairs = np.load(archive)['pairs']
        if not len(pairs):
            continue
        budget.check()
        records = compact.load(c.HERE/f'cache/data/{name}.bin')
        encoded = encode_cpu(records[pairs[:, :2].reshape(-1)], pairs[:, 2].astype(bool).repeat(2), pairs[:, 3].repeat(2))
        assert np.array_equal(encoded['mask'][::2], encoded['mask'][1::2])
        assert not (encoded['y'][::2]&encoded['y'][1::2]).any()
        assert np.array_equal(encoded['x'][::2][encoded['mask'][::2]], encoded['x'][1::2][encoded['mask'][1::2]])
        key = f'exposed_fresh_S{n}_pairs'
        output = c.ART/(key+'.npz')
        assert not output.exists()
        np.savez_compressed(output, **encoded)
        report['populations'][key] = dict(pairs=len(pairs), factor_groups=len(np.unique(encoded['group'])), artifact_sha256=c.sha(output))
    c.write('EXPOSED_ASSESSMENT_PAIRS.json', report)
    print(json.dumps(report), flush=True)


def append_validation(populations):
    report = json.loads((c.HERE/'EXPOSED_ASSESSMENT_PAIRS.json').read_text())
    assert report['status'] == 'completed'
    result = dict(populations)
    for name, spec in report['populations'].items():
        path = c.ART/(name+'.npz')
        assert c.sha(path) == spec['artifact_sha256'] and name not in result
        result[name] = dict(np.load(path))
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    budget = Budget('prepare_exposed_assessment_pairs', 60)
    status = 'failed_or_partial'
    try:
        prepare(budget)
        status = 'completed'
    finally:
        print(json.dumps(budget.finish(status)), flush=True)
