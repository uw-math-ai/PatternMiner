"""Add the explicitly failed first assessment to development, without training on it."""
import json
from pathlib import Path
import numpy as np
import torch
import session as s
import compact
from prepare_development import encode_cpu
from full_resources import Budget


def main(budget):
    assert (s.HERE/'FRESH_ASSESSMENT_EXPOSURE.json').exists()
    assert not (s.HERE/'FIRST_ASSESSMENT_DEVELOPMENT.json').exists()
    metadata=json.loads((s.HERE/'DATA.json').read_text());reports={}
    for n in [7,8,9,10]:
        name=f'S{n}_assessment';spec=metadata['populations'][name]
        path=s.HERE/f'cache/data/{name}.bin';assert s.sha(path)==spec['sha256'];records=compact.load(path)
        with np.load(s.ART/f'fresh_assessment_{name}.npz') as archive:pairs=archive['pairs'].copy()
        rng=np.random.default_rng(s.SEED+87000+n);indices=[]
        for a,b in zip(spec['offsets'][:-1],spec['offsets'][1:]):
            if b>a:indices.extend(rng.choice(np.arange(a,b),min(200,b-a),replace=False).tolist())
        indices=np.array(indices,dtype=np.int64);ordinary=encode_cpu(records[indices])
        np.save(s.ART/f'first_exposed_S{n}_selected_rows.npy',indices)
        for suffix,data in [('',ordinary),('_pairs',encode_cpu(records[pairs[:,:2].reshape(-1)],pairs[:,2].astype(bool).repeat(2),pairs[:,3].repeat(2)))]:
            budget.check();key=f'first_exposed_S{n}{suffix}';output=s.ART/(key+'.npz')
            assert not output.exists();np.savez_compressed(output,**data)
            reports[key]=dict(inputs=len(data['x']),factor_groups=len(np.unique(data['group'])),artifact_sha256=s.sha(output))
    s.write('FIRST_ASSESSMENT_DEVELOPMENT.json',dict(status='completed',populations=reports,
        source_sha256=s.sha(Path(__file__)),exposure_sha256=s.sha(s.HERE/'FRESH_ASSESSMENT_EXPOSURE.json'),
        role='Development only after the recorded independent failure. No rows are added to training. Every first-assessment factor orbit is excluded from new training and replacement assessment.'))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('prepare_first_assessment_development',300);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
