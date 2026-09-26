"""Verify factor exclusions and the factor fields of every newly generated row."""
import gzip,json,time
from pathlib import Path
import numpy as np
import session as s
import compact
import plan_neighborhoods as pn
from full_resources import Budget


def main(budget):
    with gzip.open(s.ART/'forbidden_factor_orbits.json.gz','rt') as stream:
        historical={tuple(tuple(p) for p in pair) for pair in json.load(stream)}
    seen=set(historical);results={};total=0
    jobs=[]
    for prefix in ['', 'SECOND_']:
        plan=json.loads((s.HERE/(prefix+'DATA_PLAN.json')).read_text())
        data=json.loads((s.HERE/(prefix+'DATA.json')).read_text());assert data['status']=='completed'
        for name,spec in plan['populations'].items():
            jobs.append((name,spec['pairs'],data['populations'][name],s.HERE/f'cache/data/{name}.bin','sha256'))
    for prefix,filename in [('', 'neighbors_train.bin'),('SECOND_', 'second_neighbors_train.bin')]:
        plan=json.loads((s.HERE/(prefix+'NEIGHBOR_PLAN.json')).read_text())
        data=json.loads((s.HERE/(prefix+'NEIGHBOR_DATA.json')).read_text());assert data['status']=='completed'
        jobs.append((prefix+'neighbors',plan['pairs'],data,s.HERE/'cache/data'/filename,'data_sha256'))
    for name,pairs,spec,path,hash_key in jobs:
        budget.check();assert s.sha(path)==spec[hash_key];records=compact.load(path)
        offsets=spec['offsets'];assert len(offsets)==len(pairs)+1 and offsets[-1]==len(records)
        for group,pair in enumerate(pairs):
            key=pn.orbit(pair);assert key not in seen,(name,group);seen.add(key)
            start,end=offsets[group:group+2]
            for begin in range(start,end,65536):
                budget.check();block=records[begin:min(begin+65536,end)]
                assert (block['group']==group).all()
                for factor,index in [('u',0),('v',1)]:
                    values=np.asarray(pair[factor]);n=len(values)
                    assert (block['n']>=n).all()
                    expected=np.broadcast_to(np.arange(1,18),(len(block),17)).copy()
                    expected[:,:n]=values
                    expected[np.arange(17)[None,:]>=block['n'][:,None]]=0
                    assert np.array_equal(block['permutations'][:,index,:],expected),(name,group,factor)
        results[name]=dict(groups=len(pairs),rows=len(records),sha256=spec[hash_key]);total+=len(records)
    report=dict(status='passed',source_sha256=s.sha(Path(__file__)),historical_excluded_orbits=len(historical),
        new_orbits=len(seen)-len(historical),rows_checked=total,populations=results,
        checks=['Every new factor orbit is distinct from historical exclusions and every other new population.',
                'Every generated row has exactly the factors and group number in its frozen plan, with the recorded stable extension.',
                'Both assessment rounds and every development population are disjoint from both new training populations.'],
        label_scope='Only factor fields, ambient sizes, group numbers and file hashes are inspected. No replacement-assessment model predictions are evaluated.')
    s.write('ALL_FACTOR_SEPARATION.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    while not (s.HERE/'SECOND_NEIGHBOR_PREPARED.json').exists():time.sleep(2)
    budget=Budget('all_new_factor_separation',240);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
