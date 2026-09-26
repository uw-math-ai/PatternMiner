"""Extend the degree-one support census to the newly generated exact data."""
import json
from pathlib import Path
import numpy as np
import session as s
from full_resources import Budget


def main(budget):
    metadata=json.loads((s.HERE/'SECOND_NEIGHBOR_DATA.json').read_text())
    path=s.HERE/'cache/data/second_neighbors_Q.npy'
    assert s.sha(path)==metadata['root_sha256']
    q=np.load(path,mmap_mode='r')
    counts=dict(rows=len(q),disconnected=0,not_root_multiple=0,nonunimodal=0)
    examples={}
    for start in range(0,len(q),65536):
        budget.check();v=np.asarray(q[start:start+65536]);positive=v>0
        assert (v>=0).all() and positive.any(1).all()
        first=positive.argmax(1);last=v.shape[1]-1-positive[:,::-1].argmax(1)
        diff=np.diff(v,axis=1)
        tests=dict(disconnected=positive.sum(1)!=last-first+1,
            not_root_multiple=v.max(1)!=np.gcd.reduce(v,axis=1),
            nonunimodal=(np.maximum.accumulate(diff<0,axis=1)&(diff>0)).any(1))
        for name,condition in tests.items():
            counts[name]+=int(condition.sum())
            if condition.any() and name not in examples:
                row=start+int(np.flatnonzero(condition)[0]);examples[name]=dict(row=row,Q=q[row].tolist())
    previous=json.loads((s.HERE/'NEIGHBOR_ROOT_SUPPORT_CHECK.json').read_text())
    report=dict(status='completed',source_sha256=s.sha(Path(__file__)),root_sha256=s.sha(path),
        neighbor_data_sha256=s.sha(s.HERE/'SECOND_NEIGHBOR_DATA.json'),new_counts=counts,examples=examples,
        previous_census_sha256=s.sha(s.HERE/'NEIGHBOR_ROOT_SUPPORT_CHECK.json'),
        combined_counts={key:counts[key]+previous['combined_counts'][key] for key in counts},exact_product_queries=0,
        scope='Entries in two retained exact corpora; counts need not be counts of distinct coefficient polynomials. No new exact-product queries and no assessment data are used.')
    assert not (s.HERE/'SECOND_NEIGHBOR_ROOT_SUPPORT_CHECK.json').exists()
    s.write('SECOND_NEIGHBOR_ROOT_SUPPORT_CHECK.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    budget=Budget('second_neighbor_root_support_scan',180);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
