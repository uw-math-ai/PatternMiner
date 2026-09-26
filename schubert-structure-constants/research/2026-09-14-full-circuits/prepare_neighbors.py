"""Prepare the fresh development populations; leave assessment predictions unopened."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import compact
import mine_all_certificates
from full_resources import Budget

def prepare(budget):
    report=json.loads((s.HERE/'NEIGHBOR_DATA.json').read_text());assert report['status']=='completed'
    path=s.HERE/'cache/data/neighbors_train.bin';assert c.sha(path)==report['data_sha256'];data=compact.load(path)
    offsets=report['offsets'];jobs=[(str(path),offsets[i:i+33],0) for i in range(0,len(offsets)-1,32)];results=[]
    with ProcessPoolExecutor(max_workers=2,initializer=mine_all_certificates.init_worker) as pool:
        for pairs,_ in pool.map(mine_all_certificates.worker,jobs):budget.check();results.append(pairs)
    pairs=np.concatenate(results);groups=data['group'][pairs[:,0]]
    out=s.ART/'neighbors_training_pairs.npy';np.save(out,pairs)
    s.write('NEIGHBOR_PREPARED.json',dict(status='completed',rows=len(data),pairs=len(pairs),factor_groups=len(np.unique(groups)),
        pair_sha256=c.sha(out),data_metadata_sha256=c.sha(s.HERE/'NEIGHBOR_DATA.json'),source_sha256=c.sha(Path(__file__))))
    print(json.dumps(dict(rows=len(data),pairs=len(pairs),groups=len(np.unique(groups)))),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('prepare_current_failure_neighbors',1800);status='failed_or_partial'
    try:prepare(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
