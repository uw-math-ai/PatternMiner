"""Generate exact neighboring training products and auxiliary root targets."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import compact
from exact_roots import root_targets
from full_resources import Budget

def initialize():torch.set_num_threads(1)

def worker(job):
    group,pair=job
    records,calls,terms,excluded=c.previous_generation.worker((pair,pair['n'],group))
    roots,info=root_targets(records,pair,group)
    assert calls+1==pair['exact_query_cost']
    values=records['coefficients'].astype(np.int64)
    assert (values.max(2)[:,0]<=values.sum(2)[:,1]).all()
    assert (values.max(2)[:,1]<=values.sum(2)[:,0]).all()
    return records,roots,dict(group=group,parent=pair['parent'],rows=len(records),exact_queries=calls+1,product_output_terms=terms,excluded_terms_above_17=excluded,root_product=info)

def generate(budget):
    plan=json.loads((s.HERE/'NEIGHBOR_PLAN.json').read_text());assert plan['status']=='frozen_before_labels'
    path=s.HERE/'cache/data/neighbors_train.bin';qraw=s.HERE/'cache/data/neighbors_Q.bin'
    assert not path.exists() and not qraw.exists()
    report=dict(status='running',groups=[],offsets=[0],plan_sha256=c.sha(s.HERE/'NEIGHBOR_PLAN.json'),
        source_sha256=c.sha(Path(__file__)),worker_sha256=s.sha(Path(__file__)))
    with path.open('wb') as out,qraw.open('wb') as roots,ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        pending=deque();group=0
        while group<len(plan['pairs']) or pending:
            while group<len(plan['pairs']) and len(pending)<4:
                pair=plan['pairs'][group]
                for _ in range(pair['exact_query_cost']):budget.exact_query()
                pending.append(pool.submit(worker,(group,pair)));group+=1
            records,q,info=pending.popleft().result();out.write(records.tobytes());roots.write(q.tobytes())
            report['groups'].append(info);report['offsets'].append(report['offsets'][-1]+len(records))
            s.write('NEIGHBOR_DATA.json',report)
    count=report['offsets'][-1];assert count>0
    qpath=s.HERE/'cache/data/neighbors_Q.npy';raw=np.memmap(qraw,dtype=np.int16,mode='r',shape=(count,16));np.save(qpath,raw)
    assert np.array_equal(raw,np.load(qpath,mmap_mode='r'));rawhash=c.sha(qraw);del raw;qraw.unlink()
    report.update(rows=count,exact_queries=budget.exact_queries,data_sha256=c.sha(path),root_sha256=c.sha(qpath),removed_redundant_raw_q_sha256=rawhash,archives=[])
    with path.open('rb') as source:
        for part in range(1000):
            budget.check();block=source.read(compact.DTYPE.itemsize*250000)
            if not block:break
            archive=s.ART/f'neighbors_train_{part:04d}.bin.gz'
            with archive.open('wb') as out:
                with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:z.write(block)
            report['archives'].append(dict(name=archive.name,sha256=c.sha(archive),uncompressed_sha256=hashlib.sha256(block).hexdigest()))
    archive=s.ART/'neighbors_Q.npy.gz'
    with qpath.open('rb') as source,archive.open('wb') as out:
        with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:shutil.copyfileobj(source,z)
    report.update(status='completed',root_archive_sha256=c.sha(archive));s.write('NEIGHBOR_DATA.json',report)
    print(json.dumps(dict(rows=count,queries=budget.exact_queries)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    plan=json.loads((s.HERE/'NEIGHBOR_PLAN.json').read_text())
    budget=Budget('exact_current_failure_neighbors',7200,query_limit=plan['exact_query_cost']);status='failed_or_partial'
    try:generate(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
