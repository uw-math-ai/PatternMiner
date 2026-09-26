"""Generate exact product groups into compact, append-only populations."""
from collections import Counter
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
import random
import shutil
import torch
import numpy as np
import lrcalc
import common as r
import compact
from budget import Budget

PLAN = {'train': {8:400,9:4000,10:1800}, 'validation': {9:100,10:60},
        'assessment': {7:20,8:40,9:100,10:60}, 'discovery': {9:80,10:40},
        'confirmation': {9:80,10:40}}
CAPS = {7:7,8:9,9:11,10:13}


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def make_plan(budget):
    seen=set()
    for name in r.c.a.NAMES.values():
        for row in r.c.a.read(r.c.PREVIOUS/'artifacts'/name):
            seen.add(r.symmetry.product_orbit(r.c.a.original.group(row)))
    previous=json.loads((r.PREVIOUS/'FRESH_PAIRS.json').read_text())
    for population in previous.values():
        for row in population['pairs']:
            seen.add(r.symmetry.product_orbit(r.c.a.original.group(row)))
    historical=len(seen)
    plan={}
    for part, sizes in PLAN.items():
        for n, count in sizes.items():
            budget.check()
            rng=random.Random(r.SEED+1000*n+list(PLAN).index(part))
            pairs=[]
            def sample():
                while True:
                    p=list(range(1,n+1));rng.shuffle(p)
                    if 1<=r.c.a.original.length(p)<=CAPS[n]:
                        return p
            while len(pairs)<count:
                row=dict(u=sample(),v=sample())
                orbit=r.symmetry.product_orbit(r.c.a.original.group(row))
                if orbit in seen:
                    continue
                seen.add(orbit);pairs.append(row)
            plan[f'S{n}_{part}']=dict(n=n,cap=CAPS[n],pairs=pairs)
    r.write('DATA_PLAN.json',dict(historical_orbits_excluded=historical,populations=plan))
    return plan


def product_rows(record, n, budget):
    u,v=tuple(record['u']),tuple(record['v'])
    terms={}
    calls=outputs=excluded=0
    for side,p in enumerate([u,v]):
        for i in sorted(r.c.a.original.desc(r.c.a.original.inv(p))):
            left=r.c.a.original.left_swap(u,i) if side==0 else u
            right=r.c.a.original.left_swap(v,i) if side==1 else v
            budget.exact_query();calls+=1
            result=lrcalc.schubmult(list(r.c.a.original.trim(left)),list(r.c.a.original.trim(right)))
            outputs+=len(result)
            for w,value in result.items():
                w=r.c.a.original.trim(w)
                if len(w)>17:
                    excluded+=1
                    continue
                if w not in terms:
                    terms[w]=[[0]*17,[0]*17]
                terms[w][side][i]=int(value)
    rows=[]
    for w,(aa,bb) in sorted(terms.items()):
        if not any(aa) or not any(bb):
            continue
        N=max(n,len(w))
        pad=lambda p: list(p)+list(range(len(p)+1,N+1))
        rows.append(dict(u=pad(u),v=pad(v),w=pad(w),A=aa[:N],B=bb[:N]))
    return rows,calls,outputs,excluded


def main(budget):
    if (r.HERE/'DATA_PLAN.json').exists():
        plan=json.loads((r.HERE/'DATA_PLAN.json').read_text())['populations']
    else:
        plan=make_plan(budget)
    storage=r.HERE/'cache/data'
    storage.mkdir(parents=True,exist_ok=True)
    report=json.loads((r.HERE/'DATA.json').read_text()) if (r.HERE/'DATA.json').exists() else dict(dtype=compact.DTYPE.descr,record_bytes=compact.DTYPE.itemsize,populations={},status='running')
    report.update(source_sha256=r.c.a.sha(Path(__file__)),plan_sha256=r.c.a.sha(r.HERE/'DATA_PLAN.json'),
                  worker_processes=3,query_accounting='Parent reserves all descent-product calls before dispatch; reservations are charged conservatively even on interruption')
    order=sorted(plan,key=lambda name:(name.endswith('_train'),name))
    for name in order:
        n=plan[name]['n']
        path=storage/f'{name}.bin'
        previous=report['populations'].get(name)
        if previous and previous['completed_groups']==previous['planned_groups']:
            assert digest(path)==previous['sha256']
            continue
        if path.exists():
            raise RuntimeError(f'An incomplete population needs explicit archival before resumption: {path}')
        stat=dict(planned_groups=len(plan[name]['pairs']),completed_groups=0,rows=0,exact_product_calls=0,
                  product_output_terms=0,excluded_output_terms_above_17=0,offsets=[0],coefficient_max=0)
        report['populations'][name]=stat
        with path.open('wb') as stream, ProcessPoolExecutor(max_workers=3,initializer=worker_init) as pool:
            pending=deque()
            next_group=0
            while next_group<len(plan[name]['pairs']) or pending:
                while next_group<len(plan[name]['pairs']) and len(pending)<6:
                    budget.check()
                    if budget.exact_remaining-budget.exact_queries < 2*n+1000 or shutil.disk_usage(storage).free<500_000_000:
                        report['status']='resource_limited_partial'
                        next_group=len(plan[name]['pairs'])
                        break
                    record=plan[name]['pairs'][next_group]
                    count=sum(len(r.c.a.original.desc(r.c.a.original.inv(tuple(record[k])))) for k in ['u','v'])
                    for _ in range(count):budget.exact_query()
                    pending.append((next_group,count,pool.submit(worker,(record,n,next_group))))
                    next_group+=1
                if not pending:break
                group,reserved,future=pending.popleft()
                data,calls,outputs,excluded=future.result()
                assert calls==reserved
                stream.write(data.tobytes())
                stat['rows']+=len(data)
                stat['completed_groups']+=1
                stat['exact_product_calls']+=calls
                stat['product_output_terms']+=outputs
                stat['excluded_output_terms_above_17']+=excluded
                stat['offsets'].append(stat['rows'])
                stat['coefficient_max']=max(stat['coefficient_max'],int(data['coefficients'].max()) if len(data) else 0)
                if (group+1)%50==0:
                    stream.flush()
                    r.write('DATA.json',report)
                    print(json.dumps(dict(population=name,groups=group+1,rows=stat['rows'],queries=budget.exact_queries)),flush=True)
        stat['sha256']=digest(path)
        stat['bytes']=path.stat().st_size
        stat['archives']=[]
        with path.open('rb') as source:
            part=0
            while True:
                budget.check()
                block=source.read(compact.DTYPE.itemsize*250000)
                if not block:break
                archive=r.ART/f'{name}_{part:04d}.bin.gz'
                with archive.open('wb') as file:
                    with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as stream:
                        stream.write(block)
                stat['archives'].append(dict(name=archive.name,records=len(block)//compact.DTYPE.itemsize,
                                             sha256=digest(archive),uncompressed_sha256=hashlib.sha256(block).hexdigest()))
                part+=1
        r.write('DATA.json',report)
    if report['status']=='running':report['status']='completed'
    report['exact_queries']=budget.exact_queries
    r.write('DATA.json',report)


def worker_init():
    torch.set_num_threads(1)


class QueryCounter:
    def exact_query(self):pass


def worker(job):
    record,n,group=job
    rows,calls,outputs,excluded=product_rows(record,n,QueryCounter())
    data=compact.from_rows(rows,group)
    assert np.all(data['lengths'][:,2].astype(int)==data['lengths'][:,0].astype(int)+data['lengths'][:,1].astype(int)-1)
    return data,calls,outputs,excluded


if __name__=='__main__':
    torch.set_num_threads(1)
    budget=Budget('expanded_product_generation_parallel')
    status='failed_or_partial'
    try:
        main(budget);status='completed'
    finally:
        print(json.dumps(budget.finish(status)),flush=True)
