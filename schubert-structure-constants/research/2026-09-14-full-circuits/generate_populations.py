"""Generate the frozen fresh populations without evaluating a predictor."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip, hashlib, json, time
from pathlib import Path
import torch
import session as s
from session import c
import compact
from full_resources import Budget

def worker(job):return c.previous_generation.worker(job)

def initialize():torch.set_num_threads(1)


def generate(budget):
    plan=json.loads((s.HERE/'DATA_PLAN.json').read_text());assert plan['planned_descent_product_calls']<=budget.query_limit
    assert not (s.HERE/'DATA.json').exists()
    report=dict(status='running',plan_sha256=c.sha(s.HERE/'DATA_PLAN.json'),source_sha256=c.sha(Path(__file__)),
                dtype=compact.DTYPE.descr,record_bytes=compact.DTYPE.itemsize,populations={},workers=2,
                query_accounting='Parent charges every descent-product call before dispatch, including any interrupted work.')
    for name,spec in plan['populations'].items():
        path=s.HERE/f'cache/data/{name}.bin';assert not path.exists()
        stat=dict(rows=0,completed_groups=0,planned_groups=len(spec['pairs']),offsets=[0],exact_queries=0,excluded_output_terms_above_17=0)
        report['populations'][name]=stat
        with path.open('wb') as stream,ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
            pending=deque();next_group=0
            while next_group<len(spec['pairs']) or pending:
                while next_group<len(spec['pairs']) and len(pending)<4:
                    budget.check();row=spec['pairs'][next_group]
                    calls=sum(len(c.prior.c.a.original.desc(c.prior.c.a.original.inv(tuple(row[k])))) for k in ['u','v'])
                    for _ in range(calls):budget.exact_query()
                    pending.append((next_group,calls,pool.submit(worker,(row,spec['n'],next_group))));next_group+=1
                group,reserved,future=pending.popleft();records,calls,outputs,excluded=future.result();assert calls==reserved
                stream.write(records.tobytes());stat['rows']+=len(records);stat['completed_groups']+=1;stat['offsets'].append(stat['rows'])
                stat['exact_queries']+=calls;stat['excluded_output_terms_above_17']+=excluded
                if (group+1)%50==0:
                    stream.flush();s.write('DATA.json',report)
                    print(json.dumps(dict(population=name,groups=group+1,rows=stat['rows'],queries=budget.exact_queries,elapsed=time.monotonic()-budget.start)),flush=True)
        stat.update(sha256=c.sha(path),bytes=path.stat().st_size,archives=[])
        with path.open('rb') as stream:
            for part in range(10000):
                budget.check();block=stream.read(compact.DTYPE.itemsize*250000)
                if not block:break
                archive=s.ART/f'{name}_{part:04d}.bin.gz'
                with archive.open('wb') as out:
                    with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:z.write(block)
                stat['archives'].append(dict(name=archive.name,sha256=c.sha(archive),uncompressed_sha256=hashlib.sha256(block).hexdigest()))
        s.write('DATA.json',report)
    report.update(status='completed',exact_queries=budget.exact_queries);s.write('DATA.json',report)



if __name__=='__main__':
    torch.set_num_threads(1)
    (s.HERE/'cache/data').mkdir(parents=True,exist_ok=True)
    queries=json.loads((s.HERE/'DATA_PLAN.json').read_text())['planned_descent_product_calls']
    budget=Budget('fresh_exact_populations',7200,query_limit=queries);status='failed_or_partial'
    try:generate(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
