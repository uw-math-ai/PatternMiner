"""Atomic cumulative accounting and bounded reservations for concurrent runs."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def read(name,default=None):
    path=HERE/name
    return json.loads(path.read_text()) if path.exists() else default


def write(name,value):
    path=HERE/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


@contextmanager
def lock():
    with (HERE/'RESOURCE_LOCK').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


class Budget:
    def __init__(self,name,wall_limit,gpu=False,query_limit=0):
        self.name=name;self.wall_limit=wall_limit;self.gpu=gpu;self.query_limit=query_limit;self.exact_queries=0
        self.start=time.monotonic()
        with lock():
            agreement=read('resource_agreement.json');assert agreement['status']=='confirmed'
            ledger=read('RESOURCE_USE.json',[]);active=read('RESOURCE_RESERVATIONS.json',{})
            assert name not in active and all(r['name']!=name for r in ledger)
            reservation=dict(wall_seconds=wall_limit,gpu_seconds_charged=wall_limit if gpu else 0,exact_queries=query_limit)
            for key,cap in [('wall_seconds','wall_seconds'),('gpu_seconds_charged','gpu_seconds'),('exact_queries','exact_queries')]:
                used=sum(r[key] for r in ledger)+sum(r[key] for r in active.values())
                assert used+reservation[key]<=agreement[cap],(name,key,used,reservation[key],agreement[cap])
            active[name]=reservation;write('RESOURCE_RESERVATIONS.json',active)

    def check(self):
        if time.monotonic()-self.start>=self.wall_limit:raise TimeoutError(f'{self.name}: reserved wall time exhausted')
        assert self.exact_queries<=self.query_limit

    def exact_query(self):
        self.check();assert self.exact_queries<self.query_limit;self.exact_queries+=1

    def finish(self,status):
        elapsed=time.monotonic()-self.start
        record=dict(name=self.name,status=status,wall_seconds=elapsed,gpu_seconds_charged=elapsed if self.gpu else 0,
                    gpu_charge_scope='Entire timed process' if self.gpu else 'CPU only',exact_queries=self.exact_queries)
        with lock():
            ledger=read('RESOURCE_USE.json',[]);active=read('RESOURCE_RESERVATIONS.json',{})
            assert self.name in active and all(r['name']!=self.name for r in ledger)
            del active[self.name];ledger.append(record)
            write('RESOURCE_USE.json',ledger);write('RESOURCE_RESERVATIONS.json',active)
        return record
