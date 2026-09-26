"""Mine training collisions for every positive supplied coefficient.

Three CPU workers share the four available CPUs with the running trainer.
No exact queries or GPU operations occur. The elapsed wall charge is deferred
until the trainer closes its sequential ledger.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import torch
import common as r
import compact


def worker(job):
    path,offsets,source_offset=job
    data=compact.load(Path(path))
    pairs=[]
    groups=set()
    cuts=np.arange(17)
    for begin,end in zip(offsets[:-1],offsets[1:]):
        if begin==end:continue
        block=data[begin:end]
        inverse=np.where(block['permutations']>0,block['permutations'],18+cuts).argsort(-1)+1
        supports=((block['coefficients']>0)*(1<<cuts[:16])).sum(-1)
        for reverse in [0,1]:
            aa,bb=(1,0) if reverse else (0,1)
            candidates=np.flatnonzero(inverse[0,bb,:-1]>inverse[0,bb,1:])
            base={int(j) for k in candidates for j in [k,k+1]}
            columns={i:sorted(base|{i,i+1}) for i in range(16)}
            table={}
            for t,row in enumerate(block):
                valid=int(supports[t,bb])
                supplied=int(supports[t,aa])
                winverse=inverse[t,2].tolist()
                while supplied:
                    bit=supplied&-supplied
                    i=bit.bit_length()-1
                    supplied-=bit
                    key=(int(row['n']),i,tuple(winverse[k] for k in columns[i]))
                    patterns=table.setdefault(key,{})
                    if valid not in patterns:
                        for old,old_index in patterns.items():
                            if old&valid==0:
                                pairs.append((source_offset+old_index,source_offset+begin+t,reverse,i))
                                groups.add(int(row['group']))
                        patterns[valid]=begin+t
    return np.asarray(pairs,dtype=np.int64).reshape(-1,4),sorted(groups)


def init_worker():torch.set_num_threads(1)


def main(start):
    if (r.HERE/'HARD_ALL_CERTIFICATES.json').exists():raise RuntimeError('This mining run already exists')
    metadata=json.loads((r.HERE/'DATA.json').read_text())['populations']
    training=json.loads((r.HERE/'TRAINING_DATA.json').read_text())['sources']
    jobs=[];source_offset=0
    for name,entry in training.items():
        if name!='legacy_train':
            offsets=metadata[name]['offsets']
            for begin in range(0,len(offsets)-1,32):
                jobs.append((name,(str(r.HERE/f'cache/data/{name}.bin'),offsets[begin:begin+33],source_offset)))
        source_offset+=entry['rows']
    results=[];groups={name:set() for name in training if name!='legacy_train'};counts=Counter()
    with ProcessPoolExecutor(max_workers=3,initializer=init_worker) as pool:
        futures=[(name,pool.submit(worker,job)) for name,job in jobs]
        for index,(name,future) in enumerate(futures):
            if time.monotonic()-start>900:raise TimeoutError('All-certificate mining reservation exhausted')
            pairs,found_groups=future.result()
            results.append(pairs);groups[name].update(found_groups);counts[name]+=len(pairs)
            if (index+1)%20==0:print(json.dumps(dict(chunks=index+1,total_chunks=len(futures),pairs=sum(counts.values()))),flush=True)
    pairs=np.concatenate(results)
    np.save(r.ART/'training_all_certificate_pairs.npy',pairs)
    report=dict(pairs=len(pairs),populations={name:dict(pairs=counts[name],factor_groups=len(groups[name])) for name in groups},
                training_rows=source_offset,columns=['left_global_row','right_global_row','reverse','supplied_index_zero_based'],
                source_sha256=r.c.a.sha(Path(__file__)),pair_sha256=r.c.a.sha(r.ART/'training_all_certificate_pairs.npy'),
                source_plan_sha256=r.c.a.sha(r.HERE/'DATA_PLAN.json'),
                scope='Every positive supplied coefficient, only new training groups; no model scores or exact queries')
    r.write('HARD_ALL_CERTIFICATES.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    pending=r.HERE/'CPU_ALL_CERTIFICATES_PENDING.json'
    if pending.exists():raise RuntimeError('This mining run already exists')
    torch.set_num_threads(1)
    start=time.monotonic();status='failed_or_partial'
    try:
        main(start);status='completed'
    finally:
        r.write(pending.name,dict(name='mine_all_supplied_training_pairs',status=status,
                    wall_seconds=time.monotonic()-start,gpu_seconds_charged=0,
                    gpu_charge_scope='CPU only, three workers',exact_queries=0,
                    accounting='Deferred append until the concurrent training ledger is closed'))
