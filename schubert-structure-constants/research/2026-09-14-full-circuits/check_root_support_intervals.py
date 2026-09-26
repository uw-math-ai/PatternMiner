"""Scan retained exact degree-one coefficients for disconnected root support."""
import json
from pathlib import Path
import numpy as np
import session as s
import compact
from full_resources import Budget


def main(budget):
    source=s.PREVIOUS;specs=[]
    roots=json.loads((source/'ROOT_TARGETS.json').read_text())['populations']
    for name in ['S8_train','S9_train','S10_train']:
        specs.append((name,source/f'cache/data/{name}_Q.npy',source/f'cache/data/{name}.bin',roots[name]['sha256']))
    for prefix in ['adaptive','screened','late']:
        report=json.loads((source/(prefix.upper()+'_DATA.json')).read_text())
        specs.append((prefix,source/f'cache/data/{prefix}_Q.npy',source/f'cache/data/{prefix}_train.bin',report['root_sha256']))
    results={};witnesses=[]
    for name,path,raw,expected in specs:
        budget.check();assert s.sha(path)==expected
        q=np.load(path,mmap_mode='r');report=dict(rows=len(q),disconnected=0,not_root_multiple=0,nonunimodal=0,examples=[])
        for begin in range(0,len(q),65536):
            budget.check();v=np.asarray(q[begin:begin+65536]);positive=v>0
            assert (v>=0).all() and positive.any(1).all()
            first=positive.argmax(1);last=15-positive[:,::-1].argmax(1)
            disconnected=positive.sum(1)!=last-first+1
            g=np.gcd.reduce(v,axis=1);not_root=(v.max(1)!=g)
            difference=np.diff(v,axis=1);declined=np.maximum.accumulate(difference<0,axis=1)
            nonunimodal=(declined&(difference>0)).any(1)
            report['disconnected']+=int(disconnected.sum());report['not_root_multiple']+=int(not_root.sum());report['nonunimodal']+=int(nonunimodal.sum())
            for kind,condition in [('disconnected',disconnected),('nonunimodal',nonunimodal),('not_root_multiple',not_root)]:
                if any(example['kind']==kind for example in report['examples']):continue
                if not condition.any():continue
                row=begin+int(np.flatnonzero(condition)[0]);record=compact.load(raw)[row];N=int(record['n'])
                item=dict(source=name,row=row,kind=kind,Q=q[row].tolist(),
                    u=record['permutations'][0,:N].tolist(),v=record['permutations'][1,:N].tolist(),w=record['permutations'][2,:N].tolist(),
                    A=record['coefficients'][0,:N-1].tolist(),B=record['coefficients'][1,:N-1].tolist())
                report['examples'].append(item);witnesses.append(item)
        report['Q_sha256']=expected;results[name]=report
    report=dict(status='completed',source_sha256=s.sha(Path(__file__)),sources=results,
        rows=sum(v['rows'] for v in results.values()),disconnected=sum(v['disconnected'] for v in results.values()),
        not_root_multiple=sum(v['not_root_multiple'] for v in results.values()),nonunimodal=sum(v['nonunimodal'] for v in results.values()),
        exact_product_queries=0,role='Finite exact check of a proposed consequence of the frozen primitive-leading-form theorem; not a proof by enumeration')
    s.write('ROOT_SUPPORT_INTERVAL_CHECK.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    budget=Budget('root_support_interval_scan',180);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
