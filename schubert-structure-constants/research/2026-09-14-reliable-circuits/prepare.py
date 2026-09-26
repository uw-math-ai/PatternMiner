"""Prepare training records and validation without opening assessment scores."""
import json
from pathlib import Path
import numpy as np
import torch
import common as r
import compact
from budget import Budget


def collisions(data, offsets, budget):
    pairs=[]
    cuts=np.arange(17)
    for begin,end in zip(offsets[:-1],offsets[1:]):
        budget.check()
        if begin==end:continue
        block=data[begin:end]
        inverse=np.where(block['permutations']>0,block['permutations'],18+cuts).argsort(-1)+1
        inverse*=cuts[None,None,:]<block['n'][:,None,None]
        supports=((block['coefficients']>0)*(1<<cuts[:16])).sum(-1)
        for reverse in [False,True]:
            aa,bb=(1,0) if reverse else (0,1)
            table={}
            for t,row in enumerate(block):
                n=int(row['n'])
                i=int(np.flatnonzero(row['coefficients'][aa])[0])
                iv=inverse[t,bb]
                candidates=np.flatnonzero(iv[:n-1]>iv[1:n])
                exposed=sorted({i,i+1}|{int(j) for k in candidates for j in [k,k+1]})
                key=(n,i,tuple(inverse[t,2,exposed].tolist()))
                valid=int(supports[t,bb])
                patterns=table.setdefault(key,{})
                if valid not in patterns:
                    for old,old_index in patterns.items():
                        if old&valid==0:pairs.append((old_index,begin+t,int(reverse)))
                    patterns[valid]=begin+t
    return np.asarray(pairs,dtype=np.int64).reshape(-1,3)


def encoded(records, reverse=None):
    if reverse is None:
        records=np.repeat(records,2)
        reverse=np.arange(len(records))%2==1
    pieces={k:[] for k in ['x','mask','y','a']}
    with torch.inference_mode():
        for begin in range(0,len(records),2048):
            tensor=compact.tensors(records[begin:begin+2048])
            values=compact.encode(tensor,torch.as_tensor(reverse[begin:begin+2048],device='cuda'))
            for key,value in zip(pieces,values):pieces[key].append(value.cpu().numpy())
    out={key:np.concatenate(values) for key,values in pieces.items()}
    out['group']=np.unique(records['group'],return_inverse=True)[1].astype(np.int32)
    return out


def main(budget):
    report=json.loads((r.HERE/'DATA.json').read_text())
    old_members=json.loads((r.c.PREVIOUS/'SPLITS.json').read_text())
    legacy=[]
    allowed=[]
    for name in r.POOLS:
        budget.check()
        if name=='fresh_train':
            rows=r.c.a.read(r.PREVIOUS/'artifacts/fresh_S9_train.jsonl.gz')+r.c.a.read(r.PREVIOUS/'artifacts/fresh_S10_train.jsonl.gz')
        else:
            spec=old_members[name]
            source=r.c.a.read(r.c.PREVIOUS/'artifacts'/spec['source'])
            rows=[source[k] for k in spec['indices']]
            del source
        legacy.append(compact.from_rows(rows))
        allowed.append(np.load(r.PREVIOUS/'cache'/name/'symmetry_allowed.npy')[::2])
        del rows
    legacy=np.concatenate(legacy)
    path=r.HERE/'cache/data/legacy_train.bin'
    legacy.tofile(path)
    np.save(r.HERE/'cache/legacy_reflection.npy',np.concatenate(allowed))
    sources={'legacy_train':dict(rows=len(legacy),weight=.20)}
    for n,weight in [(8,.05),(9,.50),(10,.25)]:
        name=f'S{n}_train'
        sources[name]=dict(rows=report['populations'][name]['rows'],weight=weight)
    r.write('TRAINING_DATA.json',dict(sources=sources,legacy_sources=r.POOLS,
            source_sha256=r.c.a.sha(Path(__file__)),dtype=compact.DTYPE.descr))
    del legacy
    validation={}
    hard_records=[]
    hard_reverse=[]
    hard_metadata=[]
    global_group=0
    for name in ['S9_validation','S10_validation']:
        budget.check()
        data=compact.load(r.HERE/f'cache/data/{name}.bin')
        spec=report['populations'][name]
        rng=np.random.default_rng(r.SEED+int(name[1:name.index('_')]))
        indices=[]
        for begin,end in zip(spec['offsets'][:-1],spec['offsets'][1:]):
            indices.extend(sorted(rng.choice(np.arange(begin,end),size=min(200,end-begin),replace=False).tolist()))
        selected=np.asarray(indices,dtype=np.int64)
        values=encoded(data[selected])
        np.savez_compressed(r.ART/f'{name}.npz',**values)
        validation[name]=dict(indices=selected.tolist(),examples=len(values['x']),factor_groups=len(np.unique(values['group'])))
        pairs=collisions(data,spec['offsets'],budget)
        for left,right,reverse in pairs:
            records=np.array(data[[left,right]],copy=True)
            records['group']+=global_group
            hard_records.extend(records)
            hard_reverse.extend([bool(reverse)]*2)
            hard_metadata.append(dict(population=name,rows=[int(left),int(right)],reverse=bool(reverse),group=int(records['group'][0])))
        global_group+=spec['completed_groups']
    hard=encoded(np.asarray(hard_records,dtype=compact.DTYPE),np.asarray(hard_reverse))
    np.testing.assert_array_equal(hard['mask'][::2],hard['mask'][1::2])
    for k in range(0,len(hard['x']),2):
        np.testing.assert_array_equal(hard['x'][k,hard['mask'][k]],hard['x'][k+1,hard['mask'][k+1]])
    assert not (hard['y'][::2]&hard['y'][1::2]).any()
    np.savez_compressed(r.ART/'hard_validation.npz',**hard)
    r.write('VALIDATION_DATA.json',dict(populations=validation,hard_pairs=hard_metadata,
              hard_pair_count=len(hard_metadata),features_equal_verified=True,
              assessment_or_confirmation_scores_used=False))


if __name__=='__main__':
    torch.set_num_threads(4)
    budget=Budget('expanded_data_preparation',gpu=True)
    status='failed_or_partial'
    try:
        main(budget);status='completed'
    finally:
        print(json.dumps(budget.finish(status)),flush=True)
