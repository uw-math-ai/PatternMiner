"""Prepare fresh training and development data without opening assessment labels."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import lrcalc
import context as c
import compact
import mine_all_certificates
from resources import Budget


def root_worker(job):
    name,group,begin,end,pair=job
    trim=c.prior.c.a.original.trim
    raw=lrcalc.schubmult(list(trim(tuple(pair['u']))),list(trim(tuple(pair['v']))))
    product={trim(tuple(w)):int(value) for w,value in raw.items()}
    records=compact.load(c.HERE/f'cache/data/{name}.bin')[begin:end]
    roots=np.zeros((len(records),16),np.int16)
    for k,row in enumerate(records):
        n=int(row['n']);w=tuple(map(int,row['permutations'][2,:n]))+(n+1,);inverse=np.argsort(w)
        f=np.zeros(n,np.int64);f[:n-1]=-row['coefficients'][:,:n-1].astype(np.int64).sum(0)
        for i in range(n):
            if inverse[i]<inverse[i+1]:
                changed=list(w);changed[inverse[i]],changed[inverse[i+1]]=changed[inverse[i+1]],changed[inverse[i]]
                f[i]+=product.get(trim(tuple(changed)),0)
        assert np.dot(np.arange(1,n+1),f)==0
        q=np.array([np.dot(np.arange(1,n-j),f[j+1:]) for j in range(n-1)])
        assert (q>=0).all() and (q[None,:]>=row['coefficients'][:,:n-1]).all()
        padded=np.r_[0,q,0];assert np.array_equal(2*q-padded[:-2]-padded[2:],-f[:n-1])
        assert q.max(initial=0)<=32767;roots[k,:n-1]=q
    return roots,dict(group=group,begin=begin,end=end,product_terms=len(product),
        product_sha256=hashlib.sha256(json.dumps(sorted(product.items()),separators=(',',':')).encode()).hexdigest())


def init():torch.set_num_threads(1)


def roots(budget):
    plan=json.loads((c.HERE/'DATA_PLAN.json').read_text())['populations']
    data=json.loads((c.HERE/'DATA.json').read_text());assert data['status']=='completed'
    report=dict(status='running',populations={},source_sha256=c.sha(Path(__file__)),data_sha256=c.sha(c.HERE/'DATA.json'))
    assert not (c.HERE/'ROOT_TARGETS.json').exists()
    with ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
        for name,spec in plan.items():
            if not name.endswith('_train'):continue
            stat=data['populations'][name];path=c.HERE/f'cache/data/{name}_Q.npy';assert not path.exists()
            values=np.lib.format.open_memmap(path,mode='w+',dtype=np.int16,shape=(stat['rows'],16));values[:]=-1
            info=dict(groups=[],rows=stat['rows']);report['populations'][name]=info
            pending=deque();g=0
            while g<len(spec['pairs']) or pending:
                while g<len(spec['pairs']) and len(pending)<6:
                    budget.exact_query();pending.append(pool.submit(root_worker,(name,g,stat['offsets'][g],stat['offsets'][g+1],spec['pairs'][g])));g+=1
                q,item=pending.popleft().result();values[item['begin']:item['end']]=q;info['groups'].append(item)
                if len(info['groups'])%100==0:
                    values.flush();c.write('ROOT_TARGETS.json',report);print(json.dumps(dict(population=name,groups=len(info['groups']),queries=budget.exact_queries)),flush=True)
            values.flush();assert (values>=0).all();info['sha256']=c.sha(path)
            archive=c.ART/f'{name}_Q.npy.gz'
            with path.open('rb') as src,archive.open('wb') as out:
                with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:shutil.copyfileobj(src,z)
            info['archive_sha256']=c.sha(archive);c.write('ROOT_TARGETS.json',report)
    report.update(status='completed',exact_queries=budget.exact_queries,checks='Boundary identity, Cartan relation, and Q >= A,B >= 0 on every labelled row')
    c.write('ROOT_TARGETS.json',report)


def encode_cpu(records,reverse=None,supplied=None):
    if reverse is None:records=np.repeat(records,2);reverse=np.arange(len(records))%2==1
    pieces={k:[] for k in ['x','mask','y','a']}
    for begin in range(0,len(records),512):
        xx=compact.tensors(records[begin:begin+512],device='cpu')
        values=compact.encode(xx,torch.as_tensor(reverse[begin:begin+512]),None if supplied is None else torch.as_tensor(supplied[begin:begin+512]))
        for k,v in zip(pieces,values):pieces[k].append(v.numpy())
    result={k:np.concatenate(v) for k,v in pieces.items()};result['group']=records['group'].copy();return result


def prepare(budget):
    report=dict(status='running',training={},development={},development_pair_policy='Distinct ordered row pairs and orientation; every supplied index is evaluated, so repeated certificate indices are removed.',source_sha256=c.sha(Path(__file__)),data_sha256=c.sha(c.HERE/'DATA.json'))
    assert not (c.HERE/'PREPARED.json').exists()
    data=json.loads((c.HERE/'DATA.json').read_text());assert data['status']=='completed'
    with ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
        for name,stat in data['populations'].items():
            if name.endswith('_assessment'):continue
            path=c.HERE/f'cache/data/{name}.bin';records=compact.load(path);assert c.sha(path)==stat['sha256']
            offsets=stat['offsets'];jobs=[(str(path),offsets[begin:begin+33],0) for begin in range(0,len(offsets)-1,32)]
            results=[]
            for pairs,groups in pool.map(mine_all_certificates.worker,jobs):budget.check();results.append(pairs)
            pairs=np.concatenate(results)
            if name.endswith('_validation') and len(pairs):
                _,keep=np.unique(pairs[:,:3],axis=0,return_index=True);pairs=pairs[np.sort(keep)]
            pairpath=c.ART/f'{name}_pairs.npy';np.save(pairpath,pairs)
            pairgroups=records['group'][pairs[:,0]] if len(pairs) else np.empty(0,np.int32)
            info=dict(rows=len(records),pairs=len(pairs),pair_groups=len(np.unique(pairgroups)),pair_sha256=c.sha(pairpath))
            if name.endswith('_train'):
                report['training'][name]=info
            else:
                rng=np.random.default_rng(c.SEED+int(name[1:name.index('_')]))
                indices=np.concatenate([rng.choice(np.arange(a,b),min(200,b-a),replace=False) for a,b in zip(offsets[:-1],offsets[1:])])
                ordinary=encode_cpu(records[indices]);np.savez_compressed(c.ART/f'fresh_{name}.npz',**ordinary)
                np.save(c.ART/f'{name}_selected_rows.npy',indices)
                if len(pairs):
                    encoded=encode_cpu(records[pairs[:,:2].reshape(-1)],pairs[:,2].astype(bool).repeat(2),pairs[:,3].repeat(2))
                    assert np.array_equal(encoded['mask'][::2],encoded['mask'][1::2])
                    assert not (encoded['y'][::2]&encoded['y'][1::2]).any()
                    for ix in range(len(pairs)):
                        valid=encoded['mask'][2*ix];assert np.array_equal(encoded['x'][2*ix,valid],encoded['x'][2*ix+1,valid])
                    np.savez_compressed(c.ART/f'fresh_{name}_pairs.npz',**encoded)
                info.update(oriented_examples=len(ordinary['x']),groups=int(len(np.unique(ordinary['group']))));report['development'][name]=info
            c.write('PREPARED.json',report);print(json.dumps(dict(population=name,**info)),flush=True)
    report['status']='completed';c.write('PREPARED.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['roots','prepare']);args=parser.parse_args()
    torch.set_num_threads(1)
    budget=Budget('fresh_'+args.action,1800,query_limit=2000 if args.action=='roots' else 0);status='failed_or_partial'
    try:
        (roots if args.action=='roots' else prepare)(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
