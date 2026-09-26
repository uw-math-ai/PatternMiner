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

def init():torch.set_num_threads(1)


def encode_cpu(records,reverse=None,supplied=None):
    if reverse is None:records=np.repeat(records,2);reverse=np.arange(len(records))%2==1
    pieces={k:[] for k in ['x','mask','y','a']}
    for begin in range(0,len(records),512):
        xx=compact.tensors(records[begin:begin+512],device='cpu')
        values=compact.encode(xx,torch.as_tensor(reverse[begin:begin+512]),None if supplied is None else torch.as_tensor(supplied[begin:begin+512]))
        for k,v in zip(pieces,values):pieces[k].append(v.numpy())
    result={k:np.concatenate(v) for k,v in pieces.items()};result['group']=records['group'].copy();return result


def prepare(budget):
    report=dict(status='running',training={},development={},development_pair_policy='Distinct ordered row pairs and orientation; every supplied index is evaluated, so repeated certificate indices are removed.',source_sha256=c.sha(Path(__file__)),data_sha256=c.sha(s.HERE/'DATA.json'))
    assert not (s.HERE/'PREPARED.json').exists()
    data=json.loads((s.HERE/'DATA.json').read_text());assert data['status']=='completed'
    with ProcessPoolExecutor(max_workers=2,initializer=init) as pool:
        for name,stat in data['populations'].items():
            if name.endswith('_assessment'):continue
            path=s.HERE/f'cache/data/{name}.bin';records=compact.load(path);assert c.sha(path)==stat['sha256']
            offsets=stat['offsets'];jobs=[(str(path),offsets[begin:begin+33],0) for begin in range(0,len(offsets)-1,32)]
            results=[]
            for pairs,groups in pool.map(mine_all_certificates.worker,jobs):budget.check();results.append(pairs)
            pairs=np.concatenate(results)
            if name.endswith('_validation') and len(pairs):
                _,keep=np.unique(pairs[:,:3],axis=0,return_index=True);pairs=pairs[np.sort(keep)]
            pairpath=s.ART/f'{name}_pairs.npy';np.save(pairpath,pairs)
            pairgroups=records['group'][pairs[:,0]] if len(pairs) else np.empty(0,np.int32)
            info=dict(rows=len(records),pairs=len(pairs),pair_groups=len(np.unique(pairgroups)),pair_sha256=c.sha(pairpath))
            if name.endswith('_train'):
                report['training'][name]=info
            else:
                rng=np.random.default_rng(s.SEED+int(name[1:name.index('_')]))
                indices=np.concatenate([rng.choice(np.arange(a,b),min(200,b-a),replace=False) for a,b in zip(offsets[:-1],offsets[1:])])
                ordinary=encode_cpu(records[indices]);np.savez_compressed(s.ART/f'new_{name}.npz',**ordinary)
                np.save(s.ART/f'{name}_selected_rows.npy',indices)
                if len(pairs):
                    encoded=encode_cpu(records[pairs[:,:2].reshape(-1)],pairs[:,2].astype(bool).repeat(2),pairs[:,3].repeat(2))
                    assert np.array_equal(encoded['mask'][::2],encoded['mask'][1::2])
                    assert not (encoded['y'][::2]&encoded['y'][1::2]).any()
                    for ix in range(len(pairs)):
                        valid=encoded['mask'][2*ix];assert np.array_equal(encoded['x'][2*ix,valid],encoded['x'][2*ix+1,valid])
                    np.savez_compressed(s.ART/f'new_{name}_pairs.npz',**encoded)
                info.update(oriented_examples=len(ordinary['x']),groups=int(len(np.unique(ordinary['group']))));report['development'][name]=info
            s.write('PREPARED.json',report);print(json.dumps(dict(population=name,**info)),flush=True)
    report['status']='completed';s.write('PREPARED.json',report)



if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('prepare_fresh_development',1800);status='failed_or_partial'
    try:prepare(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
