"""Measure fitting and robustness on development populations only."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import common as r
import compact
from prepare import encoded


def measure(model, data, recorded=None):
    count=np.zeros(len(data['x']),np.int32)
    errors=np.zeros_like(count)
    first_good=np.zeros(len(count),bool)
    last_good=np.zeros_like(first_good)
    recorded_good=np.zeros_like(first_good)
    with torch.inference_mode():
        for begin in range(0,len(count),512):
            block={k:torch.as_tensor(v[begin:begin+512],device='cuda') for k,v in data.items() if k in ['x','mask','y','a']}
            row,ii=block['a'].nonzero(as_tuple=True)
            good=[]
            for start in range(0,len(row),512):
                ix=row[start:start+512]
                xx=r.c.a.change_witness(block['x'][ix],ii[start:start+512])
                pred=model(xx).masked_fill(~block['mask'][ix],-1e9).argmax(1)
                good.append(block['y'][ix,pred].cpu().numpy())
            good=np.concatenate(good)
            rr=row.cpu().numpy()
            jj=ii.cpu().numpy()
            first=block['a'].long().argmax(1).cpu().numpy()
            last=15-block['a'].flip(1).long().argmax(1).cpu().numpy()
            count[begin:begin+len(first)]=np.bincount(rr,minlength=len(first))
            errors[begin:begin+len(first)]=np.bincount(rr,weights=~good,minlength=len(first)).astype(np.int32)
            first_good[begin+rr[jj==first[rr]]]=good[jj==first[rr]]
            last_good[begin+rr[jj==last[rr]]]=good[jj==last[rr]]
            if recorded is not None:
                select=jj==recorded[begin+rr]
                assert select.sum()==len(first)
                recorded_good[begin+rr[select]]=good[select]
    result=dict(inputs=int(count.sum()),errors=int(errors.sum()),
                supplied_accuracy=float(1-errors.sum()/count.sum()),
                all_supplied_correct=float((errors==0).mean()),
                first=float(first_good.mean()),last=float(last_good.mean()))
    if recorded is not None:result['recorded']=float(recorded_good.mean())
    if len(count)%2==0:
        result['pairs']=len(count)//2
        result['both_first']=float((first_good[::2]&first_good[1::2]).mean())
        result['both_last']=float((last_good[::2]&last_good[1::2]).mean())
        result['both_all_supplied']=float(((errors[::2]==0)&(errors[1::2]==0)).mean())
        if recorded is not None:result['both_recorded']=float((recorded_good[::2]&recorded_good[1::2]).mean())
    return result,dict(count=count,errors=errors,first_good=first_good,last_good=last_good,recorded_good=recorded_good)


def main(args):
    checkpoint=r.ART/f'{args.name}.pt'
    source=r.ART/f'{args.arm}.pt'
    if checkpoint.exists():raise RuntimeError('Use a new diagnostic name')
    before=r.c.a.sha(source)
    shutil.copyfile(source,checkpoint)
    assert before==r.c.a.sha(source)==r.c.a.sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=r.model_from(checkpoint).eval()
    code=r.ART/f'{args.name}_source.py'
    shutil.copyfile(Path(__file__),code)
    report=dict(checkpoint_sha256=before,step=saved['step'],source_sha256=r.c.a.sha(code),populations={})
    pairs=np.load(r.ART/('complete_training_certificate_pairs.npy' if args.complete else 'training_all_certificate_pairs.npy'))
    records=np.zeros((len(pairs),2),dtype=compact.DTYPE)
    offset=0
    for name,spec in json.loads((r.HERE/'TRAINING_DATA.json').read_text())['sources'].items():
        selected=(pairs[:,:2]>=offset)&(pairs[:,:2]<offset+spec['rows'])
        if selected.any():
            data=compact.load(r.HERE/f'cache/data/{name}.bin')
            records[selected]=data[pairs[:,:2][selected]-offset]
        offset+=spec['rows']
    assert (records['n']>0).all()
    data=encoded(records.reshape(-1),pairs[:,2].astype(bool).repeat(2))
    metric,arrays=measure(model,data,pairs[:,3].repeat(2))
    report['populations']['training_all_certificate_pairs']=metric
    strict=np.load(r.ART/('complete_training_rank_indices.npy' if args.complete else 'training_all_certificate_rank_indices.npy'))
    metric['strict_both_all_supplied']=float(((arrays['errors'][2*strict]==0)&(arrays['errors'][2*strict+1]==0)).mean())
    metric['strict_both_recorded']=float((arrays['recorded_good'][2*strict]&arrays['recorded_good'][2*strict+1]).mean())
    legacy_rows=json.loads((r.HERE/'TRAINING_DATA.json').read_text())['sources']['legacy_train']['rows']
    legacy=pairs[:,0]<legacy_rows
    for name,include in [('legacy',legacy),('new',~legacy)]:
        if include.any():
            metric[name+'_pairs']=int(include.sum())
            metric[name+'_both_all_supplied']=float(((arrays['errors'][::2]==0)&(arrays['errors'][1::2]==0))[include].mean())
    np.savez_compressed(r.ART/f'{args.name}_training.npz',**arrays)
    for name,data in [('new_validation',dict(np.load(r.ART/'hard_validation.npz'))),
                      ('historical',r.c.load('matched_pairs'))]:
        metric,arrays=measure(model,data)
        report['populations'][name]=metric
        np.savez_compressed(r.ART/f'{args.name}_{name}.npz',**arrays)
    r.write(f'{args.name}.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--complete',action='store_true')
    args=parser.parse_args()
    pending=r.HERE/f'GPU_{args.name}_PENDING.json'
    if pending.exists():raise RuntimeError('This diagnostic already ran')
    torch.set_num_threads(1)
    start=time.monotonic();status='failed_or_partial'
    try:
        main(args);status='completed'
    finally:
        elapsed=time.monotonic()-start
        r.write(pending.name,dict(name=f'{args.name}_fit_diagnostic',status=status,
                    wall_seconds=elapsed,gpu_seconds_charged=elapsed,
                    gpu_charge_scope='Entire process wall time; concurrent training also charges its full elapsed time',
                    exact_queries=0,accounting='Deferred append until the concurrent training ledger is closed'))
