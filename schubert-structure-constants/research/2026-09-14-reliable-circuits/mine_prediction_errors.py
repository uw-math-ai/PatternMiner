"""Find confidently incorrect witness choices inside the training split."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import common as r
import compact
import training_sampling
from train_expanded import training_tensors


def mine(model,tr,offsets,weights,seed,budget,samples=200000,cap=16384,source_layouts=None):
    generator=torch.Generator(device='cuda').manual_seed(seed)
    chunks=[]
    for source,weight in enumerate(weights):
        count=round(samples*weight) if source<len(weights)-1 else samples-sum(len(z) for z in chunks)
        chunks.append(training_sampling.sample(source,count,offsets,generator,source_layouts))
    indices=torch.cat(chunks)
    reverse=torch.rand(samples,device='cuda',generator=generator)<.5
    records=[];margins=[];inputs=0
    model.eval()
    with torch.inference_mode():
        for begin in range(0,len(indices),1024):
            budget.check()
            idx=indices[begin:begin+1024]
            rev=reverse[begin:begin+1024]
            x,mask,y,aa=compact.encode({key:value[idx] for key,value in tr.items()},rev)
            example,supplied=aa.nonzero(as_tuple=True)
            for start in range(0,len(example),1024):
                row=example[start:start+1024];ii=supplied[start:start+1024]
                xx=r.c.a.change_witness(x[row],ii)
                scores=model(xx).masked_fill(~mask[row],-1e9)
                assert torch.isfinite(scores).all()
                pred=scores.argmax(1)
                wrong=~y[row,pred]
                inputs+=len(row)
                if wrong.any():
                    margin=scores.max(1).values-scores.masked_fill(~y[row],-1e9).max(1).values
                    records.append(torch.stack([idx[row[wrong]],rev[row[wrong]].long(),ii[wrong],pred[wrong]],1))
                    margins.append(margin[wrong])
    if records:
        records=torch.cat(records).cpu().numpy()
        margins=torch.cat(margins).cpu().numpy()
        unique,first=np.unique(records,axis=0,return_index=True)
        raw_errors=len(records);records=unique;margins=margins[first]
        distinct=len(records)
        keep=np.argsort(-margins,kind='stable')[:cap]
        records,margins=records[keep],margins[keep]
    else:
        records=np.zeros((0,4),np.int64);margins=np.zeros(0,np.float32)
        raw_errors=distinct=0
    return records,margins,dict(sampled_oriented_triples=samples,supplied_inputs=inputs,
                                incorrect_inputs=raw_errors,distinct_incorrect_inputs=distinct,
                                retained=len(records),seed=seed,cap=cap)


class Limit:
    def __init__(self):self.start=time.monotonic()
    def check(self):
        if time.monotonic()-self.start>300:raise TimeoutError('Initial error-mining reservation exhausted')


def main(args,budget):
    source=Path(args.checkpoint)
    checkpoint=r.ART/f'{args.name}.pt'
    if checkpoint.exists():raise FileExistsError(checkpoint)
    before=r.c.a.sha(source);shutil.copyfile(source,checkpoint)
    assert before==r.c.a.sha(source)==r.c.a.sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=r.model_from(checkpoint).eval()
    tr,eligible,offsets,weights=training_tensors()
    records,margins,report=mine(model,tr,offsets,weights,r.SEED+7721,budget,args.samples)
    artifact=r.ART/f'{args.name}_errors.npz'
    np.savez_compressed(artifact,records=records,margins=margins)
    source_code=r.ART/f'{args.name}_source.py'
    shutil.copyfile(Path(__file__),source_code)
    names=list(json.loads((r.HERE/'TRAINING_DATA.json').read_text())['sources'])
    report.update(checkpoint_sha256=before,step=saved['step'],source_sha256=r.c.a.sha(source_code),
                  training_spec_sha256=r.c.a.sha(r.HERE/'TRAINING_DATA.json'),
                  artifact_sha256=r.c.a.sha(artifact),
                  retained_by_source={name:int(((records[:,0]>=offsets[i])&(records[:,0]<offsets[i+1])).sum()) for i,name in enumerate(names)},
                  columns=['global_training_row','reverse','supplied_index_zero_based','incorrect_predicted_index_zero_based'],
                  scope='Training data only; every valid supplied index is checked; labels are reused from retained exact products, with no new queries')
    r.write(f'{args.name}.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--samples',type=int,default=200000)
    args=parser.parse_args()
    pending=r.HERE/f'GPU_{args.name}_PENDING.json'
    if pending.exists():raise RuntimeError('Error mining already ran')
    torch.set_num_threads(1);budget=Limit();status='failed_or_partial'
    try:
        main(args,budget);status='completed'
    finally:
        elapsed=time.monotonic()-budget.start
        r.write(pending.name,dict(name=args.name,status=status,wall_seconds=elapsed,gpu_seconds_charged=elapsed,
                    gpu_charge_scope='Entire process wall time; concurrent training also charges its full elapsed time',
                    exact_queries=0,accounting='Deferred append until the concurrent training ledger is closed'))
