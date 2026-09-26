"""Prospective witness selection, after the coefficient inequality was proposed.

Given one exact ordinary-support witness A_i>0, rank possible opposite
deletions j. Labels are B_j>0. No coefficients, root coefficients, or output
labels are input features. Evaluation groups all outputs and both factor
orientations by the unordered stable input pair.
"""
from degree_one import *
import gzip
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier

WIDTH=16;DIM=22
torch.set_num_threads(4)

def group(x):return tuple(sorted((trim(x['u']),trim(x['v']))))
def bucket(g):return int(hashlib.sha256(('witness-v1:'+str(g)).encode()).hexdigest()[:12],16)%10
def read(name):
    with gzip.open(HERE/name,'rt') as f:return [json.loads(s) for s in f]

def encode(row,reverse=False,last=False):
    u,v,w=map(tuple,(row['u'],row['v'],row['w']));a,b=row['A'],row['B']
    if reverse:u,v=v,u;a,b=b,a
    active=[j for j,c in enumerate(a) if c];i=active[-1] if last else active[0]
    n=len(w);assert n-1<=WIDTH
    iu,iv,iw=inv(u),inv(v),inv(w);xx=np.zeros((WIDTH,DIM),np.float32);mask=np.zeros(WIDTH,bool);yy=np.zeros(WIDTH,bool)
    for j in range(n-1):
        xs=[(j+1)/n,(i+1)/n,(j-i)/n]
        xs += [p[k]/n for p in (iu,iv,iw) for k in (j,j+1)]
        xs += [p[k]/n for p in (iu,iv,iw) for k in (i,i+1)]
        xs += [float(iu[j]>iu[j+1]),float(iw[j]>iw[j+1]),float(i==j),1/n]
        xs += [length(p)/(n*n) for p in (u,v,w)]
        assert len(xs)==DIM
        xx[j]=xs;mask[j]=iv[j]>iv[j+1];yy[j]=b[j]>0
    assert np.any(yy) and np.all(~yy|mask)
    return xx,mask,yy

def arrays(rows,last=False):
    values=[encode(row,rev,last) for row in rows for rev in (False,True)]
    return tuple(np.stack([r[i] for r in values]) for i in range(3))

def metrics(score,mask,y):
    score=np.where(mask,score,-1e20);order=np.argsort(-score,axis=1,kind='stable')
    good=np.take_along_axis(y,order,axis=1);calls=np.argmax(good,axis=1)+1
    return dict(n=len(y),top1=float(good[:,0].mean()),mean_oracle_calls=float(calls.mean()),
                random_expected_calls=float(((mask.sum(1)+1)/(y.sum(1)+1)).mean()))

class Selector(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(DIM,96)
        layer=nn.TransformerEncoderLayer(96,4,192,.1,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Linear(96,1)
    def forward(self,x):
        return self.head(self.encoder(self.input(x),src_key_padding_mask=x.abs().sum(-1)==0)).squeeze(-1)

def main(seed,steps):
    start=time.time();rng=random.Random(20260908)
    names=['stable_degree_one_S7_3000_length7.jsonl.gz','stable_degree_one_S8_2000_length9.jsonl.gz',
           'stable_degree_one_S9_500_length11.jsonl.gz']
    source=read(names[0]);allgroups={group(x) for x in source}
    rows={'train':[x for x in source if bucket(group(x))<7],
          'validation':[x for x in source if bucket(group(x))==7],
          'new_products':[x for x in source if bucket(group(x))>=8]}
    for name,label,cap in zip(names[1:],['new_inputs_S8','new_inputs_S9'],[15000,10000]):
        z=[x for x in read(name) if group(x) not in allgroups]
        rows[label]=rng.sample(z,min(cap,len(z)))
    sets={name:{group(x) for x in rr} for name,rr in rows.items()}
    assert not (sets['train']&sets['validation'] or sets['train']&sets['new_products'] or sets['validation']&sets['new_products'])
    assert not sets['train']&sets['new_inputs_S8'] and not sets['train']&sets['new_inputs_S9']
    data={name:arrays(rr) for name,rr in rows.items()}
    out=dict(seed=seed,steps=steps,task='select opposite ordinary-support witness',
        discovery_origin='Conjecture and proof attempt preceded training; no mechanistic discovery claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names},
        split={name:dict(examples=len(xx),unordered_factor_pairs=len(sets[name])) for name,(xx,_,_) in data.items()},
        configuration=dict(width=WIDTH,feature_count=DIM,layers=2,hidden=96,heads=4,feedforward=192,
                           batch_size=256,lr=.0005,weight_decay=.01,dropout=.1),models={})
    # A fixed mathematical baseline: try left descents nearest the supplied i.
    for name,(xx,mask,y) in data.items():
        if name=='train':continue
        score=-np.abs(xx[:,:,2])-np.arange(WIDTH)[None,:]*1e-7
        out['models'].setdefault('nearest_index',{})[name]=metrics(score,mask,y)
    x,mask,y=data['train'];candidates=np.argwhere(mask)
    if len(candidates)>250000:candidates=candidates[np.random.default_rng(20260908).choice(len(candidates),250000,replace=False)]
    gbm=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.08,random_state=seed)
    gbm.fit(x[candidates[:,0],candidates[:,1]],y[candidates[:,0],candidates[:,1]])
    for name,(xx,mm,yy) in data.items():
        if name=='train':continue
        score=gbm.predict_proba(xx.reshape(-1,DIM))[:,1].reshape(xx.shape[:2])
        out['models'].setdefault('gradient_boosting',{})[name]=metrics(score,mm,yy)
    print(json.dumps(dict(baselines=out['models'])),flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model=Selector().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    out['parameter_count']=sum(p.numel() for p in model.parameters());out['device']=device
    tr=tuple(torch.as_tensor(z,device=device) for z in data['train']);gen=torch.Generator(device=device).manual_seed(seed)
    best=-1;beststate=None;history=[]
    def predict(xx):
        model.eval()
        with torch.no_grad():return np.concatenate([model(torch.as_tensor(xx[k:k+512],device=device)).cpu().numpy() for k in range(0,len(xx),512)])
    for step in range(1,steps+1):
        model.train();ind=torch.randint(len(tr[0]),(256,),generator=gen,device=device)
        xx,mm,yy=(z[ind] for z in tr);score=model(xx).masked_fill(~mm,-1e9)
        loss=(torch.logsumexp(score,dim=1)-torch.logsumexp(score.masked_fill(~yy,-1e9),dim=1)).mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%200==0 or step==steps:
            vx,vm,vy=data['validation'];m=metrics(predict(vx),vm,vy);history.append(dict(step=step,loss=float(loss.detach()),validation=m))
            if m['top1']>best:best=m['top1'];beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};out['best_step']=step
            print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(beststate)
    hard=[]
    for name,(xx,mm,yy) in data.items():
        if name=='train':continue
        score=predict(xx);out['models'].setdefault('transformer',{})[name]=metrics(score,mm,yy)
        np.savez_compressed(HERE/f'witness_predictions_seed{seed}_{name}.npz',scores=score,mask=mm,labels=yy)
        if name=='new_inputs_S9':
            pred=np.where(mm,score,-1e20).argmax(1)
            for k in np.flatnonzero(~yy[np.arange(len(yy)),pred])[:20]:
                hard.append(dict(row=rows[name][int(k)//2],reverse=bool(k%2),proposed_deletion=int(pred[k])+1))
    # Change to another valid supplied witness. This changes inputs but not B.
    xx,mm,yy=arrays(rows['new_inputs_S9'],last=True)
    out['valid_witness_intervention']=metrics(predict(xx),mm,yy)
    out['history']=history;out['hard_examples']=hard;out['seconds']=time.time()-start
    torch.save(dict(state_dict=beststate,config=out['configuration']),HERE/f'witness_model_seed{seed}.pt')
    (HERE/f'witness_model_seed{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(models=out['models'],intervention=out['valid_witness_intervention'],seconds=out['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=1200);a=p.parse_args()
    main(a.seed,a.steps)
