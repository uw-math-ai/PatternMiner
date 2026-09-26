"""Original E031 deterministic depth-two fitting and prediction, unchanged."""

def fit(rows,weights,depth=0):
    totals=[sum(weights[r['label']] for r in rows if r['label']==y) for y in (0,1)]
    label=int(totals[1]>totals[0])
    if depth==2 or not all(totals):return dict(label=label,counts=[sum(r['label']==y for r in rows) for y in (0,1)])
    def cost(rs):
        ts=[sum(weights[r['label']] for r in rs if r['label']==y) for y in (0,1)]
        return 0 if sum(ts)==0 else 2*ts[0]*ts[1]/sum(ts)
    best=None
    for f in (0,1):
        vals=sorted({r['x'][f] for r in rows})
        for a,b in zip(vals,vals[1:]):
            threshold=(a+b)/2;left=[r for r in rows if r['x'][f]<=threshold];right=[r for r in rows if r['x'][f]>threshold]
            candidate=(cost(left)+cost(right),f,threshold)
            if best is None or candidate<best[0]:best=(candidate,left,right)
    if best is None:return dict(label=label)
    (score,f,t),left,right=best
    return dict(feature=f,threshold=t,left=fit(left,weights,depth+1),right=fit(right,weights,depth+1))


def predict(tree,x,root_patch=None):
    if 'label' in tree:return tree['label']
    branch=x[tree['feature']]<=tree['threshold'] if root_patch is None else root_patch
    return predict(tree['left'] if branch else tree['right'],x)
