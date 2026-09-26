"""Original E031 prediction and metric functions, unchanged."""

def prediction(model,x,activation=None):
    branch=x[model['feature']]<=model['threshold'] if activation is None else activation
    return model['left_label'] if branch else 1-model['left_label']


def metrics(rows,predict):
    confusion=[[0,0],[0,0]]
    for r in rows:confusion[r['label']][predict(r['x'])]+=1
    n=len(rows)
    return dict(n=n,confusion=confusion,correct=sum(confusion[i][i] for i in (0,1)),accuracy=sum(confusion[i][i] for i in (0,1))/n if n else None)
