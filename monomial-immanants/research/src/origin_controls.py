"""Refit E031 on its frozen labeled corpus; no regeneration of the shape census."""
import hashlib,json
from pathlib import Path
from origin_stump import prediction,metrics
from origin_tree import fit,predict

HERE=Path(__file__).resolve().parents[1]/'artifacts/historical'
def load(name):return json.loads((HERE/name).read_text())

def fit_stump(rows):
    totals=[sum(r['label']==c for r in rows) for c in (0,1)]
    candidates=[]
    for feature in (0,1):
        values=sorted({r['x'][feature] for r in rows})
        for a,b in zip(values,values[1:]):
            for left in (0,1):
                model=dict(feature=feature,threshold=(a+b)/2,left_label=left)
                error=sum((prediction(model,r['x'])!=r['label'])/totals[r['label']] for r in rows)/2
                candidates.append((error,feature,model['threshold'],left,model))
    return min(candidates,key=lambda c:c[:4])[-1]

def check_interventions(model,records,tree=False):
    for row in records:
        x=[row['target'][1],row['target'][6]]
        source=[row['source'][1],row['source'][6]]
        feature=model['feature']
        fn=predict if tree else prediction
        before=fn(model,x)
        own=x[feature]<=model['threshold']
        patch=source[feature]<=model['threshold']
        assert before==row['before'] and fn(model,x,own)==row['identity_patch']==before
        key='source_root_patch' if tree else 'branch_patch'
        assert fn(model,x,patch)==row[key]
        control=list(x);control[1-feature]=source[1-feature]
        key='inactive_root_input_prediction' if tree else 'unused_coordinate_control'
        assert fn(model,control)==row[key]

def main():
    old=load('origin_stump.json');revised=load('origin_tree.json')
    pilot=load('origin_pilot.json')
    genuine={tuple(t) for t in load('origin_genuine_targets.json')}
    assert len(genuine)==1498
    base=old['rows'];hard=old['hard_rows']
    for row in base+hard:
        assert row['x']==[row['target'][1],row['target'][6]]
        assert (tuple(row['target']) in genuine)==bool(row['label'])
    for row in base:
        if row['label']:
            expected='test' if int(hashlib.sha256(str(tuple(row['target'])).encode()).hexdigest(),16)%3==0 else 'train'
        else:expected='test' if row['group']%3==0 else 'train'
        assert row['split']==expected
    train=[r for r in base if r['split']=='train']
    test=[r for r in base if r['split']=='test']
    assert all(r['group']%3==0 for r in hard)
    stump=fit_stump(train)
    assert stump==old['model']==dict(feature=0,threshold=37.0,left_label=0)
    for name,rows in [('train',train),('test',test),('hard',hard)]:
        assert metrics(rows,lambda x:prediction(stump,x))==old[name]
    check_interventions(stump,old['interventions'])
    parents={tuple(r['target'][1:]):r['group'] for r in base if not r['label']}
    added=[]
    for row in pilot['records']:
        group=parents[tuple(row['parent'][1:])]
        if group%3==1 and len(row['components'])>1:
            t=[24]+row['target'][1:];assert tuple(t) not in genuine
            added.append(dict(x=[t[1],t[6]],label=0,group=group,target=t))
    assert len(added)==revised['added_training_rows']==12
    assert {r['group'] for r in added}.isdisjoint({r['group'] for r in hard})
    train2=train+added
    weights={y:1/sum(r['label']==y for r in train2) for y in (0,1)}
    tree=fit(train2,weights)
    assert tree==revised['tree']
    for name,rows in [('train',train2),('test',test),('hard',hard)]:
        assert metrics(rows,lambda x:predict(tree,x))==revised[name]
    assert [r for r in test+hard if predict(tree,r['x'])!=r['label']]==revised['remaining_test_errors']
    check_interventions(tree,revised['interventions'],True)
    child=tree['left']
    for row in revised['child_interventions']:
        x=[row['target'][1],row['target'][6]];src=[row['source'][1],row['source'][6]]
        assert x[tree['feature']]<=tree['threshold']
        f=child['feature'];before=predict(child,x)
        assert before==row['before']==row['identity_patch']
        assert predict(child,x,x[f]<=child['threshold'])==before
        assert predict(child,x,src[f]<=child['threshold'])==row['genuine_child_branch_patch']
        control=list(x);control[1-f]=src[1-f]
        assert predict(child,control)==row['unused_child_input_control']==before
    assert not old['reliability_gate_passed'] and not revised['reliability_gate_passed']
    assert [old[k]['correct'] for k in ('train','test','hard')]==[550,279,0]
    assert [revised[k]['correct'] for k in ('train','test','hard')]==[562,278,14]
    print(json.dumps(dict(status='PASS',stump=stump,tree=tree,
        initial={k:old[k] for k in ('train','test','hard')},
        revised={k:revised[k] for k in ('train','test','hard')},
        added_training_rows=12,stump_interventions=len(old['interventions']),
        tree_root_interventions=len(revised['interventions']),tree_child_interventions=len(revised['child_interventions']),
        reliability_gate_passed=False,
        scope='Fresh deterministic refitting and branch replay on frozen labels; original universal shape census and obstruction search are not regenerated. Hard assessment was reused after inspection; no new generalization claim.'),indent=2))

if __name__=='__main__':main()
