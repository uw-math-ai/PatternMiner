"""Reserve a replacement assessment before adapting to the first S10 failure."""
import gzip,json,random
from pathlib import Path
import numpy as np
import torch
import session as s
from session import c
import compact
import plan_neighborhoods as pn
from full_resources import Budget


def main(budget):
    assert not (s.HERE/'SECOND_DATA_PLAN.json').exists()
    failed=json.loads((s.HERE/'FRESH_ASSESSMENT.json').read_text())
    assert failed['status']=='completed' and not failed['passes']
    with gzip.open(s.ART/'forbidden_factor_orbits.json.gz','rt') as stream:
        seen={tuple(tuple(p) for p in pair) for pair in json.load(stream)}
    for spec in json.loads((s.HERE/'DATA_PLAN.json').read_text())['populations'].values():
        seen.update(pn.orbit(pair) for pair in spec['pairs'])
    seen.update(pn.orbit(pair) for pair in json.loads((s.HERE/'NEIGHBOR_PLAN.json').read_text())['pairs'])
    initial_count=len(seen);populations={};cost=0
    for n,count in {7:30,8:30,9:100,10:240}.items():
        rng=random.Random(s.SEED+81000+n);cap={7:7,8:9,9:11,10:13}[n];pairs=[]
        def sample():
            while True:
                p=list(range(1,n+1));rng.shuffle(p)
                if 1<=c.prior.c.a.original.length(p)<=cap:return p
        while len(pairs)<count:
            budget.check();pair=dict(u=sample(),v=sample());key=pn.orbit(pair)
            if key in seen:continue
            seen.add(key);pairs.append(pair)
        calls=sum(len(c.prior.c.a.original.desc(c.prior.c.a.original.inv(tuple(pair[k])))) for pair in pairs for k in ['u','v'])
        cost+=calls;populations[f'second_S{n}_assessment']=dict(n=n,cap=cap,pairs=pairs,descent_product_calls=calls)
    assert cost<3600
    s.write('SECOND_DATA_PLAN.json',dict(status='frozen_before_labels',populations=populations,
        planned_descent_product_calls=cost,forbidden_orbits=initial_count,source_sha256=s.sha(Path(__file__)),
        predecessor_failure_sha256=s.sha(s.HERE/'FRESH_ASSESSMENT.json'),
        sampling='Same uniform conditional factor sampling and length caps as the first assessment. Exclude all old, new development, first-assessment and training factor orbits before label generation.',
        assessment_policy='Open once after all twenty-seven development criteria pass .99. Freeze model before predictions; retain every positive supplied index, both orientations and both members of difficult pairs.'))
    # Only now inspect the exposed S10 failures to choose disjoint training neighbors.
    source=s.HERE/'cache/data/S10_assessment.bin';records=compact.load(source)
    archive=s.ART/'fresh_assessment_S10_assessment.npz'
    with np.load(archive) as result:bad=result['pairs'][~result['pair_good']].copy()
    anchors={}
    for pair in bad:
        record=records[pair[0]];n=int(record['n']);perms=record['permutations']
        factors=pn.normalized(dict(u=perms[0,:n].tolist(),v=perms[1,:n].tolist()));key=pn.orbit(factors)
        anchors.setdefault(key,dict(pair=factors,failed_pairs=[]))['failed_pairs'].append(pair.tolist())
    anchors=sorted(anchors.values(),key=lambda v:(v['pair']['u'],v['pair']['v']))
    pools=[pn.candidates(item['pair'],seen,s.SEED+83000+i) for i,item in enumerate(anchors)]
    allowance=4827-cost;cursor=[0]*len(pools);selected=[];chosen=set();training_cost=0
    while True:
        progress=False
        for parent,pool in enumerate(pools):
            while cursor[parent]<len(pool):
                pair=pool[cursor[parent]];cursor[parent]+=1;key=pn.orbit(pair)
                if key in chosen or training_cost+pair['exact_query_cost']>allowance:continue
                chosen.add(key);training_cost+=pair['exact_query_cost'];selected.append(dict(parent=parent,**pair));progress=True;break
        if not progress:break
    assert selected and not chosen&seen and cost+training_cost<=4827
    s.write('SECOND_NEIGHBOR_PLAN.json',dict(status='frozen_before_labels',anchors=anchors,pairs=selected,groups=len(selected),
        exact_query_cost=training_cost,forbidden_orbits=len(seen),source_sha256=s.sha(Path(__file__)),
        second_assessment_plan_sha256=s.sha(s.HERE/'SECOND_DATA_PLAN.json'),exposed_failures_sha256=s.sha(archive),
        policy='Round robin over the newly exposed failure factor neighborhoods. No development rows enter training. All first and replacement assessment factor orbits remain excluded.'))
    print(json.dumps(dict(assessment_queries=cost,training_queries=training_cost,anchors=len(anchors),neighbor_groups=len(selected))),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('freeze_second_populations',180);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
