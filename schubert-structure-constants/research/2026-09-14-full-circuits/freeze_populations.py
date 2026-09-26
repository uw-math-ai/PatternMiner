"""Reserve new assessment and development factors before further adaptive data."""
import gzip
import json
from pathlib import Path
import random
import torch
import session as s
from session import c
import adaptive_data
import plan_neighborhoods as pn
from full_resources import Budget


def main(budget):
    assert not (s.HERE/'DATA_PLAN.json').exists()
    seen=adaptive_data.forbidden_orbits(budget)
    for name in ['ADAPTIVE_PLAN.json','SCREENED_PLAN.json','LATE_PLAN.json']:
        seen.update(pn.orbit(pair) for pair in json.loads((s.PREVIOUS/name).read_text())['pairs'])
    forbidden=len(seen)
    archive=s.ART/'forbidden_factor_orbits.json.gz'
    with archive.open('wb') as out:
        with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as stream:
            stream.write(json.dumps(sorted(seen),separators=(',',':')).encode())
    populations={};cost=0
    sizes={'assessment':{7:100,8:100,9:320,10:240},'validation':{9:100,10:80}}
    caps={7:7,8:9,9:11,10:13}
    for phase,counts in sizes.items():
        for n,count in counts.items():
            rng=random.Random(s.SEED+1000*n+list(sizes).index(phase));pairs=[]
            def sample():
                while True:
                    values=list(range(1,n+1));rng.shuffle(values)
                    if 1<=c.prior.c.a.original.length(values)<=caps[n]:return values
            while len(pairs)<count:
                budget.check();pair=dict(u=sample(),v=sample());key=pn.orbit(pair)
                if key in seen:continue
                seen.add(key);pairs.append(pair)
            calls=sum(len(c.prior.c.a.original.desc(c.prior.c.a.original.inv(tuple(pair[k])))) for pair in pairs for k in ['u','v'])
            populations[f'S{n}_{phase}']=dict(n=n,cap=caps[n],pairs=pairs,descent_product_calls=calls)
            cost+=calls
    assert cost<=8000
    report=dict(status='frozen_before_labels',seed=s.SEED,populations=populations,
        planned_descent_product_calls=cost,forbidden_orbits=forbidden,forbidden_archive_sha256=s.sha(archive),
        source_sha256=s.sha(Path(__file__)),predecessor_manifest_sha256=s.sha(s.PREVIOUS/'EVIDENCE_MANIFEST.json'),
        sampling='Uniform permutations conditional on positive length within the preceding caps; canonical factor orbits excluded across all old and new populations before label generation.',
        assessment_policy='Open once only after every declared development criterion reaches 99%; every supplied index, both orientations and both difficult-pair members must pass.',
        protected_circuit_populations='Original discovery and confirmation populations remain unopened; included in historical factor exclusions')
    s.write('DATA_PLAN.json',report)
    print(json.dumps(dict(queries=cost,forbidden_orbits=forbidden,populations={name:len(v['pairs']) for name,v in populations.items()})),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('freeze_fresh_factor_populations',180);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
