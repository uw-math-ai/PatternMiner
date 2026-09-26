"""Read-only E072 saved-record checks, without importing scientific modules."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'research/e072-beta221'
ARTIFACT_SHA = '8c16d0e187f3da83eade2e31f366fb3b624836d621fc8e6de92af7e9f8f9482d'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    raw = (PACK/'result.json.gz').read_bytes()
    assert sha(raw) == ARTIFACT_SHA
    assert (PACK/'result.sha256').read_text() == ARTIFACT_SHA+'  result.json.gz\n'
    record = json.loads(gzip.decompress(raw))
    assert record['status'] == 'completed' and record['gate_passed'] is False
    exported = json.loads((PACK/'EXPORT.json').read_text())['files']
    pinned = 0
    for entry in exported:
        assert sha((ROOT/entry['destination']).read_bytes()) == entry['published_sha256']
        if entry['mode'] == 'verbatim':
            assert entry['source_sha256'] == entry['published_sha256']
        else:
            assert entry['mode'] == 'publication-path-normalization'
        if entry['source'] in record['pins']:
            assert entry['source_sha256'] == record['pins'][entry['source']]
            pinned += 1
    groups = record['groups']
    assert len(groups) == 30 and len({tuple(g['output']) for g in groups}) == 30
    assert len(record['feature_names']) == 47
    assert all(sum(g['output']) == 10 and g['output'][0] <= 5 for g in groups)

    def evaluate(gs, weights):
        predictions = []
        for g in gs:
            assert len(g['features']) == len(g['negative'])
            choices = []
            for row in g['features']:
                assert len(row) == len(g['positive'])
                assert all(len(f) == len(weights) for f in row)
                scores = [sum(a*b for a,b in zip(f,weights)) for f in row]
                choices.append(scores.index(max(scores)))
            predictions.append(choices)
        return {'predictions': predictions, 'sources': sum(map(len, predictions)),
                'groups': len(gs), 'collisions': sum(len(p)-len(set(p)) for p in predictions),
                'injective_groups': sum(len(p) == len(set(p)) for p in predictions)}

    domains = {s: [] for s in ('training', 'validation', 'test')}
    collisions = []
    w = record['selected_weights']
    for g in groups:
        slot = int(sha(('E072:'+','.join(map(str,g['output']))).encode()),16)%5
        expected = 'validation' if slot == 0 else 'test' if slot == 1 else 'training'
        assert expected == g['split']
        if g['negative']:
            domains[expected].append(g)
        ns, ps = g['negative'], g['positive']
        assert len({json.dumps(x,sort_keys=True) for x in ns}) == len(ns)
        assert len({json.dumps(x,sort_keys=True) for x in ps}) == len(ps)
        assert len(ps)-len(ns) == g['coefficient']
        pairs = g['lookup_pairs']
        assert [p['source'] for p in pairs] == list(range(len(ns)))
        images = {p['image'] for p in pairs}
        assert len(images) == len(ns) and all(p['inverse'] == p['source'] for p in pairs)
        assert images.isdisjoint(g['lookup_survivors'])
        assert images|set(g['lookup_survivors']) == set(range(len(ps)))
        assert len(g['lookup_survivors']) == g['coefficient']
        choices = evaluate([g],w)['predictions'][0]
        for i,j in itertools.combinations(range(len(choices)),2):
            if choices[i] == choices[j]:
                collisions.append({'output':g['output'],'split':expected,
                                   'source_a':i,'source_b':j,'image':choices[i]})
    assert collisions == record['collision_witnesses']
    for s,gs in domains.items():
        assert evaluate(gs,w) == record['selected'][s]
    for m in record['models']:
        for s in ('training','validation'):
            assert evaluate(domains[s],m['weights']) == m[s]
    chosen = min(record['models'],key=lambda m:(m['validation']['collisions'],
                 m['training']['collisions'],sum(map(abs,m['weights'])),m['seed']))
    assert chosen['seed'] == record['selected_seed'] and chosen['weights'] == w
    for a in record['ablations']:
        for s in ('validation','test'):
            assert evaluate(domains[s],a['removed_weights']) == a[s]
    for b in record['random_baselines']:
        for s,gs in domains.items():
            computed = evaluate(gs,b['weights'])
            del computed['predictions']
            assert computed == b['scores'][s]
    assert sum(len(g['positive']) for g in groups) == 97
    assert sum(len(g['negative']) for g in groups) == 50
    assert sum(g['coefficient'] for g in groups) == 47
    assert [record['selected'][s]['collisions'] for s in domains] == [0,1,5]
    return {'status':'passed saved-record checks', 'ordinary_targets':30,
            'available_historical_pins_checked':pinned,'total_historical_pins':len(record['pins']),
            'learned_gate_passed':False,'scientific_reruns':0}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
