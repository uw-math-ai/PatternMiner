"""Bounded origin-sensitive matching pilot; no external services or torch."""
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.budget import RunBudget, retained_bytes
from src.e015_cyclic_orbit_morse import wall_cap

PROTOCOL = 'experiments/E072-beta221-matching-pilot.md'
INPUTS = [PROTOCOL,
    'experiments/E072-root-preflight.md',
    'research-notes/P12-tail-ten-cubic-length-five-beta221-working-route.md',
    'research-notes/P12-beta221-ordinary-negative-corner-domain.md',
    'experiments/P12-beta221-ordinary-negative-corner-domain-static-review.md',
    'experiments/P12-tail-ten-cubic-length-five-beta221-static-review.md']
BASES = ((4, 2), (3, 2, 1), (2, 2, 2), (3, 1, 1, 1))
CONTENTS = ((2, 2), (2, 1, 1))
TAGS = ('D', 'E', 'F', 'G', 'H')
NAMES = ['cell_overlap', 'entry_agreement', 'negative_entry_disagreement',
         'corner_in_image', 'corner_entry_1', 'corner_entry_2', 'corner_entry_3']
NAMES += [f'{t}_base{j}_content{k}' for t in TAGS for j in range(4) for k in range(2)]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(obj):
    return json.dumps(obj, separators=(',', ':'), sort_keys=True)


def cells(inner, outer):
    return tuple((r+1, c) for r, width in enumerate(outer)
                 for c in range(width, (inner[r] if r < len(inner) else 0), -1))


def brute_lr(inner, content, outer):
    if len(inner) > len(outer) or any(a > b for a, b in zip(inner, outer)):
        return ()
    positions = cells(inner, outer)
    letters = tuple(i+1 for i, count in enumerate(content) for _ in range(count))
    result = []
    for word in sorted(set(itertools.permutations(letters))):
        filling = dict(zip(positions, word))
        if len(positions) != len(word):
            continue
        if any(filling.get((r, c+1), v) < v or
               filling.get((r+1, c), v+1) <= v
               for (r, c), v in filling.items()):
            continue
        if any(word[:i].count(k) < word[:i].count(k+1)
               for i in range(1, len(word)+1) for k in range(1, len(content))):
            continue
        result.append((positions, word))
    return tuple(result)


def features(neg, pos):
    a = dict(zip(map(tuple, neg['cells']), neg['entries']))
    a[tuple(neg['corner'])] = 1  # Separate factor, used only as a scoring proxy.
    b = dict(zip(map(tuple, pos['cells']), pos['entries']))
    common = set(a) & set(b)
    corner = b.get(tuple(neg['corner']), 0)
    f = [len(common), sum(a[c] == b[c] for c in common),
         -sum(a[c] != b[c] for c in common), int(bool(corner))]
    f += [int(corner == k) for k in (1, 2, 3)]
    f += [int(neg['origin'][0] == t and tuple(pos['base']) == BASES[j]
              and tuple(pos['content']) == CONTENTS[k])
          for t in TAGS for j in range(4) for k in range(2)]
    return f


def split(nu):
    key = 'E072:' + ','.join(map(str, nu))
    slot = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5
    return 'validation' if slot == 0 else 'test' if slot == 1 else 'training'


def choices(group, weights):
    return [max(range(len(row)), key=lambda j: (sum(a*b for a, b in zip(row[j], weights)), -j))
            for row in group['features']]


def evaluate(groups, weights):
    predictions = [choices(g, weights) for g in groups]
    return {'sources': sum(len(p) for p in predictions),
            'groups': len(groups),
            'collisions': sum(len(p)-len(set(p)) for p in predictions),
            'injective_groups': sum(len(p) == len(set(p)) for p in predictions),
            'predictions': predictions}


def objective(groups, w):
    return (evaluate(groups, w)['collisions'], sum(map(abs, w)))


def execute(result, budget):
    from src import p12_lr_degree12 as LR
    from src import p12_nonhook_three_cube as OLD
    from src.e005_full_rows_cancellation import partitions
    from src.e004 import BetaMurnaghanNakayama
    from src.e040_sixth_surplus_primitive_tables import schur_jt, plethysm
    from src.e061_tail_eight_three_short_boundaries import hall, scalar_mn
    # Complete original primitive palette, reconstructed by its direct predicate.
    old = []
    for mu in partitions(9):
        for obj in OLD.positive_objects(mu):
            if not OLD.is_survivor(mu, obj):
                continue
            tag = obj[0]
            _, inner, kind = next(x for x in OLD.POSITIVE_SPECS if x[0] == tag)
            content = (2, 1) if kind == 'lr21' else (1, 1, 1) if kind == 'vertical' else (3,)
            candidates = brute_lr(inner, content, mu)
            if kind == 'lr21':
                candidates = tuple(x for x in candidates if x[0][x[1].index(2)][0] == obj[1])
            assert len(candidates) == 1
            old.append({'parent': mu, 'origin': obj, 'inner': inner,
                        'cells': candidates[0][0], 'entries': candidates[0][1]})
    assert len(old) == 16
    negatives = []
    for obj in old:
        mu = obj['parent']
        for r in range(len(mu)+1):
            width = mu[r] if r < len(mu) else 0
            if r and mu[r-1] <= width:
                continue
            nu = list(mu)
            if r == len(mu):
                nu.append(1)
            else:
                nu[r] += 1
            negatives.append(dict(obj, output=tuple(nu), corner=(r+1, width+1)))
    assert len(negatives) == 56
    assert sum(n['output'][0] <= 5 for n in negatives) == 50
    groups = []
    for nu in partitions(10):
        if nu[0] > 5:
            continue
        positives = []
        for base in BASES:
            origin = 'C' if base == (3, 1, 1, 1) else 'B'
            inner = (3, 1) if origin == 'C' else (2, 2)
            content0 = (1, 1) if origin == 'C' else (2,)
            square = brute_lr(inner, content0, base)
            assert len(square) == 1
            for content in CONTENTS:
                generated = LR.lr_tableaux(base, content, nu)
                independent = brute_lr(base, content, nu)
                assert set(generated) == set(independent)
                assert all(LR.is_lr_tableau(base, content, nu, w) for w in independent)
                for cs, word in independent:
                    positives.append({'base': base, 'square_origin': origin,
                                      'square_payload': square[0], 'content': content,
                                      'cells': cs, 'entries': word})
        ns = sorted((n for n in negatives if n['output'] == nu), key=canon)
        positives.sort(key=canon)
        assert len(positives) >= len(ns), ('capacity failure', nu)
        g = {'output': nu, 'split': split(nu), 'negative': ns, 'positive': positives}
        g['features'] = [[features(n, p) for p in positives] for n in ns]
        groups.append(g)
    assert len(groups) == 30
    budget.check()
    # Full, untruncated ambient plethysm; no tail truncation at degree fifteen.
    ambient = plethysm((3,), schur_jt((2, 2, 1)))
    evaluator = BetaMurnaghanNakayama(15)
    for g in groups:
        lam = (5,)+tuple(g['output'])
        count = len(g['positive'])-len(g['negative'])
        mn = scalar_mn(ambient, lam, evaluator)
        jt = hall(ambient, schur_jt(lam))
        assert count == mn == jt, (lam, count, mn, jt)
        g['coefficient'] = count
        g['lookup_pairs'] = [{'source': i, 'image': i, 'inverse': i}
                             for i in range(len(g['negative']))]
        g['lookup_survivors'] = list(range(len(g['negative']), len(g['positive'])))
    result['groups'] = groups
    result['all_negative_objects'] = negatives
    result['exact_checks'] = {'old_survivors': 16, 'negative_full': 56,
                              'negative_ordinary': 50, 'ambient_targets': 30,
                              'LR_families_checked': 240}
    domains = {name: [g for g in groups if g['split'] == name and g['negative']]
               for name in ('training', 'validation', 'test')}
    assert all(domains.values())
    result['feature_names'] = NAMES
    result['baselines'] = {}
    fixed = {'lexicographic': [0]*len(NAMES),
             'geometry': [4, 2, 1, 1, 0, 0, 0]+[0]*(len(NAMES)-7)}
    for name, w in fixed.items():
        result['baselines'][name] = {k: evaluate(v, w) for k, v in domains.items()}
    models = []
    for seed in (1729, 2718, 31415):
        rng = random.Random(seed)
        w = [0]*len(NAMES)
        score = objective(domains['training'], w)
        accepted = []
        for step in range(600):
            trial = w.copy()
            j = rng.randrange(len(w))
            trial[j] = rng.randint(-4, 4) if rng.random() < .1 else max(-4, min(4, trial[j]+rng.choice((-1, 1))))
            new = objective(domains['training'], trial)
            if new < score:
                w, score = trial, new
                accepted.append([step, list(score), w.copy()])
            if step % 100 == 0:
                budget.check()
        models.append({'seed': seed, 'weights': w, 'accepted': accepted,
                       'training': evaluate(domains['training'], w),
                       'validation': evaluate(domains['validation'], w)})
    chosen = min(models, key=lambda m: (m['validation']['collisions'],
                 m['training']['collisions'], sum(map(abs, m['weights'])), m['seed']))
    w = chosen['weights']
    result['models'] = models
    result['selected_seed'] = chosen['seed']
    result['selected_weights'] = w
    result['selected'] = {k: evaluate(v, w) for k, v in domains.items()}
    result['gate_passed'] = all(x['collisions'] == 0 for x in result['selected'].values())
    ablations = []
    for j, value in enumerate(w):
        if value:
            removed = w.copy()
            removed[j] = 0
            ablations.append({'feature': NAMES[j], 'removed_weights': removed,
                              'validation': evaluate(domains['validation'], removed)})
    result['ablations'] = ablations
    result['strongest_validation_ablation'] = (max(ablations, key=lambda a: a['validation']['collisions'])['feature']
                                                if ablations else None)
    for a in ablations:
        a['test'] = evaluate(domains['test'], a['removed_weights'])
    rng = random.Random(20260921)
    result['random_baselines'] = []
    for _ in range(100):
        rw = [rng.randint(-4, 4) for _ in NAMES]
        result['random_baselines'].append({'weights': rw, 'scores': {
            k: {key: val for key, val in evaluate(v, rw).items() if key != 'predictions'}
            for k, v in domains.items()}})
    result['collision_witnesses'] = []
    for g in groups:
        selected = choices(g, w)
        for i, j in itertools.combinations(range(len(selected)), 2):
            if selected[i] == selected[j]:
                result['collision_witnesses'].append({'output': g['output'], 'split': g['split'],
                    'source_a': i, 'source_b': j, 'image': selected[i]})
    # Exact independent re-evaluation of each chosen index and all lookup inverses.
    for g in groups:
        for i, image in enumerate(choices(g, w)):
            scores = [sum(w[k]*f[k] for k in range(len(w))) for f in g['features'][i]]
            assert image == scores.index(max(scores))
        assert len(set(p['image'] for p in g['lookup_pairs'])) == len(g['negative'])
        assert all(g['lookup_pairs'][p['image']]['source'] == p['source'] for p in g['lookup_pairs'])
        assert len(g['lookup_survivors']) == g['coefficient']
    result['status'] = 'completed'


def main():
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    out = ROOT/'artifacts/e072'
    assert not out.exists(), 'One attempt only; preserve previous output'
    paths = sorted((ROOT/'src').glob('*.py'))+[ROOT/p for p in INPUTS]+[Path(__file__).resolve()]
    pins = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert len(list((ROOT/'src').glob('*.py'))) == 120
    assert pins['src/p12_nonhook_three_cube.py'] == 'f17bcfc84cf627521795681ba058c7631f9d9acc8797d5c9cc52ee8e425f59a8'
    assert pins['src/p12_lr_degree12.py'] == '2bd184d5822e0b71d719be205b6b536dec761e0630705325d9f5eef7ae235cbe'
    result = {'pins': pins, 'status': 'started', 'python': sys.version,
              'review_status': 'root-preflight only; independent review pending'}
    with RunBudget(ROOT, 'e072-beta221-matching-pilot-v1') as budget:
        started = time.monotonic()
        before = retained_bytes(ROOT/'artifacts')
        assert before+500000 <= budget.limits['storage_bytes']
        out.mkdir()
        try:
            with wall_cap(85):
                execute(result, budget)
                assert all(digest(ROOT/p) == sha for p, sha in pins.items())
        except BaseException as error:
            result['error'] = repr(error)
            result['status'] = 'failed_or_partial'
            raise
        finally:
            result['body_wall_seconds'] = time.monotonic()-started
            result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            payload = gzip.compress(canon(result).encode(), mtime=0)
            assert len(payload)+81 <= 500000
            target = out/'result.json.gz'
            target.write_bytes(payload)
            (out/'result.sha256').write_text(digest(target)+'  result.json.gz\n')
    print(canon({k: result.get(k) for k in ('status', 'exact_checks', 'selected_seed',
                                           'gate_passed', 'selected', 'body_wall_seconds')}))


if __name__ == '__main__':
    main()
