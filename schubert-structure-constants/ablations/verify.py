"""Check the four Schubert ablations. Run with Python 3 and lrcalc installed."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from statistics import mean

import lrcalc
from A import ordinary, screen
from B.milestones import verify_milestones
from C.boolean_formula import boolean_coefficient, bruhat
from C.cycling import nonterminal_components
from D import internals, outputs

ROOT = Path(__file__).resolve().parent


def table(name, delimiter=','):
    with (ROOT / name).open() as source:
        return list(csv.DictReader(source, delimiter=delimiter))


def read_json(name):
    return json.loads((ROOT / name).read_text())


def triple(row):
    return tuple(tuple(map(int, row[key])) for key in ('u', 'v', 'w'))


def complement(p):
    return tuple(len(p) + 1 - x for x in p)


def pad(p, n):
    return tuple(p) + tuple(range(len(p) + 1, n + 1))


@lru_cache(maxsize=256)
def product(u, v):
    return {pad(p, len(u)): c for p, c in lrcalc.schubmult(u, v, len(u)).items()}


def verify_a(replay):
    counts = defaultdict(Counter)
    totals = Counter()
    for row in table('A/acquisition.csv'):
        u, v, w = triple(row)
        value = int(row['coefficient'])
        assert product(u, v).get(w, 0) == value
        policy = row['policy']
        totals[policy] += 1
        if value == 0 and not ordinary._screen_fails((u, v, complement(w))):
            counts[policy][int(row['block'])] += 1
    sampling = read_json('A/sampling.json')
    for policy, expected in sampling['hard_counts'].items():
        assert totals[policy] == 2048
        assert [counts[policy][i] for i in range(1, 9)] == expected
    assert [sum(counts[p].values()) for p in ('ordinary', 'screen', 'ranked')] == [6, 432, 122]

    results = defaultdict(Counter)
    for row in table('A/assessment.csv'):
        u, v, w = triple(row)
        value = int(row['coefficient'])
        assert product(u, v).get(w, 0) == value
        pop = results[row['cohort']]
        pop['rows'] += 1
        pop['zeros'] += value == 0
        for policy, rule in [('ordinary', ordinary), ('screen', screen)]:
            predicted = bool(int(row[policy]))
            if replay:
                assert rule.certifies_zero(u, v, w) == predicted, (policy, u, v, w)
            assert not predicted or value == 0
            pop[policy] += predicted
    expected = {'7_ordinary': (377, 377, 377), '8_ordinary': (382, 382, 382),
                '7_screen': (83, 83, 83), '8_screen': (107, 105, 106)}
    for cohort, values in expected.items():
        pop = results[cohort]
        assert pop['rows'] == 384
        assert tuple(pop[k] for k in ('zeros', 'ordinary', 'screen')) == values
    print('A: 6 ordinary, 432 conditionally sampled, 122 top-scoring hard negatives. '
          'Held-out hard negatives certified: 188/190 and 189/190.')


def projection_index(p, retained):
    ends = (0, *retained, len(p))
    return tuple(x for lo, hi in zip(ends, ends[1:]) for x in sorted(p[lo:hi]))


def minimal_obstructions(u, v, w):
    n = len(w)
    dims = [i for i in range(1, n) if u[i - 1] > u[i] or v[i - 1] > v[i]]
    for size in range(len(dims) + 1):
        obstructions = []
        for retained in combinations(dims, size):
            a, b, c = (projection_index(p, retained) for p in (u, v, complement(w)))
            if not any(bruhat(z, complement(c)) for z in product(a, b)):
                obstructions.append(retained)
        if obstructions:
            return obstructions
    return []


def verify_b():
    sizes = defaultdict(list)
    for row in table('B/projections.tsv', '\t'):
        u, v, w = triple(row)
        assert product(u, v).get(w, 0) == 0
        sets = minimal_obstructions(u, v, w)
        observed = {','.join(map(str, dims)) for dims in sets}
        assert observed == set(row['all_minimal_size_dimension_sets'].split('|'))
        size = len(sets[0])
        assert size == int(row['fewest_retained_dimensions_certifying_zero'])
        sizes[row['source']].append(size)
    assert sorted(sizes['B']) == [1] * 4 + [2] * 4 and sizes['Paper'] == [4]
    print('Earlier projection comparison: all eight zeros verified. Their minimal projections retain one or two dimensions; '
          'the rank-seven paper example needs four.')


def verify_c():
    rows = table('C/boolean-targets.csv')
    assert len(rows) == 5374
    for row in rows:
        u, v, w = triple(row)
        coefficient = product(u, v).get(w, 0)
        assert coefficient == int(row['coefficient']) == boolean_coefficient(u, v, w)
    for n in range(2, 8):
        # All subwords of s_1 ... s_(n-1), grouped by length.
        subwords = defaultdict(list)
        for mask in range(1 << (n - 1)):
            p = list(range(1, n + 1))
            for i in range(n - 1):
                if mask & (1 << i):
                    p[i], p[i + 1] = p[i + 1], p[i]
            subwords[mask.bit_count()].append(tuple(p))
        w = tuple(range(2, n + 1)) + (1,)
        count = sum(boolean_coefficient(u, v, w) for i in range(n)
                    for u in subwords[i] for v in subwords[n - 1 - i])
        assert count == math.comb(2 * n, n) // (n + 1)
    # Recompute the histogram on all homogeneous ordered triples in S_6.
    perms = list(permutations(range(1, 7)))
    lengths = {p: sum(x > y for x, y in combinations(p, 2)) for p in perms}
    per_length = Counter(lengths.values())
    histogram = Counter()
    for u in perms:
        for v in perms:
            targets = per_length[lengths[u] + lengths[v]]
            if targets:
                values = product(u, v)
                histogram[0] += targets - len(values)
                histogram.update(values.values())
    archive = read_json('C/enumeration.json')['6']
    assert histogram == {int(k): v for k, v in archive['ordered_coefficient_histogram'].items()}
    assert sum(histogram.values()) == 8881334
    components = read_json('C/components.json')
    assert nonterminal_components() == {frozenset(c['codes']) for c in components}
    for comp in components:
        _, u, v, w, value = comp['representative']
        assert product(tuple(u), tuple(v)).get(tuple(w), 0) == value == comp['coefficient']
    assert Counter(c['coefficient'] for c in components) == {0: 1, 1: 4, 2: 26}
    zero = next(c for c in components if c['coefficient'] == 0)
    assert zero['unordered_component_size'] == 5
    print('C: 5,374 Boolean-target values, Catalan counts through rank seven, '
          '8,881,334 rank-six triples, and all 31 unresolved representatives verified.')


def verify_d(replay):
    rows = [json.loads(line) for line in (ROOT / 'D/confirmation.jsonl').read_text().splitlines()]
    effects = table('D/joint-patches.csv')
    menus = read_json('D/head-sets.json')
    nominations = read_json('D/nominations.json')
    assert len(rows) == len(effects) == 498 and len(menus) == 19
    assert len({r['group'] for r in rows}) == 22
    group_changes = defaultdict(lambda: defaultdict(list))
    paired = []
    for i in range(0, len(rows), 2):
        pair = rows[i:i + 2]
        assert pair[0]['pair'] == pair[1]['pair'] and pair[0]['group'] == pair[1]['group']
        assert set(pair[0]['valid']).isdisjoint(pair[1]['valid'])
        paired.append(tuple(all(r[name] in r['valid'] for r in pair) for name in ('outputs', 'internals')))
        for j, r in enumerate(pair):
            effect = effects[i + j]
            assert (int(effect['pair']), int(effect['member'])) == (r['pair'], r['member'])
            if replay:
                args = (*triple(r), r['i'], tuple(r['candidates']))
                for name, rule in [('outputs', outputs), ('internals', internals)]:
                    assert rule.select_j(*args) == r[name], (name, r['pair'], j)
            if all(member['destination_model_valid'] for member in pair):
                for key in menus:
                    group_changes[key][r['group']].append(float(effect[key]))
    scores = {key: mean(mean(v) for v in groups.values()) for key, groups in group_changes.items()}
    ranking = sorted(scores, key=scores.get, reverse=True)
    for name, rank, contrast in [('outputs', 2, 10.46), ('internals', 1, 16.18)]:
        choice = nominations[name]
        assert ranking.index(choice) + 1 == rank and round(scores[choice], 2) == contrast
    assert Counter(paired) == {(True, True): 168, (True, False): 6,
                              (False, True): 46, (False, False): 29}
    print('D: selected head sets rank 2 and 1. Both moves valid on 174/249 and 214/249 pairs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-rules', action='store_true', help='also rerun A and D rules on every assessment row')
    replay = parser.parse_args().replay_rules
    verify_a(replay)
    verify_b()
    verify_milestones(product, minimal_obstructions)
    verify_c()
    verify_d(replay)
