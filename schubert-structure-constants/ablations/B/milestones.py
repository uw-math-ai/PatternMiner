"""Find the first complete projection certificates in the budgeted query logs."""
import csv
import json
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).with_name('matched-projection')


@lru_cache(maxsize=None)
def descents(perm):
    return frozenset(i for i in range(1, len(perm)) if perm[i - 1] > perm[i])


@lru_cache(maxsize=None)
def length(perm):
    return sum(x > y for x, y in combinations(perm, 2))


def first_certificates(arm):
    by_triple = defaultdict(list)
    with (DATA / arm / 'questions.tsv').open() as source:
        for count, row in enumerate(csv.DictReader(source, delimiter='\t'), 1):
            u, v, w = (row[key] for key in ('u', 'v', 'w'))
            if len(w) in (7, 8, 9):
                by_triple[u, v, w].append((count, row['question'], row['answer'], frozenset(map(int, row['retained']))))

    earliest = {}
    for (u, v, w), answers in by_triple.items():
        rank = len(w)
        depth = rank - 3
        dims = descents(u) | descents(v)
        if not descents(w) <= dims or len(dims) < depth or length(u) + length(v) != length(w):
            continue
        vanishing = [k for k, kind, value, _ in answers if (kind, value) in [('coefficient', '0'), ('empty', '1')]]
        if not vanishing:
            continue
        nonempty = [(k, retained) for k, kind, value, retained in answers if kind == 'empty' and value == '0' and len(retained) >= depth - 1]
        counts = [min(vanishing)]
        for subset in combinations(dims, depth - 1):
            first = next((k for k, retained in nonempty if set(subset) <= retained), None)
            if first is None:
                break
            counts.append(first)
        else:
            when = max(counts)
            if rank not in earliest or when < earliest[rank][0]:
                earliest[rank] = (when, u, v, w)
    return earliest


def verify_milestones(product, minimal_obstructions):
    expected = json.loads((DATA / 'first-certificates.json').read_text())
    for arm, witnesses in expected.items():
        earliest = first_certificates(arm)
        assert set(earliest) == {7, 8, 9}
        for rank, (count, *perms) in sorted(earliest.items()):
            row = witnesses[str(rank)]
            u, v, w = (tuple(map(int, perm)) for perm in perms)
            assert (u, v, w) == tuple(tuple(row[name]) for name in ('u', 'v', 'w'))
            assert count == row['questions'] and product(u, v).get(w, 0) == 0
            assert len(minimal_obstructions(u, v, w)[0]) == row['checked_depth'] == rank - 3
        print(arm, [earliest[rank][0] for rank in (7, 8, 9)])
