"""Exact finite audit of a basis-specific Morse-matching theorem."""

from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
from itertools import combinations
import hashlib
import json
from pathlib import Path
import resource
import signal
import time

from .budget import RunBudget
from .e005_full_rows_cancellation import ROOT, freeze


def canonical_triple(triple, n):
    # A lexicographically minimal translate contains zero.
    return min(tuple(sorted((v - start) % n for v in triple)) for start in triple)


def orbit_matrix(n):
    orbits = Counter(canonical_triple(t, n) for t in combinations(range(n), 3))
    labels = sorted(orbits)
    rows = []
    for triple in labels:
        row = [0] * (n // 2)
        for x, y in combinations(triple, 2):
            d = min(y - x, n - (y - x))
            row[d - 1] += 1
        rows.append(row)
    return labels, rows, orbits


def rational_rank(rows):
    matrix = [[Fraction(x) for x in row] for row in rows]
    rank = 0
    for col in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][col]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        value = matrix[rank][col]
        matrix[rank] = [v / value for v in matrix[rank]]
        for i in range(rank + 1, len(matrix)):
            value = matrix[i][col]
            if value:
                matrix[i] = [x - value * y for x, y in zip(matrix[i], matrix[rank])]
        rank += 1
    return rank


def peel(rows):
    active = set(range(len(rows[0])))
    trace = []
    while active:
        pivot = None
        for i, row in enumerate(rows):
            support = [j for j in sorted(active) if row[j]]
            if len(support) == 1:
                pivot = (i, support[0])
                break
        if pivot is None:
            break
        trace.append(pivot)
        active.remove(pivot[1])
    assert len({i for i, _ in trace}) == len(trace)
    return trace, sorted(active)


def acyclic(rows, matching):
    # matching maps each matched column to its assigned row.
    width = len(rows[0])
    graph = [set() for _ in range(width + len(rows))]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if value:
                source, target = (width + i, j) if matching.get(j) == i else (j, width + i)
                graph[source].add(target)
    indegree = [0] * len(graph)
    for neighbors in graph:
        for v in neighbors:
            indegree[v] += 1
    ready = [v for v, count in enumerate(indegree) if count == 0]
    visited = 0
    while ready:
        v = ready.pop()
        visited += 1
        for w in graph[v]:
            indegree[w] -= 1
            if indegree[w] == 0:
                ready.append(w)
    return visited == len(graph)


def enumerate_matchings(rows):
    width = len(rows[0])
    neighbors = [[i for i, row in enumerate(rows) if row[j]] for j in range(width)]
    totals = {'recursion_nodes': 0, 'saturating': 0, 'acyclic': 0}
    def visit(j, matching, used):
        totals['recursion_nodes'] += 1
        assert totals['recursion_nodes'] <= 2_000_000
        if j == width:
            totals['saturating'] += 1
            totals['acyclic'] += int(acyclic(rows, matching))
            return
        for i in neighbors[j]:
            if i not in used:
                matching[j] = i
                visit(j + 1, matching, used | {i})
                del matching[j]
    visit(0, {}, set())
    return totals


def independent_raw_inclusion(n, labels, rows, orbits):
    accum = {t: [0] * (n // 2) for t in labels}
    for x, y in combinations(range(n), 2):
        d = min(y - x, n - (y - x))
        for z in range(n):
            if z not in (x, y):
                t = canonical_triple((x, y, z), n)
                accum[t][d - 1] += 1
    for t, row in zip(labels, rows):
        assert accum[t] == [orbits[t] * value for value in row]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def wall_cap(seconds):
    def timeout(_signum, _frame):
        raise TimeoutError('E015 wall-time limit reached')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    outdir = ROOT / 'artifacts/e015'
    assert not (outdir / 'result.json').exists(), 'Frozen run already exists'
    proof = ROOT / 'proofs/P12-cyclic-orbit-Morse-classification.md'
    protocol = ROOT / 'experiments/E015-cyclic-orbit-Morse-classification.md'
    result = {'experiment': 'E015', 'code_sha256': digest(Path(__file__)),
              'proof_sha256': digest(proof), 'protocol_sha256': digest(protocol),
              'degrees': []}
    with RunBudget(ROOT, 'e015-cyclic-orbit-morse-v1') as budget, wall_cap(29):
        started = time.monotonic()
        for n in range(6, 31):
            labels, rows, orbits = orbit_matrix(n)
            rank = rational_rank(rows)
            assert rank == n // 2
            trace, remaining = peel(rows)
            predicted = n in (6, 12, 24)
            assert (not remaining) == predicted, (n, remaining)
            matching = {col: row for row, col in trace}
            assert acyclic(rows, matching)
            item = {'n': n, 'triple_representatives': labels, 'matrix': rows,
                    'column_rank': rank, 'cokernel_dimension': len(rows) - rank,
                    'peeling_trace': trace, 'remaining_columns': remaining,
                    'complete_acyclic_matching': not remaining}
            if n <= 10:
                independent_raw_inclusion(n, labels, rows, orbits)
                counts = enumerate_matchings(rows)
                assert counts['saturating'] > 0
                assert (counts['acyclic'] > 0) == predicted
                item['exhaustive_matching_counts'] = counts
                item['raw_inclusion_control'] = 'passed'
            if n == 7:
                expected = {(0, 1, 2): [2, 1, 0], (0, 1, 3): [1, 1, 1],
                            (0, 1, 5): [1, 1, 1], (0, 1, 4): [1, 0, 2],
                            (0, 2, 4): [0, 2, 1]}
                assert dict(zip(labels, rows)) == expected
                assert item['cokernel_dimension'] == 2
            result['degrees'].append(item)
            assert time.monotonic() - started < 29
            budget.check()
        result['elapsed_seconds'] = time.monotonic() - started
        result['status'] = 'passed; finite audit of a separately proved basis-specific theorem'
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        outdir.mkdir(exist_ok=True)
        sha = freeze(outdir / 'result.json', result)
        print(json.dumps({'sha256': sha, 'degrees': 25, 'seconds': result['elapsed_seconds'],
                          'complete_degrees': [r['n'] for r in result['degrees'] if r['complete_acyclic_matching']],
                          'exhaustive': [{k: r[k] for k in ('n', 'exhaustive_matching_counts')}
                                         for r in result['degrees'] if r['n'] <= 10]}), flush=True)


if __name__ == '__main__':
    main()
