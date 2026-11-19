"""Independently enumerate rank-six cycling components that never reach a terminal test."""
from bisect import bisect_left
from collections import defaultdict
from itertools import combinations, permutations

from .boolean_formula import bruhat


def nonterminal_components(n=6):
    perms = list(permutations(range(1, n + 1)))
    index = {p: i for i, p in enumerate(perms)}
    by_length = defaultdict(list)
    length = []
    descents = []
    interval_counts = []
    swapped = []
    intervals = [tuple(combinations(range(lo, hi), 2))
                 for lo in range(n) for hi in range(lo + 2, n + 1)]
    capacities = [len(pairs) for pairs in intervals]
    for i, p in enumerate(perms):
        length.append(sum(a > b for a, b in combinations(p, 2)))
        by_length[length[-1]].append(i)
        descents.append(sum(1 << j for j in range(n - 1) if p[j] > p[j + 1]))
        interval_counts.append([sum(p[j] > p[k] for j, k in pairs) for pairs in intervals])
        swapped.append([index[p[:j] + (p[j + 1], p[j]) + p[j + 2:]] for j in range(n - 1)])

    complements = [tuple(n + 1 - x for x in p) for p in perms]
    ordered = [[bruhat(p, q) for q in complements] for p in perms]
    surviving = set()
    degree = n * (n - 1) // 2
    full_mask = (1 << (n - 1)) - 1
    for a in range(1, len(perms)):  # A triple containing the identity has a known pairing.
        for b in range(a, len(perms)):
            if not ordered[a][b]:
                continue
            targets = by_length[degree - length[a] - length[b]]
            for c in targets[bisect_left(targets, b):]:
                if not (ordered[a][c] and ordered[b][c]):
                    continue
                if descents[a] | descents[b] | descents[c] != full_mask:
                    continue
                if all(x + y + z >= cap for x, y, z, cap in
                       zip(interval_counts[a], interval_counts[b], interval_counts[c], capacities)):
                    surviving.add((a, b, c))

    unseen = surviving.copy()
    closed = []
    while unseen:
        first = unseen.pop()
        component = {first}
        queue = [first]
        reaches_terminal = False
        for triple in queue:
            for j in range(n - 1):
                has_descent = [s for s, p in enumerate(triple) if descents[p] & (1 << j)]
                if len(has_descent) != 1:
                    continue
                source = has_descent[0]
                for target in range(3):
                    if source == target:
                        continue
                    moved = list(triple)
                    for slot in (source, target):
                        moved[slot] = swapped[triple[slot]][j]
                    neighbor = tuple(sorted(moved))
                    if neighbor not in surviving:
                        reaches_terminal = True
                    elif neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
        if not reaches_terminal:
            # Use the same integer encoding as the archived representative table.
            size = len(perms)
            closed.append(frozenset((a * size + b) * size + c for a, b, c in component))
    return set(closed)
