"""Exact type A structure constants for boolean target permutations."""

def bruhat(a, b):
    """Return a <= b in the strong Bruhat order."""
    return all(x <= y for k in range(1, len(a))
               for x, y in zip(sorted(a[:k]), sorted(b[:k])))


def reduced_word(p):
    """One-based simple indices, multiplied on the right from left to right."""
    q = list(p)
    out = []
    while True:
        i = next((i for i in range(len(q) - 1) if q[i] > q[i + 1]), None)
        if i is None:
            return out[::-1]
        q[i], q[i + 1] = q[i + 1], q[i]
        out.append(i + 1)


def is_boolean(p):
    word = reduced_word(p)
    return len(word) == len(set(word))


def boolean_coefficient(u, v, w, details=False):
    """Return None outside the boolean-target family, otherwise 0 or 1."""
    word = reduced_word(w)
    if len(word) != len(set(word)):
        return None
    wu, wv = reduced_word(u), reduced_word(v)
    if len(wu) + len(wv) != len(word) or not bruhat(u, w) or not bruhat(v, w):
        return dict(coefficient=0, reason='degree_or_bruhat') if details else 0
    position = {r: j for j, r in enumerate(word)}
    supply = {r: int(r in wu) + int(r in wv) - 1 for r in word}
    support = sorted(word)
    components = []
    for r in support:
        if not components or r != components[-1][-1] + 1:
            components.append([])
        components[-1].append(r)
    flow = []
    for component in components:
        if sum(supply[r] for r in component):
            return dict(coefficient=0, reason='component_imbalance') if details else 0
        left_sum = 0
        for r in component[:-1]:
            left_sum += supply[r]
            if position[r] > position[r + 1]:
                source, target, amount = r, r + 1, left_sum
            else:
                source, target, amount = r + 1, r, -left_sum
            flow.append([source, target, amount])
            if amount < 0:
                return dict(coefficient=0, reason='negative_edge_flow',
                            word=word, flow=flow) if details else 0
    result = dict(coefficient=1, word=word, supply=supply, flow=flow)
    return result if details else 1


if __name__ == '__main__':
    import sys, json
    u,v,w = [tuple(map(int,s)) for s in sys.argv[1:]]
    print(json.dumps(boolean_coefficient(u,v,w,details=True),indent=2))
