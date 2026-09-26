"""Cover enumeration and descents extracted unchanged; no basin search."""

def descents(w):
    return tuple(i for i in range(len(w) - 1) if w[i] > w[i + 1])


def up_cover_data(x):
    out = []
    for a in range(len(x)):
        for b in range(a + 1, len(x)):
            p, q = x[a], x[b]
            if p >= q or any(p < x[h] < q for h in range(a + 1, b)):
                continue
            z = list(x)
            z[a], z[b] = z[b], z[a]
            out.append((a, b, p, q, tuple(z)))
    return out
