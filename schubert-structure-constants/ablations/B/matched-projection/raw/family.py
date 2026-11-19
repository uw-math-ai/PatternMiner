"""An oracle-independent family with minimum certificate size n - 2.

The uniform construction and its Monk-formula proof appear in REPORT.md.
No ranked examples, fitted parameters, oracle calls, or search are used.
"""

def example(n):
    if not isinstance(n, int) or n < 4:
        return None
    u = [2, 1, 4, 3] + list(range(5, n + 1))
    v = [3, 1, 2, 4] + list(range(5, n + 1))
    for k in range(5, n, 2):
        u[k - 1], u[k] = u[k], u[k - 1]
    for k in range(4, n, 2):
        v[k - 1], v[k] = v[k], v[k - 1]
    w = [n, 2, 1, 3] + list(range(4, n))
    return u, v, w
