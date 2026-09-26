"""Enriched rank-matrix / Rothe predicate battery for the hard-zero residual.

Beyond the base necessary conditions (grading + Bruhat + descent-union), we mine
finer features of the rank-slack configuration

    s(i,j)  = r_u(i,j) + r_v(i,j) - r_w(i,j)   (the "rank inequality" slack),
    e_u(i,j)= r_u(i,j) - r_w(i,j) >= 0,  e_v(i,j)= r_v(i,j) - r_w(i,j) >= 0,

restricted to interior cells and to Fulton's essential set of w, plus tight-cell
counts / patterns.  The goal is to find a small, statable combinatorial predicate
that separates the residual zeros the trained model resolves.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..combinatorics import permutations as P
from ..combinatorics import rank_matrix as RM
from ..combinatorics import rothe


def enriched_features(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> Dict[str, float]:
    N = max(len(P.canonical_perm(u)), len(P.canonical_perm(v)), len(P.canonical_perm(w)), 1)
    up, vp, wp = P.pad(u, N), P.pad(v, N), P.pad(w, N)
    Ru = RM.rank_matrix(up).astype(np.int64)
    Rv = RM.rank_matrix(vp).astype(np.int64)
    Rw = RM.rank_matrix(wp).astype(np.int64)
    s = Ru + Rv - Rw
    eu = Ru - Rw
    ev = Rv - Rw
    # interior cells 1..N-1 x 1..N-1
    I = slice(1, N)
    si = s[I, I]; eui = eu[I, I]; evi = ev[I, I]
    both = (eui == 0) & (evi == 0)
    f: Dict[str, float] = {}
    f["min_slack"] = float(si.min()) if si.size else 0.0
    f["n_slack0"] = float((si == 0).sum())
    f["n_slack1"] = float((si == 1).sum())
    f["mean_slack"] = float(si.mean()) if si.size else 0.0
    f["n_eu0"] = float((eui == 0).sum())
    f["n_ev0"] = float((evi == 0).sum())
    f["n_both_tight"] = float(both.sum())
    f["has_both_tight"] = float(both.any())
    # both-tight forming a 2x2 block or a full row/col of the interior
    if both.shape[0] >= 2 and both.shape[1] >= 2:
        b2 = both[:-1, :-1] & both[1:, :-1] & both[:-1, 1:] & both[1:, 1:]
        f["both_tight_2x2"] = float(b2.any())
    else:
        f["both_tight_2x2"] = 0.0
    f["both_tight_full_row"] = float(both.all(axis=1).any()) if both.size else 0.0
    f["both_tight_full_col"] = float(both.all(axis=0).any()) if both.size else 0.0
    # essential set of w: slack there
    ess = RM.essential_set(wp)
    if ess:
        es = [int(s[i, j]) for (i, j) in ess]
        ebo = [bool((eu[i, j] == 0) and (ev[i, j] == 0)) for (i, j) in ess]
        f["ess_min_slack"] = float(min(es))
        f["ess_n_slack0"] = float(sum(1 for x in es if x == 0))
        f["ess_n_both_tight"] = float(sum(ebo))
        f["ess_has_both_tight"] = float(any(ebo))
        f["ess_size"] = float(len(ess))
    else:
        for k in ("ess_min_slack", "ess_n_slack0", "ess_n_both_tight",
                  "ess_has_both_tight", "ess_size"):
            f[k] = 0.0
    # Rothe / length context
    f["ell_w"] = float(P.length(wp))
    f["n_rdesc_w"] = float(P.n_right_descents(wp))
    f["n_ldesc_w"] = float(P.n_left_descents(wp))
    rows = rothe.rothe_row_lengths(wp)
    f["rothe_max_row"] = float(max(rows) if rows else 0)
    f["rothe_n_rows"] = float(sum(1 for r in rows if r > 0))
    return f


def feature_names() -> List[str]:
    return list(enriched_features((2, 1), (2, 1), (3, 1, 2)).keys())


def config_features(u: Sequence[int], v: Sequence[int], w: Sequence[int]) -> Dict[str, float]:
    """Configuration-level geometry of the rank-slack pattern.

    Where the aggregate battery (enriched_features) counts tight cells, these
    describe *where* they sit: distance to the essential set, containment in
    D(w), contiguity, near-tightness at corners — the shape information the
    residual model appears to use beyond aggregates.
    """
    from ..combinatorics import rothe

    N = max(len(P.canonical_perm(u)), len(P.canonical_perm(v)), len(P.canonical_perm(w)), 1)
    up, vp, wp = P.pad(u, N), P.pad(v, N), P.pad(w, N)
    Ru = RM.rank_matrix(up).astype(np.int64)
    Rv = RM.rank_matrix(vp).astype(np.int64)
    Rw = RM.rank_matrix(wp).astype(np.int64)
    s = (Ru + Rv - Rw)[1:N, 1:N]  # interior; 0-indexed cell (a,b) = matrix (a+1,b+1)
    tight = np.argwhere(s == 0)
    near = np.argwhere(s == 1)
    ess = [(i - 1, j - 1) for (i, j) in RM.essential_set(wp)]  # 0-indexed
    dw = {(i - 1, j - 1) for (i, j) in rothe.rothe_cells(wp)}
    f: Dict[str, float] = {}

    def min_cheb(cells, targets):
        if len(cells) == 0 or not targets:
            return float(N)  # sentinel: no such cell / no target
        t = np.array(list(targets))
        d = np.abs(cells[:, None, :] - t[None, :, :]).max(-1)
        return float(d.min())

    f["n_tight"] = float(len(tight))
    f["n_near"] = float(len(near))
    f["tight_dist_ess"] = min_cheb(tight, ess)   # 0 is forbidden by C-20260702-01
    f["near_dist_ess"] = min_cheb(near, ess)     # is s=1 adjacent to corners forbidden too?
    f["tight_in_dw"] = float(sum(1 for c in tight if (int(c[0]), int(c[1])) in dw))
    f["tight_out_dw"] = float(len(tight) - f["tight_in_dw"])
    # near-tight ON the essential set: the graded sequel to the theorem
    f["ess_n_slack1"] = float(sum(1 for (i, j) in ess if s[i, j] == 1)) if ess else 0.0
    f["ess_n_slack2"] = float(sum(1 for (i, j) in ess if s[i, j] == 2)) if ess else 0.0
    # contiguity / shape of the tight region
    if len(tight) > 0:
        cells = {(int(a), int(b)) for a, b in tight}
        comps, todo = 0, set(cells)
        while todo:
            comps += 1
            stack = [todo.pop()]
            while stack:
                a, b = stack.pop()
                for nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if nb in todo:
                        todo.remove(nb); stack.append(nb)
        f["tight_components"] = float(comps)
        rows = {a for a, _ in cells}; cols = {b for _, b in cells}
        f["tight_row_span"] = float(max(rows) - min(rows) + 1)
        f["tight_col_span"] = float(max(cols) - min(cols) + 1)
        f["tight_max_row_run"] = float(max(sum(1 for b in range(N - 1) if (a, b) in cells)
                                           for a in rows))
    else:
        f["tight_components"] = f["tight_row_span"] = 0.0
        f["tight_col_span"] = f["tight_max_row_run"] = 0.0
    # asymmetric tightness (u-side vs v-side)
    eu0 = ((Ru - Rw)[1:N, 1:N] == 0)
    ev0 = ((Rv - Rw)[1:N, 1:N] == 0)
    f["asym_tight"] = float(abs(int(eu0.sum()) - int(ev0.sum())))
    # slack restricted to D(w) cells
    sd = [s[a, b] for (a, b) in dw if 0 <= a < N - 1 and 0 <= b < N - 1]
    f["dw_min_slack"] = float(min(sd)) if sd else float(N)
    f["dw_n_slack0"] = float(sum(1 for x in sd if x == 0))
    return f


def config_feature_names() -> List[str]:
    return list(config_features((2, 1), (2, 1), (3, 1, 2)).keys())
