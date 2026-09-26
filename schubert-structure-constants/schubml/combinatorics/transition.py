"""Lascoux--Schuetzenberger transition tree (best-effort, optional features).

The transition recurrence reduces a permutation toward Grassmannian leaves:

* ``r`` = last descent ``max{i : w(i) > w(i+1)}``,
* ``s`` = ``max{k > r : w(k) < w(r)}``,
* ``v`` = ``w * t_{r,s}``  (length drops by 1),
* children = ``{ v * t_{i,r} : i < r, length(v t_{i,r}) = length(w) }`` (the
  permutations covering ``v`` via a transposition into column ``r``).

We expose only *structural statistics* of the resulting tree (leaf count, depth,
branching, a stable hash) which PROJECT.md uses as optional auxiliary targets,
split criteria, and post-hoc clustering variables (section 7.7).  The algebraic
coefficient identity is intentionally not relied upon anywhere downstream; the
``transition_tree`` features are nullable and disabled by default.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from .permutations import Perm, canonical_perm, length, right_descents, right_mult_t
from .known_families import grassmannian_partition, is_grassmannian


def _last_descent(w: Sequence[int]) -> Optional[int]:
    d = right_descents(w)
    return d[-1] if d else None


def transition_children(w: Sequence[int]) -> List[Perm]:
    """Transition set (children) of ``w``; empty iff ``w`` is Grassmannian/identity."""
    r = _last_descent(w)
    if r is None:
        return []
    n = len(w)
    # s = max k>r with w(k) < w(r); positions are 1-indexed
    s = None
    for k in range(n, r, -1):
        if w[k - 1] < w[r - 1]:
            s = k
            break
    if s is None:
        return []
    v = right_mult_t(w, r, s)  # length = length(w) - 1
    lw = length(w)
    children: List[Perm] = []
    for i in range(1, r):
        cand = right_mult_t(v, i, r)
        if length(cand) == lw:
            children.append(tuple(cand))
    return children


def transition_tree_stats(w: Sequence[int], max_depth: int = 64) -> Dict[str, object]:
    """Structural statistics of the transition tree rooted at ``w``.

    Returns leaf count, max depth, branching profile, leaf Grassmannian shapes,
    and a stable content hash.  Uses memoization over canonical permutations.
    """
    memo: Dict[Perm, Dict[str, object]] = {}

    def rec(node: Perm, depth: int) -> Dict[str, object]:
        cnode = canonical_perm(node)
        if cnode in memo:
            return memo[cnode]
        if depth >= max_depth:
            res = {"leaves": 1, "depth": 0, "branch": [], "shapes": []}
            memo[cnode] = res
            return res
        children = transition_children(node)
        if not children:
            shape = grassmannian_partition(node)
            res = {
                "leaves": 1,
                "depth": 0,
                "branch": [],
                "shapes": [list(shape)] if shape is not None else [],
            }
            memo[cnode] = res
            return res
        leaves = 0
        maxd = 0
        branch = [len(children)]
        shapes: List[List[int]] = []
        for c in children:
            sub = rec(c, depth + 1)
            leaves += int(sub["leaves"])  # type: ignore[arg-type]
            maxd = max(maxd, int(sub["depth"]))  # type: ignore[arg-type]
            branch.extend(sub["branch"])  # type: ignore[arg-type]
            shapes.extend(sub["shapes"])  # type: ignore[arg-type]
        res = {"leaves": leaves, "depth": maxd + 1, "branch": branch, "shapes": shapes}
        memo[cnode] = res
        return res

    root = rec(tuple(w), 0)
    leaf_shapes = sorted([tuple(s) for s in root["shapes"]])  # type: ignore[arg-type]
    branch = root["branch"]  # type: ignore[assignment]
    stats = {
        "number_of_leaves": int(root["leaves"]),  # type: ignore[arg-type]
        "max_depth": int(root["depth"]),  # type: ignore[arg-type]
        "branching_max": max(branch) if branch else 0,
        "branching_mean": (sum(branch) / len(branch)) if branch else 0.0,
        "n_internal": len(branch),
        "leaf_grassmannian_shapes": [list(s) for s in leaf_shapes],
    }
    return stats


def transition_tree_hash(w: Sequence[int], max_depth: int = 64) -> str:
    """Stable hash of the transition tree's leaf-shape multiset + coarse stats."""
    stats = transition_tree_stats(w, max_depth=max_depth)
    key = {
        "leaves": stats["number_of_leaves"],
        "depth": stats["max_depth"],
        "shapes": stats["leaf_grassmannian_shapes"],
    }
    blob = json.dumps(key, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]
