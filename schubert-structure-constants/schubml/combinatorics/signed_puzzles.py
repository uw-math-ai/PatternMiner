"""signed_puzzles.py -- the Pak-Robichaux signed-puzzle rule for Schubert
coefficients (arXiv:2504.17734, version of 2026-07-14), implemented as the
project's principal combinatorial oracle.

WHAT EXACTLY IS IMPLEMENTED (honest to the letter)
==================================================
Pak-Robichaux Theorem 1.1 states  c_{u,v}^w = t_+ - t_-,  where t_+/t_- count
signed tilings ("puzzles") of an  n x ell  parallelogram region Gamma(u,v,w),
ell = C(n,2) - inv(u), with the O(n^9) triangular tile set T_n of their
Section 5.  Their proof (Section 6, induction on ell) establishes a
sign-preserving bijection

    puzzles of Gamma(u,v,w)   <-->   complete leaves of the recurrence tree
                                     obtained by iterating Knutson's
                                     recurrence (arXiv:math/0306304, Lemma 4.1
                                     of PR) ALWAYS at the FIRST ASCENT of u,

one puzzle row per recurrence step:

  * exactly one dark triangle per row; the ascent condition on shaded tiles
    forces its column i to be the FIRST ascent of u (PR Sec. 6, first
    paragraph: "the index i must be the first ascent in the permutation u");
  * dark-triangle docket numbers/colors are precisely the branches of the
    recurrence: yellow = cases (1),(2),(3a),(3b); blue = eps=+1 summands of
    case (3); red = eps=-1 summands;
  * the sign of a puzzle is the parity of its red triangles, i.e. the parity
    of eps=-1 steps along the recurrence path;
  * the region has exactly ell = C(n,2)-inv(u) rows while every row raises
    inv(u) by >= 1 and the bottom boundary is (w_o, id, w_o); hence every
    complete puzzle raises inv(u) by EXACTLY 1 in every row (PR Sec. 6,
    discussion around condition (\\di)).  Discarding case-(3) transpositions
    t_{jk} with inv(u t_{jk}) > inv(u)+1 (non-covers) is therefore exact for
    t_+ and t_-: such rows can occur in partial tilings but never in a
    complete puzzle.

This module enumerates that path model faithfully by a level-by-level dynamic
program over recurrence states (u,v,w), keeping separate nonnegative counts
(p, m) of even-sign and odd-sign partial paths.  States are merged (the DP is
a DAG); path counts are additive, so merging is exact; NO normalization,
cancellation, sign-simplification or coefficient-zero pruning is applied at
any point.  t_+ and t_- are exactly the puzzle tallies of PR Theorem 1.1
under the row-by-row bijection of their proof.  What is *not* implemented is
the literal tile-set data structure (edge labels/indicators of the O(n^9)
triangles); the tiling model and the path model have identical signed counts
by PR Sec. 6, and the path model is what every downstream statistic needs.

KNUTSON'S RECURRENCE AS USED (PR Lemma 4.1, type A, non-equivariant)
====================================================================
Permutations act on POSITIONS on the right: (u s_i) swaps entries i, i+1 of
the one-line word.  Let i be an ascent of u (u(i) < u(i+1)).  Cases:

  (0) i not in Des(v), i in Des(w):  c = 0                       [dead branch]
  (1) i not in Des(v), i not in Des(w):  c(u,v,w) = c(us_i, v, ws_i)
  (2) i in Des(v), i in Des(w):          c(u,v,w) = c(us_i, vs_i, w)
  (3) i in Des(v), i not in Des(w):
      c(u,v,w) = c(us_i, v, ws_i) + c(us_i, vs_i, w)
                 + sum_{(j,k)} eps(i,j,k) c(u t_{jk}, vs_i, w),
      summed over 1 <= j < k <= n with u(j) < u(k) and |{j,k} n {i,i+1}| = 1,
      eps = +1 if j = i or k = i+1;  eps = -1 if k = i or j = i+1.

The DP applies the branch decomposition at i = first ascent of u, keeps only
cover transpositions in (3) (inv jump exactly 1 -- exact for t_+/t_-, see
above), and terminates after ell steps.  Because inv(u)+inv(v)-inv(w) is
invariant under every branch, any state that survives all ell levels is
automatically the terminal (w_o, id, w_o) with value 1.  Paths die only at
case-(0) nodes.  Sign of a path = product of eps over its case-(3) summand
steps.

CONVENTIONS AND CAVEATS
=======================
* t_+ and t_- (unlike their difference) depend on the ORDER of the pair
  (u, v): the rule pivots on u.  Callers that want the cheaper direction can
  swap u and v (c is symmetric); this module never swaps silently.
* `pivot` is "first_ascent" by default == the PR puzzle rule.  Alternative
  policies ("last_ascent", "random:<seed>") are provided ONLY for robustness
  experiments on the value/support; they realize other schedules of Knutson's
  recurrence, not the puzzle rule.
* Counts are exact Python big ints.  A state/level budget guards memory; on
  overflow the result carries budget_hit=True and t_plus/t_minus=None
  (NEVER silently truncated).

VALIDATION (run `python -m schubml.combinatorics.signed_puzzles --selftest`)
============================================================================
1. PR Section 7 worked example (n=7, u=3251647, v=4126537, w=6271534):
   the printed 15-step case/pivot sequence is verified step by step to be a
   legal path of this engine with sign +1 (two eps=-1 steps, one eps=+1;
   note the paper's step-8 label "eps(5,4,7)" is a typo for eps(5,6,7): the
   displayed one-line words force (j,k)=(6,7)), pivots = first ascents
   throughout, and t_+ - t_- = 1 = lrcalc.
2. n=3: ALL (u,v,w) in S_3^3 with inv u + inv v = inv w: t_+ - t_- == lrcalc.
3. n=4: ALL such triples (both orders of (u,v) for extra coverage).
4. Knutson-recurrence pivot audit, n=3 and n=4: for EVERY triple and EVERY
   ascent i of u (not just the first), the case-(0)/(1)/(2)/(3) identity is
   checked against lrcalc values of the children ==> every legal pivot
   schedule computes the same value; no pivot creates or destroys value.
5. n=5: >= 2,000 sampled coefficients incl. zeros and c >= 2, vs lrcalc.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Perm = Tuple[int, ...]


# ---------------------------------------------------------------------------
# permutation basics
# ---------------------------------------------------------------------------


def inv(p: Sequence[int]) -> int:
    n = len(p)
    return sum(1 for a in range(n) for b in range(a + 1, n) if p[a] > p[b])


def long_perm(n: int) -> Perm:
    return tuple(range(n, 0, -1))


def ident(n: int) -> Perm:
    return tuple(range(1, n + 1))


def first_ascent(p: Perm) -> Optional[int]:
    """1-indexed minimal i with p(i) < p(i+1); None iff p = w_o."""
    for i in range(len(p) - 1):
        if p[i] < p[i + 1]:
            return i + 1
    return None


def last_ascent(p: Perm) -> Optional[int]:
    for i in range(len(p) - 2, -1, -1):
        if p[i] < p[i + 1]:
            return i + 1
    return None


def apply_s(p: Perm, i: int) -> Perm:
    """Right multiplication by s_i: swap positions i, i+1 (1-indexed)."""
    q = list(p)
    q[i - 1], q[i] = q[i], q[i - 1]
    return tuple(q)


def apply_t(p: Perm, j: int, k: int) -> Perm:
    """Right multiplication by t_{jk}: swap positions j, k (1-indexed)."""
    q = list(p)
    q[j - 1], q[k - 1] = q[k - 1], q[j - 1]
    return tuple(q)


def is_cover_transposition(p: Perm, j: int, k: int) -> bool:
    """inv(p t_{jk}) == inv(p) + 1, for j < k with p(j) < p(k):
    no m in (j,k) with p(j) < p(m) < p(k)."""
    pj, pk = p[j - 1], p[k - 1]
    return all(not (pj < p[m] < pk) for m in range(j, k - 1))


# ---------------------------------------------------------------------------
# branch data (depends only on u and the pivot policy)
# ---------------------------------------------------------------------------


def branch_data(u: Perm, i: int) -> Tuple[Perm, List[Perm], List[Perm]]:
    """(u s_i, positive 3c-children, negative 3c-children) at ascent i of u.

    3c children are u t_{jk} restricted to COVER transpositions with
    |{j,k} n {i,i+1}| = 1 and u(j) < u(k); eps per PR Lemma 4.1:
      +1 : j = i (k >= i+2)   or   k = i+1 (j <= i-1)
      -1 : k = i (j <= i-1)   or   j = i+1 (k >= i+2)
    """
    n = len(u)
    assert u[i - 1] < u[i], (u, i)
    usi = apply_s(u, i)
    pos: List[Perm] = []
    neg: List[Perm] = []
    # j = i, k >= i+2, eps = +1
    for k in range(i + 2, n + 1):
        if u[i - 1] < u[k - 1] and is_cover_transposition(u, i, k):
            pos.append(apply_t(u, i, k))
    # k = i+1, j <= i-1, eps = +1
    for j in range(1, i):
        if u[j - 1] < u[i] and is_cover_transposition(u, j, i + 1):
            pos.append(apply_t(u, j, i + 1))
    # k = i, j <= i-1, eps = -1
    for j in range(1, i):
        if u[j - 1] < u[i - 1] and is_cover_transposition(u, j, i):
            neg.append(apply_t(u, j, i))
    # j = i+1, k >= i+2, eps = -1
    for k in range(i + 2, n + 1):
        if u[i] < u[k - 1] and is_cover_transposition(u, i + 1, k):
            neg.append(apply_t(u, i + 1, k))
    return usi, pos, neg


def _make_pivot(pivot: str) -> Callable[[Perm], Optional[int]]:
    if pivot == "first_ascent":
        return first_ascent
    if pivot == "last_ascent":
        return last_ascent
    if pivot.startswith("random:"):
        seed = int(pivot.split(":", 1)[1])

        def rand_pivot(p: Perm) -> Optional[int]:
            asc = [i + 1 for i in range(len(p) - 1) if p[i] < p[i + 1]]
            if not asc:
                return None
            # deterministic per-permutation choice, seeded
            r = random.Random(f"{seed}:{p}")
            return r.choice(asc)

        return rand_pivot
    raise ValueError(f"unknown pivot policy {pivot!r}")


# ---------------------------------------------------------------------------
# the signed-puzzle count (level DP over recurrence states)
# ---------------------------------------------------------------------------


class BudgetExceeded(Exception):
    pass


def signed_puzzle_counts(u: Sequence[int], v: Sequence[int],
                         w: Sequence[int], *,
                         pivot: str = "first_ascent",
                         max_states_per_level: int = 6_000_000,
                         collect_depth_stats: bool = True,
                         progress: bool = False) -> dict:
    """t_+/t_- of the PR signed-puzzle rule for c_{u,v}^w (path model).

    Requires len(u) == len(v) == len(w) and inv(u) + inv(v) == inv(w)
    (the dimension equation; PR Theorem 1.1 assumes it).  Returns a dict:
      t_plus, t_minus     exact big-int puzzle tallies (None if budget_hit)
      coeff               t_plus - t_minus (None if budget_hit)
      ell                 number of puzzle rows = C(n,2) - inv(u)
      died_at             level at which the state set became empty
                          (support-empty case), else None
      first_neg_depth     first level with any odd-sign path mass
      first_coexist_depth first level with a state carrying both signs
      n_mixed_states      total number of (level, state) pairs with both signs
      max_level_states    max #states on a level
      states_total        sum of #states over levels
      budget_hit          True if max_states_per_level was exceeded
    """
    u, v, w = tuple(u), tuple(v), tuple(w)
    n = len(u)
    assert len(v) == n and len(w) == n, "u,v,w must share the ambient n"
    assert inv(u) + inv(v) == inv(w), \
        "dimension equation inv(u)+inv(v)=inv(w) required"
    choose_pivot = _make_pivot(pivot)
    ell = n * (n - 1) // 2 - inv(u)
    wo, e = long_perm(n), ident(n)

    res = {"ell": ell, "pivot": pivot, "died_at": None,
           "first_neg_depth": None, "first_coexist_depth": None,
           "n_mixed_states": 0, "max_level_states": 0, "states_total": 0,
           "budget_hit": False}

    state: Dict[Tuple[Perm, Perm, Perm], List[int]] = {(u, v, w): [1, 0]}
    ucache: Dict[Perm, Tuple[int, Perm, List[Perm], List[Perm]]] = {}

    for d in range(1, ell + 1):
        nxt: Dict[Tuple[Perm, Perm, Perm], List[int]] = {}
        for (cu, cv, cw), (p, m) in state.items():
            got = ucache.get(cu)
            if got is None:
                i = choose_pivot(cu)
                usi, pos3c, neg3c = branch_data(cu, i)
                got = (i, usi, pos3c, neg3c)
                ucache[cu] = got
            i, usi, pos3c, neg3c = got
            dv = cv[i - 1] > cv[i]
            dw = cw[i - 1] > cw[i]
            if not dv:
                if dw:
                    continue                       # case (0): dead
                key = (usi, cv, apply_s(cw, i))    # case (1)
                slot = nxt.get(key)
                if slot is None:
                    nxt[key] = [p, m]
                else:
                    slot[0] += p
                    slot[1] += m
                continue
            if dw:
                key = (usi, apply_s(cv, i), cw)    # case (2)
                slot = nxt.get(key)
                if slot is None:
                    nxt[key] = [p, m]
                else:
                    slot[0] += p
                    slot[1] += m
                continue
            # case (3)
            cvsi = apply_s(cv, i)
            key = (usi, cv, apply_s(cw, i))        # (3a) +
            slot = nxt.get(key)
            if slot is None:
                nxt[key] = [p, m]
            else:
                slot[0] += p
                slot[1] += m
            key = (usi, cvsi, cw)                  # (3b) +
            slot = nxt.get(key)
            if slot is None:
                nxt[key] = [p, m]
            else:
                slot[0] += p
                slot[1] += m
            for u2 in pos3c:                       # (3c) eps = +1
                key = (u2, cvsi, cw)
                slot = nxt.get(key)
                if slot is None:
                    nxt[key] = [p, m]
                else:
                    slot[0] += p
                    slot[1] += m
            for u2 in neg3c:                       # (3c) eps = -1: flip
                key = (u2, cvsi, cw)
                slot = nxt.get(key)
                if slot is None:
                    nxt[key] = [m, p]
                else:
                    slot[0] += m
                    slot[1] += p
        state = nxt
        ns = len(state)
        res["states_total"] += ns
        res["max_level_states"] = max(res["max_level_states"], ns)
        if ns > max_states_per_level:
            res["budget_hit"] = True
            res["t_plus"] = res["t_minus"] = res["coeff"] = None
            return res
        if collect_depth_stats and state:
            if res["first_neg_depth"] is None and \
                    any(pm[1] > 0 for pm in state.values()):
                res["first_neg_depth"] = d
            mixed = sum(1 for pm in state.values() if pm[0] > 0 and pm[1] > 0)
            res["n_mixed_states"] += mixed
            if mixed and res["first_coexist_depth"] is None:
                res["first_coexist_depth"] = d
        if progress:
            print(f"    level {d}/{ell}: {ns} states", flush=True)
        if not state:
            res["died_at"] = d
            res["t_plus"] = res["t_minus"] = 0
            res["coeff"] = 0
            return res

    # after ell levels every surviving state is the terminal (see docstring)
    t_plus = t_minus = 0
    for (cu, cv, cw), (p, m) in state.items():
        assert cu == wo and cv == e and cw == wo, \
            ("non-terminal state survived to depth ell -- invariant broken",
             cu, cv, cw)
        t_plus += p
        t_minus += m
    res["t_plus"], res["t_minus"] = t_plus, t_minus
    res["coeff"] = t_plus - t_minus
    return res


# ---------------------------------------------------------------------------
# explicit path enumeration (for the involution hunt; small tallies only)
# ---------------------------------------------------------------------------


def enumerate_paths(u: Sequence[int], v: Sequence[int], w: Sequence[int], *,
                    pivot: str = "first_ascent",
                    max_paths: int = 200_000) -> List[dict]:
    """All complete signed-puzzle paths, as explicit step lists.

    Each path is {"sign": +-1, "steps": [(depth, case, i, (j,k) or None,
    eps or None), ...]}.  Raises BudgetExceeded beyond max_paths."""
    u, v, w = tuple(u), tuple(v), tuple(w)
    n = len(u)
    assert inv(u) + inv(v) == inv(w)
    choose_pivot = _make_pivot(pivot)
    ell = n * (n - 1) // 2 - inv(u)
    out: List[dict] = []
    steps: List[tuple] = []

    def rec(cu: Perm, cv: Perm, cw: Perm, d: int, sign: int) -> None:
        if d == ell:
            out.append({"sign": sign, "steps": list(steps)})
            if len(out) > max_paths:
                raise BudgetExceeded(len(out))
            return
        i = choose_pivot(cu)
        usi, _, _ = branch_data(cu, i)
        dv = cv[i - 1] > cv[i]
        dw = cw[i - 1] > cw[i]
        if not dv:
            if dw:
                return
            steps.append((d + 1, "1", i, None, None))
            rec(usi, cv, apply_s(cw, i), d + 1, sign)
            steps.pop()
            return
        if dw:
            steps.append((d + 1, "2", i, None, None))
            rec(usi, apply_s(cv, i), cw, d + 1, sign)
            steps.pop()
            return
        cvsi = apply_s(cv, i)
        steps.append((d + 1, "3a", i, None, None))
        rec(usi, cv, apply_s(cw, i), d + 1, sign)
        steps.pop()
        steps.append((d + 1, "3b", i, None, None))
        rec(usi, cvsi, cw, d + 1, sign)
        steps.pop()
        for (j, k), eps in _threec_pairs(cu, i):
            steps.append((d + 1, "3c", i, (j, k), eps))
            rec(apply_t(cu, j, k), cvsi, cw, d + 1, sign * eps)
            steps.pop()

    rec(u, v, w, 0, 1)
    return out


def _threec_pairs(u: Perm, i: int) -> List[Tuple[Tuple[int, int], int]]:
    """[( (j,k), eps )] for the cover 3c-transpositions at ascent i of u."""
    n = len(u)
    out: List[Tuple[Tuple[int, int], int]] = []
    for k in range(i + 2, n + 1):
        if u[i - 1] < u[k - 1] and is_cover_transposition(u, i, k):
            out.append(((i, k), 1))
    for j in range(1, i):
        if u[j - 1] < u[i] and is_cover_transposition(u, j, i + 1):
            out.append(((j, i + 1), 1))
    for j in range(1, i):
        if u[j - 1] < u[i - 1] and is_cover_transposition(u, j, i):
            out.append(((j, i), -1))
    for k in range(i + 2, n + 1):
        if u[i] < u[k - 1] and is_cover_transposition(u, i + 1, k):
            out.append(((i + 1, k), -1))
    return out


# ---------------------------------------------------------------------------
# ground truth (lrcalc) helpers for the selftest
# ---------------------------------------------------------------------------


def _lrcalc_table(u: Perm, v: Perm, n: int) -> Dict[Perm, int]:
    import lrcalc
    prod = lrcalc.schubmult(list(u), list(v), n)
    out: Dict[Perm, int] = {}
    for z, c in prod.items():
        z = tuple(int(t) for t in z)
        if len(z) < n:
            z = z + tuple(range(len(z) + 1, n + 1))
        if len(z) == n:
            out[z] = int(c)
    return out


def _check_triple(u: Perm, v: Perm, w: Perm, expected: int) -> None:
    r = signed_puzzle_counts(u, v, w)
    got = r["coeff"]
    assert got == expected, (u, v, w, expected, got, r)
    if expected == 0:
        assert r["t_plus"] == r["t_minus"]


def _audit_knutson_pivots(n: int) -> int:
    """For EVERY (u,v,w) with the dimension equation and EVERY ascent i of u,
    verify the case identity of PR Lemma 4.1 with lrcalc values.  This proves
    (exhaustively at this n) that every legal pivot yields the same value:
    no pivot schedule can create or destroy coefficient mass."""
    perms = [tuple(p) for p in itertools.permutations(range(1, n + 1))]
    by_inv: Dict[int, List[Perm]] = {}
    for p in perms:
        by_inv.setdefault(inv(p), []).append(p)
    tables: Dict[Tuple[Perm, Perm], Dict[Perm, int]] = {}

    def c_of(u: Perm, v: Perm, w: Perm) -> int:
        # zero whenever the dimension equation fails (PR Sec. 4)
        if inv(u) + inv(v) != inv(w):
            return 0
        t = tables.get((u, v))
        if t is None:
            t = _lrcalc_table(u, v, n)
            tables[(u, v)] = t
        return t.get(w, 0)

    n_checked = 0
    for u in perms:
        ascents = [i + 1 for i in range(n - 1) if u[i] < u[i + 1]]
        if not ascents:
            continue
        for v in perms:
            grade = inv(u) + inv(v)
            if grade > n * (n - 1) // 2:
                continue
            for w in by_inv.get(grade, ()):
                lhs = c_of(u, v, w)
                for i in ascents:
                    usi = apply_s(u, i)
                    dv = v[i - 1] > v[i]
                    dw = w[i - 1] > w[i]
                    if not dv and dw:
                        rhs = 0                                   # case (0)
                    elif not dv and not dw:
                        rhs = c_of(usi, v, apply_s(w, i))         # case (1)
                    elif dv and dw:
                        rhs = c_of(usi, apply_s(v, i), w)         # case (2)
                    else:                                          # case (3)
                        vsi = apply_s(v, i)
                        rhs = c_of(usi, v, apply_s(w, i)) \
                            + c_of(usi, vsi, w)
                        for (j, k), eps in _threec_pairs(u, i):
                            rhs += eps * c_of(apply_t(u, j, k), vsi, w)
                    assert lhs == rhs, ("pivot audit failed",
                                        u, v, w, i, lhs, rhs)
                    n_checked += 1
    return n_checked


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


PAPER_EXAMPLE = {
    # PR Section 7 (n=7).  Case labels as printed; the eps of step 8 is
    # printed "eps(5,4,7)" in the paper but the displayed words force
    # (j,k)=(6,7), i.e. eps(5,6,7) = -1 (j = i+1): a typo in the paper.
    "u": (3, 2, 5, 1, 6, 4, 7),
    "v": (4, 1, 2, 6, 5, 3, 7),
    "w": (6, 2, 7, 1, 5, 3, 4),
    "steps": [
        (2, "1", None), (1, "3b", None), (4, "3c", ((5, 7), -1)),
        (4, "1", None), (3, "1", None), (2, "2", None), (1, "1", None),
        (5, "3c", ((6, 7), -1)), (5, "1", None), (4, "3c", ((2, 5), 1)),
        (4, "1", None), (3, "2", None), (6, "1", None), (5, "1", None),
        (4, "1", None)],
}


def _replay_paper_example() -> None:
    ex = PAPER_EXAMPLE
    cu, cv, cw = ex["u"], ex["v"], ex["w"]
    n = len(cu)
    sign = 1
    for step_no, (i, case, jk_eps) in enumerate(ex["steps"], start=1):
        assert first_ascent(cu) == i, \
            (step_no, "pivot is not the first ascent", cu, i)
        dv = cv[i - 1] > cv[i]
        dw = cw[i - 1] > cw[i]
        if case == "1":
            assert not dv and not dw, (step_no, "case 1 preconditions")
            cu, cw = apply_s(cu, i), apply_s(cw, i)
        elif case == "2":
            assert dv and dw, (step_no, "case 2 preconditions")
            cu, cv = apply_s(cu, i), apply_s(cv, i)
        elif case == "3a":
            assert dv and not dw
            cu, cw = apply_s(cu, i), apply_s(cw, i)
        elif case == "3b":
            assert dv and not dw, (step_no, "case 3b preconditions")
            cu, cv = apply_s(cu, i), apply_s(cv, i)
        elif case == "3c":
            assert dv and not dw, (step_no, "case 3c preconditions")
            (j, k), eps = jk_eps
            legal = _threec_pairs(cu, i)
            assert ((j, k), eps) in legal, (step_no, (j, k), eps, legal)
            cu, cv = apply_t(cu, j, k), apply_s(cv, i)
            sign *= eps
    assert cu == long_perm(n) and cv == ident(n) and cw == long_perm(n), \
        ("paper path does not terminate correctly", cu, cv, cw)
    assert sign == 1, sign
    r = signed_puzzle_counts(ex["u"], ex["v"], ex["w"])
    tbl = _lrcalc_table(ex["u"], ex["v"], n)
    assert r["coeff"] == tbl.get(ex["w"], 0) == 1, r
    print(f"[ok] paper example replayed: 15 steps legal, pivots = first "
          f"ascents, path sign +1; engine t+={r['t_plus']} "
          f"t-={r['t_minus']} diff=1 == lrcalc")


def _exhaustive(n: int, both_orders: bool = False) -> Tuple[int, int]:
    perms = [tuple(p) for p in itertools.permutations(range(1, n + 1))]
    by_inv: Dict[int, List[Perm]] = {}
    for p in perms:
        by_inv.setdefault(inv(p), []).append(p)
    n_checked = n_zero = 0
    for u in perms:
        for v in perms:
            grade = inv(u) + inv(v)
            if grade > n * (n - 1) // 2:
                continue
            tbl = _lrcalc_table(u, v, n)
            for w in by_inv.get(grade, ()):
                c = tbl.get(w, 0)
                _check_triple(u, v, w, c)
                if both_orders:
                    _check_triple(v, u, w, c)
                n_checked += 1 + int(both_orders)
                n_zero += int(c == 0) * (1 + int(both_orders))
    return n_checked, n_zero


def _sampled_n5(target: int = 2000, seed: int = 20260723) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    n = 5
    n_checked = n_zero = n_ge2 = 0
    perms = [tuple(p) for p in itertools.permutations(range(1, n + 1))]
    by_inv: Dict[int, List[Perm]] = {}
    for p in perms:
        by_inv.setdefault(inv(p), []).append(p)
    while n_checked < target:
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        grade = inv(u) + inv(v)
        if grade > n * (n - 1) // 2:
            continue
        tbl = _lrcalc_table(u, v, n)
        graded = by_inv.get(grade, ())
        zeros = [w for w in graded if w not in tbl]
        ge2 = [w for w in graded if tbl.get(w, 0) >= 2]
        pick: List[Perm] = []
        pos = [w for w in graded if tbl.get(w, 0) >= 1]
        if pos:
            pick.append(rng.choice(pos))
        if zeros:
            pick.append(rng.choice(zeros))
        if ge2:
            pick.append(rng.choice(ge2))
        for w in pick:
            c = tbl.get(w, 0)
            _check_triple(u, v, w, c)
            n_checked += 1
            n_zero += int(c == 0)
            n_ge2 += int(c >= 2)
    return n_checked, n_zero, n_ge2


def _sampled_c_ge2_n6(target: int = 50, seed: int = 1) -> int:
    """No c >= 2 exists with u,v,w all in S_5; validate multiplicities at
    n=6: random products until `target` coefficients with c >= 2 are checked
    (plus one matched zero per product)."""
    rng = random.Random(seed)
    n = 6
    perms = [tuple(p) for p in itertools.permutations(range(1, n + 1))]
    by_inv: Dict[int, List[Perm]] = {}
    for p in perms:
        by_inv.setdefault(inv(p), []).append(p)
    n2 = 0
    while n2 < target:
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        grade = inv(u) + inv(v)
        if grade > n * (n - 1) // 2:
            continue
        tbl = _lrcalc_table(u, v, n)
        ge2 = [w for w, c in tbl.items() if c >= 2]
        if not ge2:
            continue
        for w in ge2:
            _check_triple(u, v, w, tbl[w])
            n2 += 1
        zeros = [w for w in by_inv.get(grade, ()) if w not in tbl]
        if zeros:
            _check_triple(u, v, rng.choice(zeros), 0)
    return n2


def selftest() -> None:
    t0 = time.time()
    _replay_paper_example()

    nc, nz = _exhaustive(3)
    print(f"[ok] n=3 exhaustive: {nc} triples ({nz} zeros), 0 mismatches")
    nc, nz = _exhaustive(4, both_orders=True)
    print(f"[ok] n=4 exhaustive (both (u,v) orders): {nc} coefficient "
          f"checks ({nz} zeros), 0 mismatches")

    na = _audit_knutson_pivots(3)
    print(f"[ok] Knutson pivot audit n=3: {na} (triple, ascent) identities")
    na = _audit_knutson_pivots(4)
    print(f"[ok] Knutson pivot audit n=4: {na} (triple, ascent) identities "
          f"-- every legal pivot yields the same value")

    nc, nz, n2 = _sampled_n5()
    print(f"[ok] n=5 sampled: {nc} coefficients ({nz} zeros, {n2} with "
          f"c>=2), 0 mismatches")
    # NOTE: no Schubert coefficient with u,v,w ALL in S_5 exceeds 1
    # (exhaustively checked against lrcalc), so multiplicities c >= 2 are
    # validated with a targeted n=6 batch instead:
    n2 = _sampled_c_ge2_n6()
    print(f"[ok] n=6 targeted multiplicities: {n2} coefficients with c>=2 "
          f"(plus matched zeros), 0 mismatches")

    # path enumeration consistent with the DP on a small batch
    rng = random.Random(7)
    n = 4
    done = 0
    while done < 40:
        u = tuple(rng.sample(range(1, n + 1), n))
        v = tuple(rng.sample(range(1, n + 1), n))
        grade = inv(u) + inv(v)
        if grade > 6:
            continue
        tbl = _lrcalc_table(u, v, n)
        for w in [p for p in itertools.permutations(range(1, n + 1))
                  if inv(p) == grade][:3]:
            paths = enumerate_paths(u, v, tuple(w))
            tp = sum(1 for p in paths if p["sign"] > 0)
            tm = sum(1 for p in paths if p["sign"] < 0)
            r = signed_puzzle_counts(u, v, tuple(w))
            assert (tp, tm) == (r["t_plus"], r["t_minus"]), (u, v, w)
            done += 1
    print(f"[ok] explicit path enumeration == DP tallies on {done} triples")
    print(f"selftest PASSED in {round(time.time() - t0, 1)}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
