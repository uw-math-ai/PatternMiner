import numpy as np

from schubml.combinatorics import bruhat
from schubml.combinatorics import permutations as P
from schubml.combinatorics import rank_matrix as RM


def test_rank_matrix_basic():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            R = RM.rank_matrix(w)
            assert R.shape == (n + 1, n + 1)
            for i in range(n + 1):
                for j in range(n + 1):
                    assert R[i, j] == RM.rank_at(w, i, j)
            # identity has the maximal rank min(i,j)
    e = P.identity(4)
    Re = RM.rank_matrix(e)
    for i in range(5):
        for j in range(5):
            assert Re[i, j] == min(i, j)


def test_bruhat_order_identity_and_w0():
    for n in range(1, 6):
        e, w0 = P.identity(n), P.longest_element(n)
        for w in P.all_permutations(n):
            assert bruhat.bruhat_leq(e, w)
            assert bruhat.bruhat_leq(w, w0)
            assert bruhat.bruhat_leq(w, w)


def test_bruhat_antisymmetry_and_covers():
    # antisymmetry: u<=w and w<=u implies u==w
    for w in P.all_permutations(4):
        for u in P.all_permutations(4):
            if bruhat.bruhat_leq(u, w) and bruhat.bruhat_leq(w, u):
                assert u == w
    # covers up from identity in S_4 are the 3 simple reflections
    covers = bruhat.cover_transpositions_up(P.identity(4))
    assert len(covers) == 3
    for c in covers:
        assert P.length(c) == 1


def test_essential_set_determines_rank_conditions():
    w = (3, 1, 4, 2)
    ess = RM.essential_set(w)
    conds = RM.essential_rank_conditions(w)
    assert len(ess) == len(conds)
    for (i, j, r) in conds:
        assert r == RM.rank_at(w, i, j)
