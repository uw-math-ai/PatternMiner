import random

import pytest

from schubml.combinatorics import permutations as P
from schubml.combinatorics import reduced_words as RW


def test_inverse_and_compose():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            assert P.inverse(P.inverse(w)) == w
            assert P.compose(w, P.inverse(w)) == P.identity(n)
            assert P.is_permutation(w)


def test_length_equals_inversions():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            assert P.length(w) == len(P.inversions(w))


def test_lehmer_code_roundtrip():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            code = P.lehmer_code(w)
            assert P.code_to_perm(code) == w
            assert sum(code) == P.length(w)


def test_canonical_perm_examples():
    assert P.canonical_perm((2, 1, 3, 4, 5)) == (2, 1)
    assert P.canonical_perm((1, 2, 3, 4)) == (1,)
    assert P.canonical_perm((3, 1, 2, 4)) == (3, 1, 2)
    assert P.canonical_perm((1,)) == (1,)


def test_descents():
    w = (3, 1, 2, 5, 4)
    assert P.right_descents(w) == [1, 4]
    assert P.left_descents(w) == P.right_descents(P.inverse(w))


def test_simple_reflection_and_mult():
    n = 5
    s2 = P.simple_transposition(2, n)
    assert s2 == (1, 3, 2, 4, 5)
    w = (3, 1, 2, 5, 4)
    assert P.right_mult_s(w, 2) == (3, 2, 1, 5, 4)
    assert P.left_mult_s(w, 1) == tuple(2 if x == 1 else (1 if x == 2 else x) for x in w)


def test_pattern_avoidance():
    assert P.avoids_pattern((1, 2, 3), (3, 2, 1))
    assert P.contains_pattern((3, 2, 1), (3, 2, 1))
    assert P.contains_pattern((2, 4, 1, 3), (2, 1))


def test_reduced_words_reconstruct():
    rng = random.Random(0)
    for n in range(1, 6):
        for w in P.all_permutations(n):
            cw = RW.canonical_reduced_word(w)
            assert RW.is_reduced_word(cw, w)
            for _ in range(3):
                assert RW.is_reduced_word(RW.random_reduced_word(w, rng), w)


def test_all_reduced_words_w0_s3():
    assert set(RW.all_reduced_words((3, 2, 1))) == {(1, 2, 1), (2, 1, 2)}


def test_longest_element():
    assert P.longest_element(4) == (4, 3, 2, 1)
    assert P.length(P.longest_element(5)) == 10
