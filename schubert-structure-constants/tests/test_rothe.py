from schubml.combinatorics import permutations as P
from schubml.combinatorics import rothe


def test_rothe_cell_count_equals_length():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            assert len(rothe.rothe_cells(w)) == P.length(w)


def test_rothe_row_lengths_equal_code():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            assert rothe.rothe_row_lengths(w) == P.lehmer_code(w)


def test_rothe_col_lengths_equal_inverse_code():
    for n in range(1, 6):
        for w in P.all_permutations(n):
            assert rothe.rothe_col_lengths(w) == P.lehmer_code(P.inverse(w))


def test_rothe_transpose_is_inverse():
    # D(w^{-1}) == transpose of D(w)
    for n in range(1, 6):
        for w in P.all_permutations(n):
            cells = set(rothe.rothe_cells(w))
            inv_cells = set(rothe.rothe_cells(P.inverse(w)))
            assert {(j, i) for (i, j) in cells} == inv_cells


def test_dominant_iff_partition_shape():
    from schubml.combinatorics import known_families as KF
    for n in range(1, 6):
        for w in P.all_permutations(n):
            assert rothe.is_partition_shape(w) == KF.is_dominant(w)


def test_rothe_matrix_and_ascii_runs():
    w = (2, 4, 1, 3)
    M = rothe.rothe_diagram_matrix(w)
    assert int(M.sum()) == P.length(w)
    s = rothe.rothe_ascii(w)
    assert isinstance(s, str) and "\n" in s
