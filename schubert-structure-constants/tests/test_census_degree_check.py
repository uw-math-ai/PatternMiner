import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from s1_atom_census import degree_tight_coefficient, face_feasible, cosets
import p1_insertion_ladder as P


def test_positive_control_does_not_vanish_by_extra_parabolic_degree():
    u = (1, 2, 3, 4, 5, 6)
    v = w = (1, 3, 2, 4, 5, 6)
    assert degree_tight_coefficient(u, v, w) == 1
    assert face_feasible(u, v, w, (2, 4), 6)
    with pytest.raises(ValueError, match="equal degrees"):
        degree_tight_coefficient(u, v, (3, 1, 4, 2, 6, 5))


def test_coefficient_matches_projection_for_every_degree_tight_grassmannian_triple_s4():
    reps = cosets((2,), 4)
    seen = set()
    for u in reps:
        for v in reps:
            for w in reps:
                if P.length(u) + P.length(v) == P.length(w):
                    nonzero = degree_tight_coefficient(u, v, w) > 0
                    assert nonzero == face_feasible(u, v, w, (2,), 4)
                    seen.add(nonzero)
    assert seen == {False, True}
