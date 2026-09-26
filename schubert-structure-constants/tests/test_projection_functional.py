import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from r2_unbounded_certificate import projected_maximum_position


def test_seed_functional_uses_the_sorted_third_factor():
    w = (2, 5, 6, 1, 3, 4)
    assert w.index(1) + 1 == 4
    assert projected_maximum_position(w, (1, 3, 5)) == 5
    assert projected_maximum_position(w, (1, 2, 3, 4, 5)) == 4
    assert projected_maximum_position(w, (2, 4)) == 4
    assert projected_maximum_position(w, ()) == 6


def test_correct_index_is_inherited_under_the_recursive_construction():
    w = (2, 5, 6, 1, 3, 4)
    E = (1, 3, 5)
    for q in range(5, 10):
        assert projected_maximum_position(w, E) == q
        w = (2,) + tuple(x + (x >= 2) for x in w)
        E = (1,) + tuple(e + 1 for e in E)
