"""Conventions and nonboolean cancellation controls for the research oracle."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/2026-09-09-block-operators'))
from operators import SubwordOperators, divided, from_word, reflect


def test_repeated_root_divided_difference_is_signed():
    assert divided({(1, 0): 1}, 0) == {(0, 0): -2}
    assert reflect(reflect({(2, 3, 1): 7}, 1), 1) == {(2, 3, 1): 7}


def test_simple_square_has_the_positive_root_convention():
    s = (2, 1)
    assert SubwordOperators((0,), 2)(s, s) == {(1,): 1}


def test_both_reduced_embeddings_are_required_for_cancellation():
    op = SubwordOperators((0, 1, 0), 3)
    s1 = from_word((0,), 3)
    assert op(s1, from_word((0, 1), 3)) == {(0, 0): 1}
    assert op(s1, from_word((1, 0), 3)) == {}


def test_nonreduced_output_rejected():
    with pytest.raises(ValueError, match='reduced'):
        SubwordOperators((0, 0), 3)
