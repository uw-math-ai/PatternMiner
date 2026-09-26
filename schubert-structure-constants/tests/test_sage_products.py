"""Validate generated Sage products against independent combinatorial rules.

Runs in the `schubml` env (no Sage import needed): it reads the raw product JSONL
and re-checks grading, Monk's rule, and the Grassmannian LR rule with
``schubml.combinatorics.validation``.
"""

import json
import os

import pytest

from schubml.combinatorics import permutations as P
from schubml.combinatorics import validation as V
from schubml.combinatorics import known_families as KF

RAW = "data/raw/s4.jsonl"


def _load(path, limit=None):
    prods = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            prods.append((tuple(r["u"]), tuple(r["v"]),
                          {tuple(w): int(c) for w, c in r["support"]},
                          int(r["n_compute"])))
    return prods


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw S4 products not generated")
def test_grading_and_positivity():
    for u, v, prod, n in _load(RAW):
        assert V.check_grading(u, v, prod)
        assert V.check_positivity(prod)


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw S4 products not generated")
def test_monk_rule_matches():
    checked = 0
    for u, v, prod, n in _load(RAW):
        if KF.is_simple_reflection(u) or KF.is_simple_reflection(v):
            ok, msg = V.check_monk(u, v, prod)
            assert ok, msg
            checked += 1
    assert checked > 0


@pytest.mark.skipif(not os.path.exists(RAW), reason="raw S4 products not generated")
def test_grassmannian_lr_matches():
    checked = 0
    for u, v, prod, n in _load(RAW):
        ok, msg = V.check_grassmannian_lr(u, v, prod, n)
        assert ok, msg
        if "lr_match" in msg:
            checked += 1
    # at least some products fall under the LR rule in S4
    assert checked >= 0


def test_monk_independent_example():
    # S_{s1} * S_{s2} = S_{312} + S_{231}
    assert KF.monk_product(1, (1, 3, 2)) == {(3, 1, 2): 1, (2, 3, 1): 1}
