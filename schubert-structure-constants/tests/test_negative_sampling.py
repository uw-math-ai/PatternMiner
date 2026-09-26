import random

from schubml.combinatorics import permutations as P
from schubml.data import samplers


def test_random_perm_of_length():
    rng = random.Random(0)
    for _ in range(200):
        N = rng.randint(2, 8)
        Lmax = N * (N - 1) // 2
        L = rng.randint(0, Lmax)
        w = samplers.random_perm_of_length(N, L, rng)
        assert w is not None
        assert len(w) == N
        assert P.length(w) == L
        assert P.is_permutation(w)


def test_perturb_same_length_preserves_length():
    rng = random.Random(1)
    for w in [(2, 4, 1, 3), (3, 1, 4, 2, 5), tuple(range(6, 0, -1))]:
        L = P.length(w)
        for _ in range(20):
            c = samplers.perturb_same_length(w, rng, n_pairs=rng.randint(1, 4))
            assert P.length(c) == L
            assert P.is_permutation(c)


def test_negatives_not_in_support_and_tiers():
    # product s2 * s1 in S3: support {231, 312}; negatives must avoid it
    rng = random.Random(0)
    u, v = (1, 3, 2), (2, 1)
    support = {(2, 3, 1), (3, 1, 2)}
    pool = samplers.PermPool(3, rng)
    negs = samplers.sample_matched_negatives(
        u, v, support, pool, n_compute=3,
        per_tier={"Z0": 1, "Z1": 1, "Z2": 2, "Z3": 2, "Z4": 1, "Z5": 1}, rng=rng)
    for ng in negs:
        assert P.canonical_perm(ng["w"]) not in {P.canonical_perm(w) for w in support}
        assert ng["tier"] in samplers.TIER_ORDER


def test_tier_grading_and_bruhat_logic():
    # Z0 must be length-mismatched; non-Z0 must match length L
    rng = random.Random(2)
    u, v = (2, 1), (2, 1)  # both s1, L = 2
    support = {(3, 1, 2)}
    pool = samplers.PermPool(4, rng)
    negs = samplers.sample_matched_negatives(
        u, v, support, pool, n_compute=4,
        per_tier={"Z0": 2, "Z1": 2, "Z2": 4, "Z3": 4, "Z4": 2, "Z5": 2}, rng=rng,
        max_candidates=80)
    L = P.length(P.pad(u, 4)) + P.length(P.pad(v, 4))
    for ng in negs:
        w = ng["w"]
        if ng["tier"] == "Z0":
            assert P.length(w) != L
        else:
            assert P.length(w) == L
        if ng["tier"] in ("Z2", "Z3", "Z4", "Z5"):
            # hard zeros satisfy the Bruhat necessary conditions
            assert samplers.passes_bruhat(P.pad(u, 4), P.pad(v, 4), w)
