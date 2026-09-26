"""Independent validation of Schubert structure-constant products.

These checks do not depend on the backend (Sage) that produced the product;
they re-derive necessary conditions and known-rule values from scratch, so a
disagreement flags a generator bug (PROJECT.md 6.3, 19.2).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .permutations import Perm, canonical_perm, length, right_descents
from .known_families import (
    grassmannian_lr_product,
    is_grassmannian,
    is_simple_reflection,
    monk_product,
)


def check_grading(u: Sequence[int], v: Sequence[int], product: Dict[Perm, int]) -> bool:
    """Every ``w`` in the support has ``length(w) = length(u) + length(v)``."""
    target = length(u) + length(v)
    return all(length(w) == target for w in product)


def check_positivity(product: Dict[Perm, int]) -> bool:
    """All coefficients are positive integers."""
    return all(isinstance(c, int) and c > 0 for c in product.values())


def check_monk(u: Sequence[int], v: Sequence[int], product: Dict[Perm, int]) -> Tuple[bool, str]:
    """If one factor is a simple reflection ``s_r``, compare to Monk's rule.

    The non-simple factor is padded to the ambient size of the product so that
    ``monk_product`` enumerates covers which move trailing fixed points (otherwise
    Monk recall is spuriously low for canonical inputs).
    """
    from .permutations import pad

    n = max([len(w) for w in product] + [len(u), len(v)], default=1)
    if is_simple_reflection(u):
        r = right_descents(u)[0]
        expected = monk_product(r, pad(v, n))
    elif is_simple_reflection(v):
        r = right_descents(v)[0]
        expected = monk_product(r, pad(u, n))
    else:
        return True, "monk_not_applicable"
    got = {canonical_perm(k): val for k, val in product.items()}
    expected = {canonical_perm(k): val for k, val in expected.items()}
    if got == expected:
        return True, "monk_match"
    return False, f"monk_mismatch expected={expected} got={got}"


def check_grassmannian_lr(
    u: Sequence[int], v: Sequence[int], product: Dict[Perm, int], n: int
) -> Tuple[bool, str]:
    """If ``(u, v)`` are same-descent Grassmannian, compare to the LR rule."""
    expected = grassmannian_lr_product(u, v, n)
    if expected is None:
        return True, "lr_not_applicable"
    got = {canonical_perm(k): val for k, val in product.items() if val != 0}
    expected = {canonical_perm(k): val for k, val in expected.items() if val != 0}
    if got == expected:
        return True, "lr_match"
    return False, f"lr_mismatch expected={expected} got={got}"


def validate_product(
    u: Sequence[int], v: Sequence[int], product: Dict[Perm, int], n: int
) -> Dict[str, object]:
    """Run all applicable checks; return a structured report."""
    report: Dict[str, object] = {}
    report["grading_ok"] = check_grading(u, v, product)
    report["positivity_ok"] = check_positivity(product)
    monk_ok, monk_msg = check_monk(u, v, product)
    report["monk_ok"] = monk_ok
    report["monk_msg"] = monk_msg
    lr_ok, lr_msg = check_grassmannian_lr(u, v, product, n)
    report["lr_ok"] = lr_ok
    report["lr_msg"] = lr_msg
    report["all_ok"] = bool(
        report["grading_ok"] and report["positivity_ok"] and monk_ok and lr_ok
    )
    return report


def check_commutativity(
    prod_uv: Dict[Perm, int], prod_vu: Dict[Perm, int]
) -> bool:
    """``c_{u,v}^w == c_{v,u}^w`` for all ``w`` (canonicalized)."""
    a = {canonical_perm(k): v for k, v in prod_uv.items()}
    b = {canonical_perm(k): v for k, v in prod_vu.items()}
    return a == b
