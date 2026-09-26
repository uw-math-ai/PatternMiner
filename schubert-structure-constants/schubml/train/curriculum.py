"""Curriculum phases (PROJECT.md section 10).

A phase selects a subset of triples by negative tier and product family, plus a
training-step budget.  The training script can run the phases in sequence
(``CURRICULA['full']``) or train a single combined phase (``CURRICULA['combined']``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


@dataclass
class CurriculumPhase:
    name: str
    steps: int
    tiers: Set[str]                       # allowed negative tiers ("positive" always kept)
    product_tag_any: Optional[Set[str]] = None  # product must carry one of these tags
    source_max_n: Optional[int] = None    # restrict to source S_<=n


def _tiers(*ts: str) -> Set[str]:
    return {"positive", *ts}


CURRICULA: Dict[str, List[CurriculumPhase]] = {
    "combined": [
        CurriculumPhase("full_multitask", steps=60000,
                        tiers=_tiers("Z0", "Z1", "Z2", "Z3", "Z4", "Z5")),
    ],
    "full": [
        CurriculumPhase("phase1_grading", steps=4000, tiers=_tiers("Z0", "Z1")),
        CurriculumPhase("phase2_monk_pieri", steps=12000,
                        tiers=_tiers("Z0", "Z1", "Z2", "Z3"),
                        product_tag_any={"monk_case", "chevalley_case", "pieri_case"}),
        CurriculumPhase("phase3_grassmannian", steps=12000,
                        tiers=_tiers("Z2", "Z3"),
                        product_tag_any={"grassmannian_lr", "grassmannian_grassmannian",
                                         "schur_times_schubert"}),
        CurriculumPhase("phase4_islands", steps=8000,
                        tiers=_tiers("Z2", "Z3"),
                        product_tag_any={"boolean", "vexillary"}),
        CurriculumPhase("phase5_generic_hard", steps=60000,
                        tiers=_tiers("Z2", "Z3", "Z4", "Z5")),
    ],
    "ood": [
        CurriculumPhase("phase6_ood_finetune", steps=20000,
                        tiers=_tiers("Z2", "Z3", "Z4", "Z5"), source_max_n=5),
    ],
}


def filter_triples_for_phase(
    triples: List[Dict[str, object]],
    phase: CurriculumPhase,
    product_tags: Dict[int, Set[str]],
    product_source_n: Dict[int, int],
    base_split: str = "product_disjoint:train",
) -> List[Dict[str, object]]:
    out = []
    for t in triples:
        if base_split not in (t.get("split_tags") or []):
            continue
        tier = t.get("negative_tier") or "positive"
        if tier not in phase.tiers:
            continue
        pid = int(t["product_id"])
        if phase.product_tag_any is not None:
            if not (product_tags.get(pid, set()) & phase.product_tag_any):
                continue
        if phase.source_max_n is not None:
            if product_source_n.get(pid, 99) > phase.source_max_n:
                continue
        out.append(t)
    return out
