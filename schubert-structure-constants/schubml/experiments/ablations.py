"""Tokenizer-view ablation specs + run summarisation (PROJECT.md 7.1, 11.3).

The transformer is fed several Schubert-relevant *views* (one-line, Rothe,
inversions, rank-matrix, reduced-word) and can be ablated to see which views are
necessary (PROJECT.md 7.1).  This module is intentionally lightweight: it does
*no* training.  It only

* enumerates the ablation configurations as plain dicts that the training script
  (``scripts/train.py`` / ``schubml.models.product_query_transformer.ModelConfig``)
  can consume, and
* loads + tabulates the eval ``metrics.json`` of finished runs for comparison.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

# The view flags understood by the tokenizer / ModelConfig.  Each ablation
# toggles a subset of these on or off relative to the full multi-view model.
ALL_VIEWS = ("one_line", "rothe", "inversions", "rank_matrix", "reduced_word")


def _cfg(name: str, description: str, *, views: Dict[str, bool],
         counting_slots: bool = True, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Assemble one ablation spec dict (the unit the train script consumes)."""
    full_views = {v: False for v in ALL_VIEWS}
    full_views.update(views)
    spec: Dict[str, object] = {
        "name": name,
        "description": description,
        "views": full_views,
        "counting_slots": counting_slots,
    }
    if extra:
        spec.update(extra)
    return spec


def view_ablation_configs() -> List[Dict[str, object]]:
    """The view / counting-slot ablation suite (PROJECT.md 7.1, 11.3).

    Returns a list of spec dicts.  Each spec names the views to enable so that a
    single field (``views``) plus ``counting_slots`` describes the configuration.
    The suite covers:

    * ``one_line`` -- the minimal raw permutation view only (lower bound),
    * ``+rothe`` -- add Rothe-cell tokens (the most important local-rule view),
    * ``+inversions`` -- add inversion tokens (Bruhat / length circuits),
    * ``+rank`` -- add rank-matrix tokens (geometric incidence inequalities),
    * ``+reduced`` -- add reduced-word tokens (braid-invariant structure),
    * ``full`` -- all views together (the reference model),
    * ``no_counting_slots`` -- full views but with the nonnegative counting
      bottleneck removed (tests whether the slots help/are needed).
    """
    specs: List[Dict[str, object]] = [
        _cfg("one_line", "Raw one-line permutation view only (baseline lower bound).",
             views={"one_line": True}),
        _cfg("plus_rothe", "One-line + Rothe-cell tokens (local tile rules).",
             views={"one_line": True, "rothe": True}),
        _cfg("plus_inversions", "One-line + inversion tokens (Bruhat/length circuits).",
             views={"one_line": True, "inversions": True}),
        _cfg("plus_rank", "One-line + rank-matrix tokens (geometric incidence).",
             views={"one_line": True, "rank_matrix": True}),
        _cfg("plus_reduced", "One-line + reduced-word tokens (braid invariance).",
             views={"one_line": True, "reduced_word": True}),
        _cfg("full", "All views (one-line + Rothe + inversions + rank + reduced).",
             views={v: True for v in ALL_VIEWS}),
        _cfg("no_counting_slots",
             "All views, but the nonnegative counting bottleneck is disabled.",
             views={v: True for v in ALL_VIEWS}, counting_slots=False),
    ]
    return specs


def ablation_config_by_name(name: str) -> Dict[str, object]:
    """Look up a single ablation spec by ``name`` (raises ``KeyError`` if absent)."""
    for spec in view_ablation_configs():
        if spec["name"] == name:
            return spec
    raise KeyError(f"unknown ablation config {name!r}; "
                   f"available: {[s['name'] for s in view_ablation_configs()]}")


# ---------------------------------------------------------------------------
# Summarising finished runs
# ---------------------------------------------------------------------------
def _load_eval_metrics(run_dir: str) -> Optional[Dict[str, object]]:
    """Load a run's ``eval/metrics.json`` (the file ``scripts/evaluate.py`` writes)."""
    for candidate in (
        os.path.join(run_dir, "eval", "metrics.json"),
        os.path.join(run_dir, "metrics.json"),
    ):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


# the headline metrics pulled from each run's eval table for the comparison
_HEADLINE_KEYS = [
    ("product_disjoint_test", "pos_accuracy"),
    ("product_disjoint_test", "pos_mcc"),
    ("hardzero_Z45", "pos_accuracy"),
    ("hardzero_Z45", "pos_mcc"),
    ("product_disjoint_test", "coeff_exact_acc"),
]


def summarize_ablation(run_dirs: Sequence[str]) -> Dict[str, object]:
    """Tabulate eval metrics across ablation ``run_dirs`` (PROJECT.md 11.3, 12).

    Each ``run_dir`` is the output directory of a training run (it should contain
    ``eval/metrics.json``).  Returns ``{"columns": [...], "rows": {run: {...}},
    "missing": [...]}`` -- a lightweight, JSON-serialisable comparison table that
    surfaces the product-disjoint and hard-zero headline numbers side by side.
    """
    columns = [f"{block}.{metric}" for block, metric in _HEADLINE_KEYS]
    rows: Dict[str, Dict[str, object]] = {}
    missing: List[str] = []
    for run_dir in run_dirs:
        metrics = _load_eval_metrics(run_dir)
        run_name = os.path.basename(os.path.normpath(run_dir))
        if metrics is None:
            missing.append(run_dir)
            continue
        row: Dict[str, object] = {}
        for (block, metric), col in zip(_HEADLINE_KEYS, columns):
            blk = metrics.get(block, {})
            row[col] = blk.get(metric) if isinstance(blk, dict) else None
        rows[run_name] = row
    return {"columns": columns, "rows": rows, "missing": missing}


def format_ablation_table(summary: Dict[str, object]) -> str:
    """Render :func:`summarize_ablation` output as a fixed-width table."""
    columns = summary.get("columns", [])
    rows = summary.get("rows", {})
    short = [c.replace("product_disjoint_test", "pd").replace("hardzero_Z45", "hz45")
             for c in columns]
    name_w = max([len("run")] + [len(r) for r in rows]) if rows else len("run")
    header = f"{'run':<{name_w}} " + " ".join(f"{c:>16}" for c in short)
    lines = [header, "-" * len(header)]
    for run_name, row in rows.items():
        cells = []
        for c in columns:
            v = row.get(c)
            cells.append(f"{v:>16.3f}" if isinstance(v, (int, float)) else f"{'--':>16}")
        lines.append(f"{run_name:<{name_w}} " + " ".join(cells))
    if summary.get("missing"):
        lines.append("")
        lines.append("missing eval metrics: " + ", ".join(summary["missing"]))
    return "\n".join(lines)
