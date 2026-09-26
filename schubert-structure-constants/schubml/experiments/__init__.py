"""Experiments: baselines, view ablations, and held-out evaluation tables.

* :mod:`baselines` -- weak + classical-ML positivity baselines and the
  ``run_baselines`` orchestrator (PROJECT.md 11).
* :mod:`ablations` -- tokenizer-view ablation specs + run summarisation
  (PROJECT.md 7.1, 11.3).
* :mod:`heldout` -- required-evaluation-table assembly + exact known-rule
  (Monk) baseline (PROJECT.md 11.4, 12.2).
"""

from . import ablations, baselines, heldout

__all__ = ["baselines", "ablations", "heldout"]
