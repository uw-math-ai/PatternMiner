"""Mechanistic interpretability + symbolic distillation (PROJECT.md 13-14).

Submodules:

* :mod:`~schubml.interp.hooks`         -- TransformerLens-style hook system.
* :mod:`~schubml.interp.probes`        -- linear / logistic probes of known features.
* :mod:`~schubml.interp.patching`      -- activation patching with matched packs.
* :mod:`~schubml.interp.path_patching` -- sender -> receiver circuit isolation.
* :mod:`~schubml.interp.sae`           -- ReLU sparse autoencoders.
* :mod:`~schubml.interp.feature_mining`-- slot / SAE feature cards.
* :mod:`~schubml.interp.symbolic`      -- predicate DSL + rule learners + cards.
"""

from __future__ import annotations

from . import (  # noqa: F401
    feature_mining,
    hooks,
    patching,
    path_patching,
    probes,
    sae,
    symbolic,
)
from .hooks import (  # noqa: F401
    HookedModule,
    HookPoint,
    mean_ablate_hook,
    patch_hook,
    patch_positions_hook,
    scale_hook,
    zero_ablate_hook,
)
from .patching import (  # noqa: F401
    PatchSite,
    activation_patch,
    reverse_patch,
)
from .path_patching import Node, path_patch  # noqa: F401
from .probes import Site, collect_activations, run_probes  # noqa: F401
from .sae import SAE, SAEConfig, encode, gather_activations, train_sae  # noqa: F401
from .symbolic import (  # noqa: F401
    Rule,
    learn_positivity_rules,
    predicate_features,
    validate_rule,
    write_conjecture_card,
)

__all__ = [
    "hooks", "probes", "patching", "path_patching", "sae", "feature_mining", "symbolic",
    "HookedModule", "HookPoint", "patch_hook", "patch_positions_hook",
    "zero_ablate_hook", "mean_ablate_hook", "scale_hook",
    "Site", "collect_activations", "run_probes",
    "PatchSite", "activation_patch", "reverse_patch",
    "Node", "path_patch",
    "SAE", "SAEConfig", "train_sae", "encode", "gather_activations",
    "Rule", "predicate_features", "learn_positivity_rules", "validate_rule",
    "write_conjecture_card",
]
