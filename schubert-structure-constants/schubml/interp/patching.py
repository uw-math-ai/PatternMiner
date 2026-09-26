"""Activation patching with matched packs (PROJECT.md 13.3, 13.4).

Given a positive triple ``A = (u, v, w_+)`` (*clean*) and a matched hard-zero
``B = (u, v, w_-)`` (*corrupt*) sharing the same factor pair ``(u, v)``:

1. cache the clean run on ``A``;
2. run the corrupt input ``B`` while patching one site's activation from the
   clean cache;
3. report the change in the positivity logit (and ``c_hat``):

       Delta = logit_pos(B patched with A) - logit_pos(B).

A large ``Delta`` flags a candidate positivity circuit (Level-4 evidence).
:func:`reverse_patch` swaps the roles (patch corrupt into clean) to distinguish
features that *cause* positivity from features that merely accompany it.

Because ``A`` and ``B`` share the factor stream, the *meaningful* patch sites are
the query / readout / slot sites that depend on ``w`` (factor-side sites are
identical and would give ``Delta ~ 0`` — still supported, as a sanity control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .hooks import HookedModule, HookPoint

# ---------------------------------------------------------------------------
# Patch-site descriptors
# ---------------------------------------------------------------------------
# Readout token layout in ``hook_readout`` (first ``Tg = n_slots + 3`` tokens):
#   0 -> SUPPORT, 1..K -> COUNT slots, K+1 -> ZERO, K+2 -> KNOWN.
READOUT_NAMES = {"support", "zero", "known"}


@dataclass
class PatchSite:
    """A site to patch.

    name:
        Hook-point name.
    head:
        For ``hook_z`` / ``hook_attn_out`` style activations ``[B, T, H, dh]``,
        restrict the patch to a single head index (others kept corrupt).
    token_index:
        Restrict the patch to a single token position of a ``[B, T, ...]`` stream.
    readout:
        For ``hook_readout``: ``"support" | "zero" | "known" | "count"`` or an int
        COUNT-slot index; convenience that resolves to the right ``token_index``.
    """

    name: str
    head: Optional[int] = None
    token_index: Optional[int] = None
    readout: Optional[object] = None

    def key(self) -> str:
        parts = [self.name]
        if self.readout is not None:
            parts.append(f"ro={self.readout}")
        if self.token_index is not None:
            parts.append(f"tok={self.token_index}")
        if self.head is not None:
            parts.append(f"h={self.head}")
        return "|".join(parts)


def _resolve_readout_index(readout, n_slots: int) -> Optional[object]:
    if readout is None:
        return None
    if readout == "support":
        return 0
    if readout == "zero":
        return n_slots + 1
    if readout == "known":
        return n_slots + 2
    if readout == "count":
        return slice(1, 1 + n_slots)
    if isinstance(readout, int):  # a single COUNT slot
        return 1 + readout
    raise ValueError(f"unknown readout selector {readout!r}")


def _make_patch_fn(src: torch.Tensor, site: PatchSite, n_slots: int):
    """Hook that overwrites (part of) the live activation with ``src``.

    Robust to a sequence-length mismatch between the clean source and the live
    (corrupt) activation: token-stream patches then act on the overlapping
    *leading* tokens only.  This matters because clean / corrupt may differ in
    the ``w`` (query) stream and hence in token count, while the global readout
    block — the meaningful patch region — always occupies the first
    ``Tg = n_slots + 3`` tokens, so the overlap covers it exactly.
    """
    tok = site.token_index
    if site.readout is not None:
        tok = _resolve_readout_index(site.readout, n_slots)

    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        s = src.to(x.dtype).to(x.device)
        sel = tok  # local copy so the closure variable is never rebound
        if sel is None and site.head is None:
            if s.shape == x.shape:
                return s
            # whole-tensor patch with a length mismatch -> patch the common prefix
            x = x.clone()
            t = min(x.shape[1], s.shape[1]) if x.dim() >= 2 else 0
            if t:
                x[:, :t] = s[:, :t]
            return x
        x = x.clone()
        # validate that the selected token index exists in both tensors
        if sel is not None and x.dim() >= 2:
            tmax = min(x.shape[1], s.shape[1])
            if isinstance(sel, slice):
                stop = min(sel.stop if sel.stop is not None else tmax, tmax)
                sel = slice(sel.start or 0, stop)
            elif sel >= tmax:
                return x  # token out of range for one side -> no-op
        if site.head is not None and sel is not None:
            x[:, sel, site.head] = s[:, sel, site.head]
        elif site.head is not None:  # patch one head across all tokens
            t = min(x.shape[1], s.shape[1])
            x[:, :t, site.head] = s[:, :t, site.head]
        else:  # patch one token (or token-slice) across all features/heads
            x[:, sel] = s[:, sel]
        return x

    return fn


def _logit_metrics(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"pos_logit": out["pos_logit"].detach(), "c_hat": out["c_hat"].detach()}


# ---------------------------------------------------------------------------
# Core patching
# ---------------------------------------------------------------------------
@torch.no_grad()
def activation_patch(
    model: HookedModule,
    clean_batch: Dict[str, torch.Tensor],
    corrupt_batch: Dict[str, torch.Tensor],
    site_names: Sequence[object],
    device: str = "cuda",
    *,
    reduce: str = "mean",
) -> Dict[str, object]:
    """Patch each site from the *clean* run into the *corrupt* run.

    ``site_names`` may mix bare strings (whole-activation patch) and
    :class:`PatchSite` descriptors.  Returns per-site

        Delta_pos = pos_logit(corrupt + clean-patch) - pos_logit(corrupt)
        Delta_c   = c_hat(...)      - c_hat(corrupt)

    plus the clean and corrupt baselines.  ``reduce='mean'`` averages over the
    batch; ``reduce='none'`` keeps the per-example vectors.
    """
    model.eval().to(device)
    clean = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in clean_batch.items()}
    corrupt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in corrupt_batch.items()}
    n_slots = int(getattr(model, "n_slots", model.cfg.n_slots))

    sites = [s if isinstance(s, PatchSite) else PatchSite(s) for s in site_names]
    names = list({s.name for s in sites})

    _, clean_cache = model.run_with_cache(clean, names=names)
    corrupt_out = model(corrupt)
    base = _logit_metrics(corrupt_out)
    clean_out = model(clean)
    clean_metrics = _logit_metrics(clean_out)

    def _red(t: torch.Tensor):
        return t.mean().item() if reduce == "mean" else t.cpu().tolist()

    results: Dict[str, object] = {}
    for site in sites:
        if site.name not in clean_cache:
            results[site.key()] = {"error": f"missing hook {site.name}"}
            continue
        fn = _make_patch_fn(clean_cache[site.name], site, n_slots)
        out = model.run_with_hooks(corrupt, fwd_hooks=[(site.name, fn)])
        m = _logit_metrics(out)
        results[site.key()] = {
            "delta_pos_logit": _red(m["pos_logit"] - base["pos_logit"]),
            "delta_c_hat": _red(m["c_hat"] - base["c_hat"]),
            "patched_pos_logit": _red(m["pos_logit"]),
        }
    results["_baseline"] = {
        "corrupt_pos_logit": _red(base["pos_logit"]),
        "clean_pos_logit": _red(clean_metrics["pos_logit"]),
        "corrupt_c_hat": _red(base["c_hat"]),
        "clean_c_hat": _red(clean_metrics["c_hat"]),
    }
    return results


@torch.no_grad()
def reverse_patch(
    model: HookedModule,
    clean_batch: Dict[str, torch.Tensor],
    corrupt_batch: Dict[str, torch.Tensor],
    site_names: Sequence[object],
    device: str = "cuda",
    *,
    reduce: str = "mean",
) -> Dict[str, object]:
    """Reverse patching (PROJECT.md 13.4): patch *corrupt* into *clean*.

    Measures whether injecting the hard-zero activation collapses the positive
    prediction:  ``Delta = pos_logit(clean + corrupt-patch) - pos_logit(clean)``.
    """
    return activation_patch(model, corrupt_batch, clean_batch, site_names,
                            device=device, reduce=reduce)


# ---------------------------------------------------------------------------
# Convenience site menus (PROJECT.md 13.3 patch-site list)
# ---------------------------------------------------------------------------
def residual_sites(n_encoder_layers: int, n_cross_layers: int) -> List[PatchSite]:
    """Residual stream by layer (encoder query stream + cross stream)."""
    sites = [PatchSite(f"encoder.blocks.{i}.hook_resid_post")
             for i in range(n_encoder_layers)]
    sites += [PatchSite("hook_query")]
    sites += [PatchSite(f"cross_blocks.{i}.hook_resid_post")
              for i in range(n_cross_layers)]
    return sites


def attn_head_sites(n_cross_layers: int, n_heads: int, *, kind: str = "cross") -> List[PatchSite]:
    """Per-head attention outputs (``hook_z``) of the cross blocks."""
    return [PatchSite(f"cross_blocks.{i}.{kind}_attn.hook_z", head=h)
            for i in range(n_cross_layers) for h in range(n_heads)]


def mlp_sites(n_cross_layers: int) -> List[PatchSite]:
    """MLP outputs (``hook_post``) by cross-block layer."""
    return [PatchSite(f"cross_blocks.{i}.mlp.hook_post") for i in range(n_cross_layers)]


def slot_sites() -> List[PatchSite]:
    """The nonnegative slot values."""
    return [PatchSite("slots.hook_slot_values")]


def readout_count_sites(n_slots: int) -> List[PatchSite]:
    """The COUNT readout tokens (whole block + named globals)."""
    sites = [PatchSite("hook_readout", readout="count"),
             PatchSite("hook_readout", readout="support"),
             PatchSite("hook_readout", readout="zero"),
             PatchSite("hook_readout", readout="known")]
    sites += [PatchSite("hook_readout", readout=k) for k in range(n_slots)]
    return sites


def default_patch_sites(model: HookedModule) -> List[PatchSite]:
    """A standard menu over the meaningful (``w``-dependent) sites."""
    cfg = model.cfg
    sites: List[PatchSite] = []
    sites += residual_sites(cfg.n_encoder_layers, cfg.n_cross_layers)
    sites += attn_head_sites(cfg.n_cross_layers, cfg.n_heads, kind="cross")
    sites += mlp_sites(cfg.n_cross_layers)
    sites += readout_count_sites(cfg.n_slots)
    sites += slot_sites()
    return sites
