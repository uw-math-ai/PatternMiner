"""A small TransformerLens-style hook system.

``HookPoint`` is an identity module that can intercept/replace the tensor passing
through it.  ``HookedModule`` provides ``run_with_cache`` and ``run_with_hooks`` so
the rest of the project can do activation caching, activation patching, path
patching, and ablation (PROJECT.md 8.8, 13).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

HookFn = Callable[[torch.Tensor, "HookPoint"], Optional[torch.Tensor]]


class HookPoint(nn.Module):
    """Identity module that applies registered hook functions in order."""

    def __init__(self) -> None:
        super().__init__()
        self.name: Optional[str] = None
        self._hooks: List[HookFn] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self._hooks:
            out = fn(x, self)
            if out is not None:
                x = out
        return x

    def add_hook(self, fn: HookFn) -> Callable[[], None]:
        self._hooks.append(fn)

        def _remove() -> None:
            if fn in self._hooks:
                self._hooks.remove(fn)

        return _remove

    def clear_hooks(self) -> None:
        self._hooks = []


class HookedModule(nn.Module):
    """Mixin adding cache/hook orchestration over contained :class:`HookPoint`s."""

    def setup_hook_names(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name

    def hook_points(self) -> Dict[str, HookPoint]:
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, HookPoint)
        }

    def reset_hooks(self) -> None:
        for hp in self.hook_points().values():
            hp.clear_hooks()

    def add_hook(self, name: str, fn: HookFn) -> Callable[[], None]:
        return self.hook_points()[name].add_hook(fn)

    @contextmanager
    def hooks(self, fwd_hooks: List[Tuple[str, HookFn]]):
        removers = [self.add_hook(name, fn) for name, fn in fwd_hooks]
        try:
            yield self
        finally:
            for r in removers:
                r()

    def run_with_hooks(self, *args, fwd_hooks: List[Tuple[str, HookFn]] = (), **kwargs):
        with self.hooks(list(fwd_hooks)):
            return self(*args, **kwargs)

    def run_with_cache(
        self,
        *args,
        names: Optional[List[str]] = None,
        retain_grad: bool = False,
        **kwargs,
    ) -> Tuple[object, Dict[str, torch.Tensor]]:
        cache: Dict[str, torch.Tensor] = {}
        points = self.hook_points()
        wanted = set(names) if names is not None else set(points.keys())

        def make(nm: str) -> HookFn:
            def _cache(x: torch.Tensor, hook: HookPoint) -> None:
                cache[nm] = x if retain_grad else x.detach()
                return None

            return _cache

        fwd_hooks = [(nm, make(nm)) for nm in points if nm in wanted]
        with self.hooks(fwd_hooks):
            out = self(*args, **kwargs)
        return out, cache


# ---------------------------------------------------------------------------
# Common hook factories for patching / ablation
# ---------------------------------------------------------------------------
def patch_hook(source_activation: torch.Tensor) -> HookFn:
    """Replace the activation with ``source_activation`` (activation patching)."""

    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return source_activation.to(x.dtype).to(x.device)

    return fn


def patch_positions_hook(source_activation: torch.Tensor, index) -> HookFn:
    """Patch only ``x[index]`` from ``source_activation`` (path/position patching)."""

    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        x = x.clone()
        x[index] = source_activation[index].to(x.dtype).to(x.device)
        return x

    return fn


def zero_ablate_hook() -> HookFn:
    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return torch.zeros_like(x)

    return fn


def mean_ablate_hook(mean_activation: torch.Tensor) -> HookFn:
    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return mean_activation.to(x.dtype).to(x.device).expand_as(x).clone()

    return fn


def scale_hook(factor: float) -> HookFn:
    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        return x * factor

    return fn
