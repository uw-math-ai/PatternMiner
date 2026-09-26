"""Path patching: sender -> receiver circuit isolation (PROJECT.md 13.5).

Standard three-run method (Wang et al., "IOI").  To measure the effect that
flows *along the direct path* from a sender component to a receiver (or to the
logits) — while every other path stays at its clean value:

1. **clean** run on ``A``: cache every site (so receivers can be frozen clean).
2. **corrupt** run on ``B``: cache the sender's corrupt activation.
3. **patched** run: replay the clean input ``A`` but

   * overwrite the *sender* hook with its corrupt value, and
   * pin every *receiver-input* hook back to its clean cache,

   so only the sender -> receiver edge carries the corrupt signal.  Read the
   effect at the receiver's activation (or, if ``receiver is None``, at the
   positivity logit / ``c_hat``).

This isolates a single edge of the computation graph, the evidence needed to
claim a *circuit* (Level-5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .hooks import HookedModule, HookPoint
from .patching import PatchSite, _make_patch_fn, _resolve_readout_index


@dataclass
class Node:
    """A component in the computation graph used as sender or receiver.

    ``name`` is a hook-point name; ``head`` / ``token_index`` / ``readout``
    optionally restrict to a head / token / readout token (same semantics as
    :class:`~schubml.interp.patching.PatchSite`).
    """

    name: str
    head: Optional[int] = None
    token_index: Optional[int] = None
    readout: Optional[object] = None

    def to_site(self) -> PatchSite:
        return PatchSite(self.name, head=self.head,
                         token_index=self.token_index, readout=self.readout)

    def key(self) -> str:
        return self.to_site().key()


def _freeze_clean_fn(clean_act: torch.Tensor):
    """Hook that pins an activation to its clean cached value.

    Both runs here use the *same* (clean) input, so shapes match exactly; the
    common-prefix guard is a defensive no-op for the matched-shape case.
    """

    def fn(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        s = clean_act.to(x.dtype).to(x.device)
        if s.shape == x.shape:
            return s
        x = x.clone()
        t = min(x.shape[1], s.shape[1]) if x.dim() >= 2 else 0
        if t:
            x[:, :t] = s[:, :t]
        return x

    return fn


def _select(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"pos_logit": out["pos_logit"].detach(), "c_hat": out["c_hat"].detach()}


@torch.no_grad()
def path_patch(
    model: HookedModule,
    clean: Dict[str, torch.Tensor],
    corrupt: Dict[str, torch.Tensor],
    sender: object,
    receiver: Optional[object] = None,
    device: str = "cuda",
    *,
    freeze: Optional[Sequence[str]] = None,
    reduce: str = "mean",
) -> Dict[str, object]:
    """Patch the ``sender -> receiver`` direct path; measure the downstream effect.

    Standard method.  When a ``receiver`` is given the effect is isolated to the
    ``sender -> receiver -> logits`` path with two surgical runs:

    * **record run** on the *clean* input with only the sender set to corrupt
      (plus any user ``freeze`` hooks pinned clean): the receiver now reflects the
      corrupt signal *only via paths through the sender*.  Capture the receiver's
      activation ``recv*``.
    * **measure run** on the clean input with the receiver patched to ``recv*``
      and nothing else changed: read the logits.

    ``delta_pos_logit = pos_logit(measure) - pos_logit(clean)`` is then the effect
    that flows specifically along ``sender -> receiver -> output``.  With
    ``receiver is None`` this collapses to direct activation patching of the
    sender into the logits.

    Parameters
    ----------
    sender:
        :class:`Node` (or hook name) whose corrupt output is injected.
    receiver:
        :class:`Node` whose state carries the effect.  ``None`` -> measure at the
        positivity logit / ``c_hat`` directly.
    freeze:
        Optional extra hook names to pin to clean during the *record* run (to
        knock out specific off-path routes); the sender and receiver are handled
        automatically and must not be listed.

    Returns the logit deltas plus the clean / corrupt baselines (and, for an
    intermediate receiver, how far the receiver activation moved).
    """
    model.eval().to(device)
    clean = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in clean.items()}
    corrupt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in corrupt.items()}
    n_slots = int(getattr(model, "n_slots", model.cfg.n_slots))

    snode = sender if isinstance(sender, Node) else Node(str(sender))
    rnode = None if receiver is None else (receiver if isinstance(receiver, Node) else Node(str(receiver)))

    # 1) clean cache (everything we may need to read / freeze)
    _, clean_cache = model.run_with_cache(clean)
    # 2) corrupt cache (sender value)
    _, corrupt_cache = model.run_with_cache(corrupt, names=[snode.name])
    if snode.name not in corrupt_cache:
        return {"error": f"missing sender hook {snode.name}"}

    clean_sel = _select(model(clean))
    corrupt_sel = _select(model(corrupt))

    def _red(t: torch.Tensor):
        return t.mean().item() if reduce == "mean" else t.cpu().tolist()

    sender_fn = _make_patch_fn(corrupt_cache[snode.name], snode.to_site(), n_slots)
    freeze_names = [n for n in (freeze or []) if n not in (snode.name,) and n in clean_cache]

    if rnode is None:
        # direct path to output: just set the sender corrupt on the clean input.
        fwd = [(snode.name, sender_fn)]
        for fn_name in freeze_names:
            fwd.append((fn_name, _freeze_clean_fn(clean_cache[fn_name])))
        patched_sel = _select(model.run_with_hooks(clean, fwd_hooks=fwd))
        return {
            "sender": snode.key(), "receiver": "logits",
            "delta_pos_logit": _red(patched_sel["pos_logit"] - clean_sel["pos_logit"]),
            "delta_c_hat": _red(patched_sel["c_hat"] - clean_sel["c_hat"]),
            "clean_pos_logit": _red(clean_sel["pos_logit"]),
            "corrupt_pos_logit": _red(corrupt_sel["pos_logit"]),
            "patched_pos_logit": _red(patched_sel["pos_logit"]),
        }

    # --- record run: capture the receiver under (sender=corrupt, rest clean) ---
    captured: Dict[str, torch.Tensor] = {}

    def _grab(x: torch.Tensor, hook: HookPoint):
        captured["recv"] = x.detach()
        return None

    record_hooks: List[Tuple[str, object]] = [(snode.name, sender_fn)]
    for fn_name in freeze_names:
        record_hooks.append((fn_name, _freeze_clean_fn(clean_cache[fn_name])))
    record_hooks.append((rnode.name, _grab))
    model.run_with_hooks(clean, fwd_hooks=record_hooks)

    # --- measure run: inject recv* at the receiver on an otherwise-clean run ---
    recv_star = captured["recv"]
    recv_patch_fn = _make_patch_fn(recv_star, rnode.to_site(), n_slots)
    patched_sel = _select(model.run_with_hooks(clean, fwd_hooks=[(rnode.name, recv_patch_fn)]))

    res: Dict[str, object] = {
        "sender": snode.key(),
        "receiver": rnode.key(),
        "delta_pos_logit": _red(patched_sel["pos_logit"] - clean_sel["pos_logit"]),
        "delta_c_hat": _red(patched_sel["c_hat"] - clean_sel["c_hat"]),
        "clean_pos_logit": _red(clean_sel["pos_logit"]),
        "corrupt_pos_logit": _red(corrupt_sel["pos_logit"]),
        "patched_pos_logit": _red(patched_sel["pos_logit"]),
    }
    if "recv" in captured:
        # how much the receiver activation moved (L2, averaged over the batch)
        clean_recv = _resolve_node_activation(clean_cache[rnode.name], rnode, n_slots)
        recv = captured["recv"]
        moved = (_resolve_node_activation(recv, rnode, n_slots) - clean_recv)
        moved = moved.reshape(moved.shape[0], -1) if moved.dim() > 1 else moved.reshape(-1, 1)
        res["receiver_delta_l2"] = float(moved.norm(dim=-1).mean().item())
    return res


def _resolve_node_activation(x: torch.Tensor, node: Node, n_slots: int) -> torch.Tensor:
    """Slice a node's activation down to the selected head/token for readout."""
    tok = node.token_index
    if node.readout is not None:
        tok = _resolve_readout_index(node.readout, n_slots)
    if tok is not None and node.head is not None:
        return x[:, tok, node.head]
    if tok is not None:
        return x[:, tok]
    if node.head is not None:
        return x[:, :, node.head]
    return x


@torch.no_grad()
def path_patch_many(
    model: HookedModule,
    clean: Dict[str, torch.Tensor],
    corrupt: Dict[str, torch.Tensor],
    senders: Sequence[object],
    receiver: Optional[object] = None,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, object]:
    """Sweep several senders against a fixed receiver; return ``{sender_key: res}``."""
    out: Dict[str, object] = {}
    for s in senders:
        res = path_patch(model, clean, corrupt, s, receiver, device=device, **kwargs)
        key = res.get("sender", str(s))
        out[key] = res
    return out
