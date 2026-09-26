"""Nonnegative sparse counting bottleneck (PROJECT.md 8.6).

The coefficient prediction is a sum of nonnegative slot values

    c_hat = sum_a z_a,   z_a = softplus(slot_logit_a) * gate_a >= 0,

where each ``z_a`` is meant to behave like a counted witness contribution.  The
slot pre-activations, values, and gates are exposed as hook points for
slot interpretation, ablation, and patching (PROJECT.md 13.8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interp.hooks import HookPoint


class CountingSlots(nn.Module):
    def __init__(self, d_model: int, n_slots: int, use_gate: bool = True):
        super().__init__()
        self.n_slots = n_slots
        self.use_gate = use_gate
        self.value_head = nn.Linear(d_model, 1)
        self.gate_head = nn.Linear(d_model, 1) if use_gate else None
        self.hook_slot_preact = HookPoint()   # [B, K]
        self.hook_slot_values = HookPoint()    # [B, K]
        self.hook_slot_gates = HookPoint()     # [B, K]

    def forward(self, count_tokens: torch.Tensor):
        """``count_tokens``: [B, K, d] -> dict with c_hat [B], slot values [B, K]."""
        preact = self.value_head(count_tokens).squeeze(-1)  # [B, K]
        preact = self.hook_slot_preact(preact)
        values = F.softplus(preact)
        if self.use_gate:
            gates = torch.sigmoid(self.gate_head(count_tokens).squeeze(-1))
            gates = self.hook_slot_gates(gates)
            values = values * gates
        else:
            gates = torch.ones_like(values)
        values = self.hook_slot_values(values)
        c_hat = values.sum(dim=-1)  # [B]
        return {"c_hat": c_hat, "slot_values": values, "slot_gates": gates, "slot_preact": preact}
