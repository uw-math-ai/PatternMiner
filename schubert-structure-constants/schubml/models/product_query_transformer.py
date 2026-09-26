"""The hooked product-query transformer (PROJECT.md section 8).

Flow:

1. Encode the FACTOR stream (u-tokens ++ v-tokens, shared role) with a
   positional-encoding-free set transformer -> ``E_F`` (symmetric in ``u, v``).
2. Encode the QUERY stream (w-tokens) with the same encoder (role distinguishes).
3. Readout tokens ``[SUPPORT, COUNT_1..K, ZERO, KNOWN]`` plus ``E_W`` cross-attend
   into ``E_F`` over several cross blocks.
4. The K COUNT tokens drive the nonnegative counting bottleneck; the heads read
   the readout tokens.

Symmetry ``c(u,v,w) = c(v,u,w)`` holds by construction: ``E_F`` is the encoding of
the *set* of factor tokens, and cross-attention into a set is order-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..data.labels import N_FAMILY_TAGS
from ..data.tokenizers import N_FIELDS
from ..interp.hooks import HookedModule, HookPoint
from .blocks import CrossBlock
from .counting_slots import CountingSlots
from .embeddings import FieldEmbedding
from .heads import Heads
from .permutation_encoder import RoleEncoder
from .rothe_encoder import ComparisonBias


@dataclass
class ModelConfig:
    d_model: int = 128
    n_encoder_layers: int = 4
    n_cross_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 512
    dropout: float = 0.0
    n_ctx: int = 14
    n_slots: int = 32
    n_family_tags: int = N_FAMILY_TAGS
    use_gate: bool = True
    use_mod: bool = True
    use_bins: bool = True
    use_exact_small: bool = True
    use_aux: bool = True
    use_comparison_bias: bool = False


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [B, T, d]; mask: [B, T] bool
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class ProductQueryTransformer(HookedModule):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = FieldEmbedding(cfg.d_model, cfg.n_ctx)
        self.encoder = RoleEncoder(
            cfg.d_model, cfg.n_encoder_layers, cfg.n_heads, cfg.d_mlp,
            cfg.n_ctx, cfg.dropout, embedding=self.embed,
        )
        self.cross_blocks = nn.ModuleList(
            [CrossBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp, cfg.dropout)
             for _ in range(cfg.n_cross_layers)]
        )
        self.comparison_bias = ComparisonBias(cfg.n_heads) if cfg.use_comparison_bias else None
        # distinct learned identity for each of the K count-query tokens
        self.count_query_embed = nn.Embedding(cfg.n_slots, cfg.d_model)
        nn.init.normal_(self.count_query_embed.weight, std=0.02)
        self.slots = CountingSlots(cfg.d_model, cfg.n_slots, use_gate=cfg.use_gate)
        self.heads = Heads(
            cfg.d_model, cfg.n_family_tags,
            use_mod=cfg.use_mod, use_bins=cfg.use_bins,
            use_exact_small=cfg.use_exact_small, use_aux=cfg.use_aux,
        )
        self.hook_factor = HookPoint()
        self.hook_query = HookPoint()
        self.hook_readout = HookPoint()
        self.hook_logits = HookPoint()
        self.setup_hook_names()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def n_slots(self) -> int:
        return self.cfg.n_slots

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        factor_tokens = batch["factor_tokens"]
        factor_mask = batch["factor_mask"].bool()
        query_tokens = batch["query_tokens"]
        query_mask = batch["query_mask"].bool()
        global_tokens = batch["global_tokens"]
        B = factor_tokens.shape[0]
        Tg = global_tokens.shape[1]
        K = self.cfg.n_slots
        if Tg != K + 3:
            raise ValueError(
                f"global tokens ({Tg}) must equal n_slots+3 ({K + 3}); "
                f"set the tokenizer's n_count_queries == model n_slots ({K})."
            )

        fac_bias = self.comparison_bias(factor_tokens) if self.comparison_bias else None
        qry_bias = self.comparison_bias(query_tokens) if self.comparison_bias else None

        E_F = self.hook_factor(self.encoder(factor_tokens, factor_mask, attn_bias=fac_bias))
        E_W = self.hook_query(self.encoder(query_tokens, query_mask, attn_bias=qry_bias))
        G0 = self.embed(global_tokens)
        # add distinct identity to each count-query token (positions 1..K)
        G0 = G0.clone()
        G0[:, 1:1 + K] = G0[:, 1:1 + K] + self.count_query_embed.weight.unsqueeze(0)

        x = torch.cat([G0, E_W], dim=1)
        x_mask = torch.cat(
            [torch.ones(B, Tg, dtype=torch.bool, device=x.device), query_mask], dim=1
        )
        for blk in self.cross_blocks:
            x = blk(x, context=E_F, q_mask=x_mask, kv_mask=factor_mask)
        x = self.hook_readout(x)

        G = x[:, :Tg]
        E_W2 = x[:, Tg:]
        # global token layout: [SUPPORT, COUNT_1..K, ZERO, KNOWN]
        h_support = G[:, 0]
        h_count = G[:, 1:1 + K]
        h_zero = G[:, 1 + K]
        h_known = G[:, 2 + K]
        readout = {
            "h_support": h_support,
            "h_count_pool": h_count.mean(dim=1),
            "h_zero": h_zero,
            "h_known": h_known,
            "h_w_pool": _masked_mean(E_W2, query_mask),
            "h_factor_pool": _masked_mean(E_F, factor_mask),
        }

        slot_out = self.slots(h_count)
        head_out = self.heads(readout)
        head_out["pos_logit"] = self.hook_logits(head_out["pos_logit"])

        out: Dict[str, torch.Tensor] = {}
        out.update(head_out)
        out["c_hat"] = slot_out["c_hat"]
        out["slot_values"] = slot_out["slot_values"]
        out["slot_gates"] = slot_out["slot_gates"]
        out["slot_preact"] = slot_out["slot_preact"]
        return out

    # convenience for swap-symmetry checks / exact symmetrisation
    def forward_symmetrized(self, batch_uv: Dict[str, torch.Tensor],
                            batch_vu: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a = self.forward(batch_uv)
        b = self.forward(batch_vu)
        out = {}
        for k in ("pos_logit", "c_hat", "support_score"):
            out[k] = 0.5 * (a[k] + b[k])
        return out
