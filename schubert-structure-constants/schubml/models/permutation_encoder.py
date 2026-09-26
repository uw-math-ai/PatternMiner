"""Role encoder: a set transformer over a role's multi-view tokens.

The encoder has **no positional encoding over the token stream** (token order is
irrelevant; all structure lives in the integer fields).  Consequently the
encoder is permutation-equivariant over tokens, which is what makes the factor
stream symmetric under ``u <-> v`` (PROJECT.md 8.1, 8.4).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..interp.hooks import HookPoint
from .blocks import EncoderBlock
from .embeddings import FieldEmbedding


class RoleEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_mlp: int,
                 n_ctx: int, dropout: float = 0.0, embedding: Optional[FieldEmbedding] = None):
        super().__init__()
        self.embed = embedding if embedding is not None else FieldEmbedding(d_model, n_ctx)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_mlp, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.hook_tokens = HookPoint()

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.hook_tokens(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=mask, attn_bias=attn_bias)
        return self.ln_final(x)
