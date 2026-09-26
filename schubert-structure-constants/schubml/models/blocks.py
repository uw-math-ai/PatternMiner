"""Hooked transformer blocks (self- and cross-attention) — PROJECT.md 8.8.

Every interesting activation passes through a named :class:`HookPoint` so the
interpretability stack can cache, ablate, and patch it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interp.hooks import HookPoint


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting cross-attention and an additive bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.hook_q = HookPoint()        # [B, Tq, H, dh]
        self.hook_k = HookPoint()        # [B, Tk, H, dh]
        self.hook_v = HookPoint()        # [B, Tk, H, dh]
        self.hook_pattern = HookPoint()  # [B, H, Tq, Tk]
        self.hook_z = HookPoint()        # [B, Tq, H, dh]
        self.hook_attn_out = HookPoint() # [B, Tq, d]

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv = x if context is None else context
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        H, dh = self.n_heads, self.d_head

        q = self.hook_q(self.q_proj(x).view(B, Tq, H, dh))
        k = self.hook_k(self.k_proj(kv).view(B, Tk, H, dh))
        v = self.hook_v(self.v_proj(kv).view(B, Tk, H, dh))

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(dh)
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            # key_padding_mask: [B, Tk] True == valid
            mask = (~key_padding_mask).view(B, 1, 1, Tk)
            scores = scores.masked_fill(mask, float("-inf"))
        pattern = self.hook_pattern(torch.softmax(scores, dim=-1))
        pattern = self.dropout(pattern)
        z = torch.einsum("bhqk,bkhd->bqhd", pattern, v)
        z = self.hook_z(z)
        out = self.o_proj(z.reshape(B, Tq, self.d_model))
        return self.hook_attn_out(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hook_pre(self.act(self.fc1(x)))
        h = self.dropout(h)
        return self.fc2(self.hook_post(h))


class EncoderBlock(nn.Module):
    """Pre-LN self-attention block (no causal mask — set encoder)."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, dropout)
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x, key_padding_mask=None, attn_bias=None):
        x = self.hook_resid_pre(x)
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = self.hook_resid_mid(x)
        x = x + self.mlp(self.ln2(x))
        return self.hook_resid_post(x)


class CrossBlock(nn.Module):
    """Pre-LN block: self-attention over queries + cross-attention into context."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, dropout)
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid_self = HookPoint()
        self.hook_resid_mid_cross = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x, context, q_mask=None, kv_mask=None):
        x = self.hook_resid_pre(x)
        x = x + self.self_attn(self.ln_self(x), key_padding_mask=q_mask)
        x = self.hook_resid_mid_self(x)
        x = x + self.cross_attn(self.ln_cross_q(x), context=self.ln_cross_kv(context),
                                key_padding_mask=kv_mask)
        x = self.hook_resid_mid_cross(x)
        x = x + self.mlp(self.ln_mlp(x))
        return self.hook_resid_post(x)
