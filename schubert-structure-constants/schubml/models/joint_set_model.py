"""Joint candidate-set model for the "vanishing is global" test.

Tests whether the residual obstruction that separates survivor zeros from
positives is a GLOBAL property of the whole expansion of ``S_u * S_v``: a model
that sees ALL candidate ``w``'s of a product jointly (cross-candidate
attention) should beat a per-triple model if the obstruction is global.

Architecture
------------
Per-candidate trunk = the frozen ``ProductQueryTransformer`` (pqt_residual_v1,
best-67500) up to its readout representation.  The trunk is NOT part of this
module: its readout vectors (``h_support`` ++ ``h_zero``, the two vectors the
original positivity / zero-obstruction heads read) are precomputed once for all
v004 triples and cached to disk (see ``scripts/train_joint.py --stage
precompute``).  This module is the candidate-set head that consumes them:

1. LayerNorm + linear projection of the 512-d readout to ``d_model``.
2. ``n_layers`` pre-LN transformer encoder layers over the CANDIDATE axis
   (each token = one candidate w of the same product).  No positional
   encoding: the candidate set is unordered, so the layer is permutation-
   equivariant by construction.
3. LayerNorm + linear positivity head per candidate.

The critical control: ``mode="solo"`` runs the *identical* parameter stack but
restricts self-attention to the diagonal (each candidate attends only to
itself), so no cross-candidate information flows.  Any joint-vs-solo delta
under identical training therefore measures exactly the value of joint
candidate context, not extra head capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class JointSetConfig:
    d_in: int = 512          # h_support (256) ++ h_zero (256) from the trunk
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    d_mlp: int = 512
    dropout: float = 0.1
    mode: str = "joint"      # "joint" = full candidate attention; "solo" = diagonal

    def to_dict(self):
        return asdict(self)


class _SetAttention(nn.Module):
    """Self-attention over the candidate axis with optional diagonal restriction."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, solo: bool) -> torch.Tensor:
        # x: [B, C, d]; mask: [B, C] bool (True = real candidate)
        B, C, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, C, H, dh)
        k = self.k_proj(x).view(B, C, H, dh)
        v = self.v_proj(x).view(B, C, H, dh)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(dh)
        if solo:
            # each candidate may attend ONLY to itself -> no cross-candidate flow
            eye = torch.eye(C, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(~eye.view(1, 1, C, C), float("-inf"))
        else:
            # mask padded candidates as keys
            scores = scores.masked_fill(~mask.view(B, 1, 1, C), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        z = torch.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, C, H * dh)
        return self.o_proj(z)


class _SetBlock(nn.Module):
    def __init__(self, cfg: JointSetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _SetAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, solo: bool) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask, solo=solo))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class JointSetModel(nn.Module):
    """Candidate-set positivity head over precomputed trunk readouts."""

    def __init__(self, cfg: JointSetConfig):
        super().__init__()
        self.cfg = cfg
        self.in_norm = nn.LayerNorm(cfg.d_in)
        self.proj = nn.Linear(cfg.d_in, cfg.d_model)
        self.blocks = nn.ModuleList([_SetBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor,
                mode: Optional[str] = None) -> torch.Tensor:
        """feats: [B, C, d_in] readouts; mask: [B, C] bool. Returns [B, C] logits."""
        solo = (mode or self.cfg.mode) == "solo"
        x = self.proj(self.in_norm(feats))
        for blk in self.blocks:
            x = blk(x, mask, solo=solo)
        return self.head(self.out_norm(x)).squeeze(-1)
