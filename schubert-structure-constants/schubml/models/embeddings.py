"""Field-based token embeddings.

A token is a vector of integer fields ``[ttype, role, value, pos, i, j, aux1,
aux2, diag, adiag]`` (see ``data.tokenizers``).  Each field has its own embedding
table; the token embedding is their sum.  This keeps the model interpretable:
ablating/patching a single field embedding isolates the contribution of, say,
the Rothe-column coordinate.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..data.tokenizers import N_FIELDS, vocab_sizes
from ..interp.hooks import HookPoint

FIELD_NAMES = ["ttype", "role", "value", "pos", "i", "j", "aux1", "aux2", "diag", "adiag"]


class FieldEmbedding(nn.Module):
    def __init__(self, d_model: int, n_ctx: int):
        super().__init__()
        self.d_model = d_model
        self.n_ctx = n_ctx
        sizes = vocab_sizes(n_ctx)
        self.tables = nn.ModuleDict(
            {name: nn.Embedding(sizes[name], d_model) for name in FIELD_NAMES}
        )
        assert len(FIELD_NAMES) == N_FIELDS
        # per-field hook points (named hook_embed_<field>)
        self.hooks = nn.ModuleDict({name: HookPoint() for name in FIELD_NAMES})
        self.hook_embed = HookPoint()
        self._init()

    def _init(self) -> None:
        for emb in self.tables.values():
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()  # index 0 is the "N/A"/pad slot

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: [..., N_FIELDS] int64 -> [..., d_model]."""
        out = None
        for idx, name in enumerate(FIELD_NAMES):
            e = self.tables[name](tokens[..., idx])
            e = self.hooks[name](e)
            out = e if out is None else out + e
        return self.hook_embed(out)
