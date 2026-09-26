"""Optional structural attention biases (PROJECT.md 7.2, 7.3).

The pairwise-comparison bias ``sign(w_i - w_j)`` and Rothe diagonal-locality bias
help the encoder see inversions/cells without rediscovering all comparisons.
These are optional; the inversion and Rothe-cell *tokens* already expose the same
information, so the model works with the bias disabled (the default).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.tokenizers import F_TTYPE, F_VALUE, TT_ONELINE


class ComparisonBias(nn.Module):
    """Learned additive attention bias from ``sign(value_i - value_j)`` between
    one-line tokens (zero elsewhere)."""

    def __init__(self, n_heads: int):
        super().__init__()
        # three buckets: less / equal / greater, per head
        self.bias = nn.Parameter(torch.zeros(3, n_heads))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, F]; produce [B, H, T, T]
        is_ol = (tokens[..., F_TTYPE] == TT_ONELINE)  # [B, T]
        val = tokens[..., F_VALUE]                    # [B, T]
        B, T = val.shape
        sign = torch.sign(val[:, :, None] - val[:, None, :]).long() + 1  # 0/1/2
        bias = self.bias[sign]  # [B, T, T, H]
        pair_ol = (is_ol[:, :, None] & is_ol[:, None, :]).unsqueeze(-1)  # [B,T,T,1]
        bias = bias * pair_ol
        return bias.permute(0, 3, 1, 2).contiguous()  # [B, H, T, T]
