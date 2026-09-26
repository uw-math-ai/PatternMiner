"""Prediction heads (PROJECT.md 8.7) and coefficient binning helpers."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

# coefficient bins (PROJECT.md 9.4)
BIN_EDGES = [0, 1, 2, 3, 5, 10, 20, 50, 100]  # bin k = (edges[k-1], edges[k]]
N_BINS = 10
EXACT_SMALL_CLASSES = 11  # {0,1,...,9, >=10}
MOD_PRIMES = [2, 3, 5, 7]


def coeff_to_bin(c: int) -> int:
    if c <= 0:
        return 0
    if c == 1:
        return 1
    if c == 2:
        return 2
    if c == 3:
        return 3
    if c <= 5:
        return 4
    if c <= 10:
        return 5
    if c <= 20:
        return 6
    if c <= 50:
        return 7
    if c <= 100:
        return 8
    return 9


def coeff_to_exact_small(c: int) -> int:
    return min(int(c), 10)


class Heads(nn.Module):
    """Bundle of linear heads reading from pooled readout vectors."""

    def __init__(self, d_model: int, n_family_tags: int, *, use_mod: bool = True,
                 use_bins: bool = True, use_exact_small: bool = True, use_aux: bool = True):
        super().__init__()
        self.use_mod = use_mod
        self.use_bins = use_bins
        self.use_exact_small = use_exact_small
        self.use_aux = use_aux

        self.positivity = nn.Linear(d_model, 1)
        self.support_score = nn.Linear(d_model, 1)         # listwise ranking score
        self.zero_obstruction = nn.Linear(d_model, 1)      # from ZERO_QUERY
        if use_bins:
            self.coeff_bin = nn.Linear(d_model, N_BINS)
        if use_exact_small:
            self.exact_small = nn.Linear(d_model, EXACT_SMALL_CLASSES)
        if use_mod:
            self.mod_heads = nn.ModuleDict(
                {str(p): nn.Linear(d_model, p) for p in MOD_PRIMES}
            )
        if use_aux:
            # length_w, length_sum, length_defect (regression); ndesc_w (regression)
            self.aux_reg = nn.Linear(d_model, 4)
            self.aux_family = nn.Linear(d_model, n_family_tags)  # multi-label logits

    def forward(self, readout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_support = readout["h_support"]
        h_count = readout["h_count_pool"]
        h_zero = readout["h_zero"]
        h_known = readout["h_known"]
        h_w = readout["h_w_pool"]
        h_fac = readout["h_factor_pool"]
        out: Dict[str, torch.Tensor] = {}
        out["pos_logit"] = self.positivity(h_support).squeeze(-1)
        out["support_score"] = self.support_score(h_support).squeeze(-1)
        out["zero_logit"] = self.zero_obstruction(h_zero).squeeze(-1)
        if self.use_bins:
            out["coeff_bin_logits"] = self.coeff_bin(h_count)
        if self.use_exact_small:
            out["exact_small_logits"] = self.exact_small(h_count)
        if self.use_mod:
            out["mod_logits"] = {p: self.mod_heads[str(p)](h_count) for p in MOD_PRIMES}
        if self.use_aux:
            out["aux_reg"] = self.aux_reg(h_w + h_fac)
            out["aux_family_logits"] = self.aux_family(h_known)
        return out
