"""Multitask training objective (PROJECT.md section 9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .heads import BIN_EDGES, MOD_PRIMES, N_BINS

_LUT_MAX = 2000


@dataclass
class LossWeights:
    positivity: float = 1.0
    coeff_bin: float = 0.5
    log_coeff_huber: float = 0.3
    mod_heads: float = 0.1
    exact_small: float = 0.3
    listwise_support: float = 1.0
    auxiliary: float = 0.2
    slot_sparsity: float = 0.01
    slot_diversity: float = 0.01
    zero_query: float = 0.1
    swap_consistency: float = 0.0


def _bin_lut(device) -> torch.Tensor:
    lut = torch.zeros(_LUT_MAX + 1, dtype=torch.long)
    edges = [1, 2, 3, 4, 6, 11, 21, 51, 101]  # bin boundaries (left-inclusive)
    for c in range(_LUT_MAX + 1):
        b = 0
        for e in edges:
            if c >= e:
                b += 1
        lut[c] = b
    return lut.to(device)


def coeff_bins_tensor(coeff: torch.Tensor) -> torch.Tensor:
    lut = _bin_lut(coeff.device)
    c = coeff.clamp(min=0, max=_LUT_MAX).long()
    return lut[c]


def tier_class_weights(tier_id: torch.Tensor, weights_by_tier: Dict[int, float]) -> torch.Tensor:
    w = torch.ones_like(tier_id, dtype=torch.float)
    for tid, val in weights_by_tier.items():
        w = torch.where(tier_id == tid, torch.full_like(w, val), w)
    return w


def pairwise_pack_loss(scores: torch.Tensor, product_ids: torch.Tensor,
                       positive: torch.Tensor) -> torch.Tensor:
    """Listwise/pairwise support loss: positives outrank hard zeros within a pack."""
    device = scores.device
    total = scores.new_zeros(())
    npairs = 0
    for pid in torch.unique(product_ids):
        idx = (product_ids == pid).nonzero(as_tuple=True)[0]
        s = scores[idx]
        p = positive[idx].bool()
        if p.any() and (~p).any():
            s_pos = s[p]
            s_neg = s[~p]
            diff = s_pos[:, None] - s_neg[None, :]
            total = total + F.softplus(-diff).sum()
            npairs += diff.numel()
    if npairs == 0:
        return scores.new_zeros(())
    return total / npairs


def compute_losses(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   weights: LossWeights,
                   tier_weights: Optional[Dict[int, float]] = None,
                   swapped_out: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Dict[str, torch.Tensor]:
    device = out["pos_logit"].device
    positive = batch["positive"].float()
    coeff = batch["coefficient"].float()
    comps: Dict[str, torch.Tensor] = {}

    # positivity (per-tier weighted BCE)
    if tier_weights:
        cw = tier_class_weights(batch["tier_id"], tier_weights)
    else:
        cw = torch.ones_like(positive)
    bce = F.binary_cross_entropy_with_logits(out["pos_logit"], positive, reduction="none")
    comps["positivity"] = (bce * cw).mean()

    # zero-query head: predict NOT positive (obstruction present)
    if "zero_logit" in out:
        comps["zero_query"] = F.binary_cross_entropy_with_logits(
            out["zero_logit"], 1.0 - positive)

    # log-coefficient huber on the counting-bottleneck output
    chat = out["c_hat"].clamp(min=0)
    comps["log_coeff_huber"] = F.huber_loss(torch.log1p(chat), torch.log1p(coeff))

    # coefficient bins
    if "coeff_bin_logits" in out:
        comps["coeff_bin"] = F.cross_entropy(out["coeff_bin_logits"], coeff_bins_tensor(coeff))

    # exact small coefficient
    if "exact_small_logits" in out:
        tgt = coeff.clamp(min=0, max=10).long()
        comps["exact_small"] = F.cross_entropy(out["exact_small_logits"], tgt)

    # mod-p heads
    if "mod_logits" in out:
        ml = out["mod_logits"]
        mod_loss = chat.new_zeros(())
        for p in MOD_PRIMES:
            tgt = (coeff.long() % p)
            mod_loss = mod_loss + F.cross_entropy(ml[p], tgt)
        comps["mod_heads"] = mod_loss / len(MOD_PRIMES)

    # listwise support
    if "product_id" in batch and "support_score" in out:
        comps["listwise_support"] = pairwise_pack_loss(
            out["support_score"], batch["product_id"], positive)

    # auxiliary (scale length targets so the MSE is O(1) and does not dwarf the
    # positivity/coefficient gradients — raw lengths reach ~40)
    if "aux_reg" in out and "aux_reg_target" in batch:
        comps_aux = F.mse_loss(out["aux_reg"], 0.1 * batch["aux_reg_target"].float())
        if "aux_family_logits" in out and "family_multihot" in batch:
            comps_aux = comps_aux + F.binary_cross_entropy_with_logits(
                out["aux_family_logits"], batch["family_multihot"].float())
        comps["auxiliary"] = comps_aux

    # slot sparsity + diversity
    sv = out["slot_values"]  # [B, K]
    comps["slot_sparsity"] = sv.abs().mean()
    if sv.shape[0] > 1 and sv.shape[1] > 1:
        svc = sv - sv.mean(dim=0, keepdim=True)
        denom = svc.norm(dim=0, keepdim=True).clamp(min=1e-6)
        corr = (svc / denom).t() @ (svc / denom)  # [K, K]
        K = corr.shape[0]
        off = (corr - torch.diag(torch.diag(corr)))
        comps["slot_diversity"] = (off ** 2).sum() / (K * (K - 1))
    else:
        comps["slot_diversity"] = sv.new_zeros(())

    # swap consistency
    if swapped_out is not None and weights.swap_consistency > 0:
        comps["swap_consistency"] = (
            (out["pos_logit"] - swapped_out["pos_logit"]) ** 2
            + (out["c_hat"] - swapped_out["c_hat"]) ** 2
        ).mean()

    total = out["pos_logit"].new_zeros(())
    for name, val in comps.items():
        w = getattr(weights, name, 0.0)
        total = total + w * val
    comps["total"] = total
    return comps
