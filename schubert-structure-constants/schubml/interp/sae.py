"""ReLU sparse autoencoder for residual / MLP activations (PROJECT.md 13.7).

A standard one-hidden-layer SAE with an L1 sparsity penalty on the hidden code:

    z   = ReLU(W_enc (x - b_dec) + b_enc)        (nonnegative features)
    x'  = W_dec z + b_dec

Loss = ||x - x'||^2  +  l1 * ||z||_1 .  Weights may be *tied* (``W_enc = W_dec^T``)
or untied.  Dead-feature **resampling** (Anthropic "Towards Monosemanticity") is
supported.  Decoder columns are kept unit-norm so the L1 penalty is comparable
across features.

Typical sites (PROJECT.md 13.7): ``encoder.blocks.1.hook_resid_pre``,
``cross_blocks.{i}.hook_resid_post``, ``cross_blocks.{i}.mlp.hook_post``,
``hook_readout`` COUNT tokens, ``slots.hook_slot_preact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hooks import HookedModule


@dataclass
class SAEConfig:
    d_in: int
    d_hidden: int
    l1: float = 1e-3
    tied: bool = False
    lr: float = 1e-3
    steps: int = 2000
    batch_size: int = 1024
    resample_every: int = 0          # 0 disables dead-feature resampling
    dead_steps: int = 200            # a feature is "dead" if unused this many steps
    seed: int = 0
    device: str = "cpu"
    normalize_decoder: bool = True


class SAE(nn.Module):
    """ReLU sparse autoencoder.

    ``SAE(d_in, d_hidden, l1)`` matches the contract; extra options come from a
    :class:`SAEConfig` when constructed via :func:`train_sae`.
    """

    def __init__(self, d_in: int, d_hidden: int, l1: float = 1e-3,
                 *, tied: bool = False, normalize_decoder: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.l1 = l1
        self.tied = tied
        self.normalize_decoder = normalize_decoder

        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if tied:
            self.W_dec = None
        else:
            self.W_dec = nn.Parameter(self.W_enc.detach().t().clone())
        with torch.no_grad():
            self._normalize_decoder()

    # --- decoder access (tied or untied) ---
    def decoder_weight(self) -> torch.Tensor:
        return self.W_enc.t() if self.tied else self.W_dec

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        if not self.normalize_decoder:
            return
        W = self.decoder_weight()
        norm = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        if self.tied:
            self.W_enc.div_(norm.t())
        else:
            self.W_dec.div_(norm)

    # --- forward / encode / decode ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc.t() + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.decoder_weight().t() + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_hat, z = self.forward(x)
        recon = F.mse_loss(x_hat, x, reduction="mean")
        l1 = z.abs().sum(dim=-1).mean()
        total = recon + self.l1 * l1
        return {"loss": total, "recon": recon.detach(), "l1": l1.detach(),
                "l0": (z > 0).float().sum(dim=-1).mean().detach()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@torch.no_grad()
def _resample_dead(sae: SAE, dead_mask: torch.Tensor, x: torch.Tensor) -> int:
    """Reinitialise dead features towards poorly-reconstructed inputs."""
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0 or x.shape[0] == 0:
        return 0
    x_hat, _ = sae.forward(x)
    err = (x - x_hat).norm(dim=-1)
    probs = (err ** 2)
    if probs.sum() <= 0:
        idx = torch.randint(0, x.shape[0], (n_dead,), device=x.device)
    else:
        idx = torch.multinomial(probs, n_dead, replacement=True)
    samples = x[idx]
    samples = F.normalize(samples - sae.b_dec, dim=-1)
    dead_idx = dead_mask.nonzero(as_tuple=True)[0]
    sae.W_enc[dead_idx] = samples * 0.2
    sae.b_enc[dead_idx] = 0.0
    if not sae.tied:
        sae.W_dec[:, dead_idx] = samples.t()
    sae._normalize_decoder()
    return n_dead


def train_sae(activations: torch.Tensor, cfg: SAEConfig) -> SAE:
    """Train an :class:`SAE` on a ``[N, d_in]`` activation matrix.

    Returns the trained module (on CPU).  Deterministic given ``cfg.seed``.
    """
    torch.manual_seed(cfg.seed)
    X = activations.detach().float()
    device = torch.device(cfg.device)
    X = X.to(device)
    sae = SAE(cfg.d_in, cfg.d_hidden, cfg.l1, tied=cfg.tied,
              normalize_decoder=cfg.normalize_decoder).to(device)
    # initialise the decoder bias at the data mean (a good baseline reconstruction)
    with torch.no_grad():
        sae.b_dec.copy_(X.mean(dim=0))
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    n = X.shape[0]
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    fired = torch.zeros(cfg.d_hidden, device=device)
    steps_since_fire = torch.zeros(cfg.d_hidden, device=device)

    for step in range(cfg.steps):
        idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=g, device=device)
        xb = X[idx]
        stats = sae.loss(xb)
        opt.zero_grad()
        stats["loss"].backward()
        opt.step()
        with torch.no_grad():
            sae._normalize_decoder()
            z = sae.encode(xb)
            active = (z > 0).any(dim=0)
            steps_since_fire = torch.where(active, torch.zeros_like(steps_since_fire),
                                           steps_since_fire + 1)
        if cfg.resample_every and (step + 1) % cfg.resample_every == 0:
            dead = steps_since_fire >= cfg.dead_steps
            _resample_dead(sae, dead, xb)
            steps_since_fire = torch.where(dead, torch.zeros_like(steps_since_fire),
                                           steps_since_fire)

    return sae.to("cpu").eval()


@torch.no_grad()
def encode(sae: SAE, activations: torch.Tensor) -> torch.Tensor:
    """Encode a ``[N, d_in]`` matrix to hidden codes ``[N, d_hidden]`` (CPU)."""
    sae = sae.to("cpu").eval()
    return sae.encode(activations.detach().float().cpu())


# ---------------------------------------------------------------------------
# Activation gathering over a dataset
# ---------------------------------------------------------------------------
def _pool(x: torch.Tensor, batch: Dict[str, torch.Tensor], hook_name: str,
          n_slots: int, readout_token: Optional[object]) -> torch.Tensor:
    """Reduce a hook activation to ``[B, d]`` rows for SAE training."""
    if x.dim() == 2:  # already [B, d] (e.g. slot vectors)
        return x
    if hook_name == "hook_readout" and readout_token is not None:
        if readout_token == "count":
            return x[:, 1:1 + n_slots].mean(dim=1)
        if readout_token == "support":
            return x[:, 0]
        if readout_token == "zero":
            return x[:, n_slots + 1]
        if readout_token == "known":
            return x[:, n_slots + 2]
        if isinstance(readout_token, int):
            return x[:, 1 + readout_token]
    if x.dim() == 4:  # [B, T, H, dh] -> flatten heads
        x = x.reshape(x.shape[0], x.shape[1], -1)
    # masked mean over valid tokens when a mask is available, else plain mean
    mask_key = "query_mask" if "query" in hook_name or "encoder" in hook_name else None
    if mask_key and mask_key in batch:
        m = batch[mask_key].bool().unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return x.mean(dim=1)


@torch.no_grad()
def gather_activations(
    model: HookedModule,
    dataset,
    hook_name: str,
    device: str = "cuda",
    *,
    max_examples: int = 8192,
    batch_size: int = 256,
    readout_token: Optional[object] = None,
    per_token: bool = False,
) -> torch.Tensor:
    """Collect a ``[N, d]`` activation matrix at ``hook_name`` over ``dataset``.

    ``per_token=True`` keeps *every valid token row* (good for SAEs on token
    streams); otherwise one pooled row per example.  Returns a CPU float tensor.
    """
    from ..data.collators import collate, to_device

    model.eval().to(device)
    n = min(len(dataset), max_examples)
    n_slots = int(getattr(model, "n_slots", model.cfg.n_slots))
    rows: List[torch.Tensor] = []
    for start in range(0, n, batch_size):
        exs = [dataset[i] for i in range(start, min(start + batch_size, n))]
        batch = to_device(collate(exs), device)
        _, cache = model.run_with_cache(batch, names=[hook_name])
        x = cache[hook_name]
        if per_token and x.dim() >= 3:
            xf = x.reshape(x.shape[0], x.shape[1], -1) if x.dim() == 4 else x
            mask_key = "query_mask" if ("query" in hook_name or "encoder" in hook_name) else None
            if mask_key and mask_key in batch:
                m = batch[mask_key].bool()
                rows.append(xf[m].detach().cpu())
            else:
                rows.append(xf.reshape(-1, xf.shape[-1]).detach().cpu())
        else:
            rows.append(_pool(x, batch, hook_name, n_slots, readout_token).detach().cpu())
    return torch.cat(rows, dim=0).float()


# ---------------------------------------------------------------------------
# Per-feature statistics
# ---------------------------------------------------------------------------
def feature_stats(sae: SAE, activations: torch.Tensor, *, top_k: int = 16) -> List[Dict[str, object]]:
    """Per-feature dict: sparsity, mean/max activation, top activating rows.

    ``top_activating`` are row indices into ``activations`` (i.e. into whatever
    example/token ordering the caller used to build the matrix).
    """
    z = encode(sae, activations).numpy()  # [N, H]
    n = z.shape[0]
    out: List[Dict[str, object]] = []
    for f in range(z.shape[1]):
        col = z[:, f]
        active = col > 0
        order = np.argsort(-col)[:top_k]
        out.append({
            "feature_id": int(f),
            "sparsity": float(active.mean()),           # fraction of rows active
            "mean_activation": float(col[active].mean()) if active.any() else 0.0,
            "max_activation": float(col.max()) if n else 0.0,
            "n_active": int(active.sum()),
            "top_activating": [int(i) for i in order if col[i] > 0],
        })
    return out
