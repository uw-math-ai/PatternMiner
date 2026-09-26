"""LightningModule wrapping the product-query transformer + multitask loss."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

from ..models.heads import MOD_PRIMES
from ..models.losses import LossWeights, coeff_bins_tensor, compute_losses
from ..models.product_query_transformer import ModelConfig, ProductQueryTransformer


class SchubertLitModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        loss_weights: Optional[Dict[str, float]] = None,
        tier_weights: Optional[Dict[int, float]] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 50000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_cfg"])
        self.model_cfg = model_cfg
        self.model = ProductQueryTransformer(model_cfg)
        self.loss_weights = LossWeights(**(loss_weights or {}))
        self.tier_weights = tier_weights or {1: 0.2, 2: 0.3, 3: 1.0, 4: 1.5, 5: 2.0, 6: 2.0}
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._val: List[Dict[str, np.ndarray]] = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        comps = compute_losses(out, batch, self.loss_weights, tier_weights=self.tier_weights)
        bs = batch["positive"].shape[0]
        for name, val in comps.items():
            self.log(f"train/{name}", val, on_step=True, on_epoch=False,
                     batch_size=bs, prog_bar=(name == "total"))
        with torch.no_grad():
            acc = ((out["pos_logit"] > 0).float() == batch["positive"]).float().mean()
            self.log("train/pos_acc", acc, on_step=True, prog_bar=True, batch_size=bs)
            # periodic stdout line so a detached run is monitorable via the log
            if self.global_step % 200 == 0:
                lr = self.optimizers().param_groups[0]["lr"] if self.optimizers() else 0.0
                print(f"[train] step={self.global_step} loss={comps['total'].item():.4f} "
                      f"pos={comps['positivity'].item():.4f} acc={acc.item():.3f} "
                      f"lr={lr:.2e}", flush=True)
        return comps["total"]

    def on_validation_end(self):
        cm = self.trainer.callback_metrics
        def g(k):
            v = cm.get(k)
            return float(v) if v is not None else float("nan")
        print(f"[val] step={self.global_step} pos_f1={g('val/pos_f1'):.3f} "
              f"pos_mcc={g('val/pos_mcc'):.3f} hardzero_f1={g('val/hardzero_pos_f1'):.3f} "
              f"coeff_exact={g('val/coeff_exact_acc'):.3f} support_ndcg={g('val/support_ndcg'):.3f}",
              flush=True)

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        rec = {
            "pos_logit": out["pos_logit"].detach().float().cpu().numpy(),
            "c_hat": out["c_hat"].detach().float().cpu().numpy(),
            "positive": batch["positive"].detach().cpu().numpy(),
            "coefficient": batch["coefficient"].detach().cpu().numpy(),
            "tier_id": batch["tier_id"].detach().cpu().numpy(),
            "product_id": batch["product_id"].detach().cpu().numpy(),
            "support_score": out.get("support_score", out["pos_logit"]).detach().float().cpu().numpy(),
        }
        if "coeff_bin_logits" in out:
            rec["bin_pred"] = out["coeff_bin_logits"].argmax(-1).detach().cpu().numpy()
        if "mod_logits" in out:
            for p in MOD_PRIMES:
                rec[f"mod{p}_pred"] = out["mod_logits"][p].argmax(-1).detach().cpu().numpy()
        self._val.append(rec)
        return None

    def on_validation_epoch_end(self):
        from .metrics import (coefficient_metrics, mod_accuracy, positivity_metrics,
                              support_ranking_metrics)
        if not self._val:
            return
        cat = {k: np.concatenate([r[k] for r in self._val]) for k in self._val[0]}
        self._val.clear()
        pm = positivity_metrics(cat["pos_logit"], cat["positive"])
        for k, v in pm.items():
            self.log(f"val/pos_{k}", v, prog_bar=(k in ("f1", "mcc")))
        bin_true = coeff_bins_tensor(torch.tensor(cat["coefficient"])).numpy() if "bin_pred" in cat else None
        cm = coefficient_metrics(cat["c_hat"], cat["coefficient"].astype(int),
                                 cat.get("bin_pred"), bin_true)
        for k, v in cm.items():
            self.log(f"val/coeff_{k}", v, prog_bar=(k == "exact_acc"))
        modp = {p: cat[f"mod{p}_pred"] for p in MOD_PRIMES if f"mod{p}_pred" in cat}
        for k, v in mod_accuracy(modp, cat["coefficient"].astype(int)).items():
            self.log(f"val/{k}", v)
        sr = support_ranking_metrics(cat["support_score"], cat["positive"],
                                     cat["coefficient"], cat["product_id"])
        for k, v in sr.items():
            self.log(f"val/support_{k}", v)
        # hard-zero stratified positivity accuracy (tiers Z4/Z5 == ids 5,6 + positives)
        hard = np.isin(cat["tier_id"], [5, 6]) | (cat["positive"] > 0)
        if hard.sum() > 0 and len(np.unique(cat["positive"][hard])) > 1:
            hm = positivity_metrics(cat["pos_logit"][hard], cat["positive"][hard])
            self.log("val/hardzero_pos_f1", hm["f1"], prog_bar=True)
            self.log("val/hardzero_pos_mcc", hm["mcc"])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
