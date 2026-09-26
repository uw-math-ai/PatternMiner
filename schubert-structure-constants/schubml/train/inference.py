"""Model loading + batched inference helpers (shared by evaluate / interp)."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.collators import collate, to_device
from ..data.datasets import SchubertTripleDataset
from ..data.tokenizers import Tokenizer, TokenizerConfig
from ..models.heads import MOD_PRIMES
from ..models.product_query_transformer import ModelConfig, ProductQueryTransformer


def load_run(out_dir: str, ckpt_path: Optional[str] = None, device: str = "cuda"):
    """Load a trained ``ProductQueryTransformer`` from a run directory.

    Returns ``(model, model_cfg, tokenizer)``.
    """
    with open(os.path.join(out_dir, "run_config.json")) as f:
        rc = json.load(f)
    mcfg = ModelConfig(**rc["model_cfg"])
    model = ProductQueryTransformer(mcfg)
    if ckpt_path is None:
        ckpt_path = _best_ckpt(out_dir)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")} or sd
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    mtr = int(rc.get("args", {}).get("max_tokens_per_role", 128))
    tok = Tokenizer(TokenizerConfig(n_ctx=mcfg.n_ctx, max_tokens_per_role=mtr))
    return model, mcfg, tok


def _best_ckpt(out_dir: str) -> str:
    ck = os.path.join(out_dir, "ckpts")
    cands = [os.path.join(ck, f) for f in os.listdir(ck) if f.endswith(".ckpt")]
    best = [c for c in cands if "last" not in c]
    return (best or cands)[0]


@torch.no_grad()
def predict(model, arrays: Dict[str, np.ndarray], perms_by_id, tok, *,
            n_count_queries: int, device: str = "cuda", batch_size: int = 512,
            num_workers: int = 2) -> Dict[str, np.ndarray]:
    ds = SchubertTripleDataset(arrays, perms_by_id, tok, n_count_queries=n_count_queries,
                               with_aux=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    num_workers=num_workers)
    keys_scalar = ["pos_logit", "c_hat", "support_score"]
    acc: Dict[str, list] = {k: [] for k in keys_scalar}
    acc.update({"positive": [], "coefficient": [], "tier_id": [], "product_id": [],
                "bin_pred": []})
    for p in MOD_PRIMES:
        acc[f"mod{p}_pred"] = []
    for batch in dl:
        batch = to_device(batch, device)
        out = model(batch)
        acc["pos_logit"].append(out["pos_logit"].float().cpu().numpy())
        acc["c_hat"].append(out["c_hat"].float().cpu().numpy())
        acc["support_score"].append(out.get("support_score", out["pos_logit"]).float().cpu().numpy())
        acc["positive"].append(batch["positive"].cpu().numpy())
        acc["coefficient"].append(batch["coefficient"].cpu().numpy())
        acc["tier_id"].append(batch["tier_id"].cpu().numpy())
        acc["product_id"].append(batch["product_id"].cpu().numpy())
        if "coeff_bin_logits" in out:
            acc["bin_pred"].append(out["coeff_bin_logits"].argmax(-1).cpu().numpy())
        if "mod_logits" in out:
            for p in MOD_PRIMES:
                acc[f"mod{p}_pred"].append(out["mod_logits"][p].argmax(-1).cpu().numpy())
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}
