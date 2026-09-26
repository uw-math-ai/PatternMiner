#!/usr/bin/env python
"""Train the Schubert product-query transformer (PROJECT.md 16 Milestones 4--5).

Runs under the ``schubml`` env on CUDA.  Example:

    python scripts/train.py --data-dir data/processed/v001 --out runs/pqt_small \
        --d-model 192 --enc-layers 4 --cross-layers 2 --heads 6 --slots 32 \
        --batch-size 384 --lr 4e-4 --max-steps 30000 --precision 16-mixed
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
except Exception:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from schubml.data.collators import collate
from schubml.data.datasets import (ProductPackBatchSampler, SchubertTripleDataset,
                                    build_perms_by_id, load_triple_arrays)
from schubml.data.schemas import PERM_JSON_COLUMNS, load_table
from schubml.data.tokenizers import Tokenizer, TokenizerConfig
from schubml.models.product_query_transformer import ModelConfig
from schubml.train.lightning_module import SchubertLitModule
from schubml.train.logging import repro_info, save_json


def subsample(arrays, n, seed=0):
    total = len(arrays["w_id"])
    if n is None or n >= total:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=False)
    return {k: v[idx] for k, v in arrays.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-split", default="product_disjoint:train")
    ap.add_argument("--val-split", default="product_disjoint:val")
    ap.add_argument("--n-ctx", type=int, default=14)
    ap.add_argument("--max-tokens-per-role", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--cross-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--d-mlp", type=int, default=768)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--comparison-bias", action="store_true")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=40000)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--no-token-cache", action="store_true",
                    help="re-tokenise on the fly (bounded RAM for large perm sets)")
    ap.add_argument("--z5-weight", type=float, default=2.0,
                    help="positivity-loss weight for Z5 residual zeros (class balance)")
    ap.add_argument("--init-weights", default=None,
                    help="warm-start: load MODEL weights only from this ckpt "
                         "(fresh optimizer/schedule), unlike --resume")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.out, exist_ok=True)

    perm_df = load_table(os.path.join(args.data_dir, "permutations.parquet"), PERM_JSON_COLUMNS)
    perms_by_id = build_perms_by_id(perm_df)

    tok = Tokenizer(TokenizerConfig(n_ctx=args.n_ctx,
                                    max_tokens_per_role=args.max_tokens_per_role))

    print(f"[train] loading triples from {args.data_dir}", flush=True)
    tr = subsample(load_triple_arrays(os.path.join(args.data_dir, "triples.parquet"),
                                      split=args.train_split), args.limit_train, args.seed)
    va = subsample(load_triple_arrays(os.path.join(args.data_dir, "triples.parquet"),
                                      split=args.val_split), args.limit_val, args.seed)
    print(f"[train] train={len(tr['w_id'])} val={len(va['w_id'])} perms={len(perms_by_id)}", flush=True)

    n_cq = args.slots
    cache = not args.no_token_cache
    train_ds = SchubertTripleDataset(tr, perms_by_id, tok, n_count_queries=n_cq, cache_tokens=cache)
    val_ds = SchubertTripleDataset(va, perms_by_id, tok, n_count_queries=n_cq, cache_tokens=cache)
    # product-pack batching: each batch holds whole product packs so the
    # listwise-support loss sees many positive/negative pairs per product.
    train_sampler = ProductPackBatchSampler(tr["product_id"], args.batch_size,
                                            shuffle=True, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                          num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0, pin_memory=True)

    mcfg = ModelConfig(
        d_model=args.d_model, n_encoder_layers=args.enc_layers,
        n_cross_layers=args.cross_layers, n_heads=args.heads, d_mlp=args.d_mlp,
        dropout=args.dropout, n_ctx=args.n_ctx, n_slots=args.slots,
        use_comparison_bias=args.comparison_bias,
    )
    tier_weights = {1: 0.2, 2: 0.3, 3: 1.0, 4: 1.5, 5: 2.0, 6: args.z5_weight}
    lit = SchubertLitModule(mcfg, lr=args.lr, weight_decay=args.weight_decay,
                            warmup_steps=args.warmup, total_steps=args.max_steps,
                            tier_weights=tier_weights)
    if args.init_weights:
        sd = torch.load(args.init_weights, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = lit.load_state_dict(sd, strict=False)
        print(f"[train] warm-started from {args.init_weights} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    n_params = sum(p.numel() for p in lit.model.parameters())
    print(f"[train] model params={n_params:,}", flush=True)

    save_json({"model_cfg": vars(mcfg), "args": vars(args), "n_params": n_params,
               "repro": repro_info(args.seed)}, os.path.join(args.out, "run_config.json"))

    ckpt = ModelCheckpoint(dirpath=os.path.join(args.out, "ckpts"),
                           filename="best-{step}", monitor="val/pos_f1", mode="max",
                           save_top_k=3, save_last=True, auto_insert_metric_name=False)
    lrmon = LearningRateMonitor(logging_interval="step")
    loggers = [CSVLogger(args.out, name="csv"), TensorBoardLogger(args.out, name="tb")]

    trainer = pl.Trainer(
        max_steps=args.max_steps, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, precision=args.precision, gradient_clip_val=args.grad_clip,
        val_check_interval=args.val_every, check_val_every_n_epoch=None,
        callbacks=[ckpt, lrmon], logger=loggers, log_every_n_steps=50,
        enable_progress_bar=False, default_root_dir=args.out,
    )
    trainer.fit(lit, train_dl, val_dl, ckpt_path=args.resume)
    print(f"[train] DONE best={ckpt.best_model_path} score={ckpt.best_model_score}", flush=True)
    save_json({"best_model_path": ckpt.best_model_path,
               "best_score": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None},
              os.path.join(args.out, "train_result.json"))


if __name__ == "__main__":
    main()
