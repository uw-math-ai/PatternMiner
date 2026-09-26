import numpy as np
import torch

from schubml.data.collators import collate
from schubml.data.tokenizers import ROLE_FACTOR, ROLE_QUERY, Tokenizer, TokenizerConfig
from schubml.models.losses import LossWeights, coeff_bins_tensor, compute_losses
from schubml.models.heads import coeff_to_bin
from schubml.models.product_query_transformer import ModelConfig, ProductQueryTransformer

tok = Tokenizer(TokenizerConfig(n_ctx=14))
K = 8


def _batch():
    def ex(u, v, w, c, pos):
        tu = tok.tokenize_role(u, ROLE_FACTOR, is_query=False)
        tv = tok.tokenize_role(v, ROLE_FACTOR, is_query=False)
        return {"factor": np.concatenate([tu, tv]),
                "query": tok.tokenize_role(w, ROLE_QUERY, is_query=True),
                "glob": tok.global_tokens(K), "positive": pos, "coefficient": c,
                "tier_id": 0 if pos else 3, "product_id": 0, "w_id": hash(w) % 100,
                "aux_reg_target": np.zeros(4, np.float32),
                "family_multihot": np.zeros(11, np.float32)}
    return collate([ex((1, 3, 2), (2, 1), (2, 3, 1), 1, 1),
                    ex((1, 3, 2), (2, 1), (1, 3, 2), 0, 0)])


def test_coeff_bins_tensor_matches_scalar():
    coeffs = torch.tensor([0, 1, 2, 3, 4, 5, 7, 15, 30, 120, 500])
    binned = coeff_bins_tensor(coeffs).numpy()
    for c, b in zip(coeffs.tolist(), binned):
        assert b == coeff_to_bin(c)


def test_losses_finite_and_components():
    torch.manual_seed(0)
    model = ProductQueryTransformer(ModelConfig(d_model=48, n_encoder_layers=2,
        n_cross_layers=2, n_heads=3, d_mlp=96, n_ctx=14, n_slots=K))
    b = _batch()
    out = model(b)
    comps = compute_losses(out, b, LossWeights(), tier_weights={3: 1.0})
    assert torch.isfinite(comps["total"])
    for key in ("positivity", "log_coeff_huber", "coeff_bin", "slot_sparsity"):
        assert key in comps and torch.isfinite(comps[key])
    # slot sparsity is nonnegative
    assert comps["slot_sparsity"] >= 0


def test_backward_runs():
    torch.manual_seed(0)
    model = ProductQueryTransformer(ModelConfig(d_model=32, n_encoder_layers=1,
        n_cross_layers=1, n_heads=2, d_mlp=64, n_ctx=14, n_slots=K))
    b = _batch()
    comps = compute_losses(model(b), b, LossWeights(), tier_weights={3: 1.0})
    comps["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
