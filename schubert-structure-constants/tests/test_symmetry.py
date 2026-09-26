"""u/v symmetry, padding consistency, and shape tests for the model."""

import numpy as np
import torch

from schubml.data.collators import collate
from schubml.data.tokenizers import ROLE_FACTOR, ROLE_QUERY, Tokenizer, TokenizerConfig
from schubml.models.product_query_transformer import ModelConfig, ProductQueryTransformer

tok = Tokenizer(TokenizerConfig(n_ctx=14))
K = 8


def _example(u, v, w):
    tu = tok.tokenize_role(u, ROLE_FACTOR, is_query=False)
    tv = tok.tokenize_role(v, ROLE_FACTOR, is_query=False)
    tw = tok.tokenize_role(w, ROLE_QUERY, is_query=True)
    return {"factor": np.concatenate([tu, tv]), "query": tw, "glob": tok.global_tokens(K),
            "positive": 1, "coefficient": 1, "tier_id": 0, "product_id": 0, "w_id": 0}


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=48, n_encoder_layers=2, n_cross_layers=2, n_heads=3,
                      d_mlp=96, n_ctx=14, n_slots=K)
    return ProductQueryTransformer(cfg).eval()


def test_uv_symmetry_by_construction():
    model = _model()
    b_uv = collate([_example((1, 3, 2), (2, 1), (2, 3, 1))])
    b_vu = collate([_example((2, 1), (1, 3, 2), (2, 3, 1))])
    with torch.no_grad():
        o1 = model(b_uv)
        o2 = model(b_vu)
    assert torch.allclose(o1["pos_logit"], o2["pos_logit"], atol=1e-4)
    assert torch.allclose(o1["c_hat"], o2["c_hat"], atol=1e-4)


def test_padding_consistency():
    # canonical (2,1) vs padded (2,1,3) index the same Schubert polynomial
    model = _model()
    b1 = collate([_example((2, 1), (1, 3, 2), (2, 3, 1))])
    b2 = collate([_example((2, 1, 3), (1, 3, 2), (2, 3, 1))])
    with torch.no_grad():
        o1, o2 = model(b1), model(b2)
    assert torch.allclose(o1["pos_logit"], o2["pos_logit"], atol=1e-4)


def test_counting_slots_nonnegative_and_shapes():
    model = _model()
    b = collate([_example((1, 3, 2), (2, 1), (2, 3, 1)),
                 _example((2, 1), (2, 1), (3, 1, 2))])
    with torch.no_grad():
        out = model(b)
    assert out["slot_values"].shape == (2, K)
    assert (out["slot_values"] >= 0).all()
    assert (out["c_hat"] >= 0).all()
    assert torch.allclose(out["c_hat"], out["slot_values"].sum(-1), atol=1e-5)


def test_hooks_and_cache():
    model = _model()
    b = collate([_example((1, 3, 2), (2, 1), (2, 3, 1))])
    out, cache = model.run_with_cache(b)
    assert "hook_logits" in model.hook_points()
    assert any("slot_values" in k for k in cache)
