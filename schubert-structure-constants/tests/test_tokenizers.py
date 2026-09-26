import numpy as np

from schubml.data.collators import collate
from schubml.data.tokenizers import (N_FIELDS, ROLE_FACTOR, ROLE_QUERY, TT_CLS,
                                     Tokenizer, TokenizerConfig, vocab_sizes)


def test_token_fields_within_vocab():
    tok = Tokenizer(TokenizerConfig(n_ctx=14))
    vs = vocab_sizes(14)
    names = ["ttype", "role", "value", "pos", "i", "j", "aux1", "aux2", "diag", "adiag"]
    # exercise permutations up to size 13 (S7 support)
    perms = [(1,), (2, 1), (3, 1, 2), (1, 3, 2, 5, 4),
             tuple(range(13, 0, -1))]  # w0 in S13, max length
    for w in perms:
        for role, q in [(ROLE_FACTOR, False), (ROLE_QUERY, True)]:
            mat = tok.tokenize_role(w, role, is_query=q)
            assert mat.shape[1] == N_FIELDS
            for k, name in enumerate(names):
                assert mat[:, k].max() < vs[name], (name, mat[:, k].max(), vs[name])
                assert mat[:, k].min() >= 0


def test_cls_token_first():
    tok = Tokenizer(TokenizerConfig(n_ctx=14))
    mat = tok.tokenize_role((3, 1, 2), ROLE_FACTOR, is_query=False)
    assert mat[0, 0] == TT_CLS


def test_global_tokens_count():
    tok = Tokenizer(TokenizerConfig(n_ctx=14))
    g = tok.global_tokens(16)
    assert g.shape[0] == 16 + 3  # SUPPORT + 16 COUNT + ZERO + KNOWN


def test_collate_padding_and_masks():
    tok = Tokenizer(TokenizerConfig(n_ctx=14))
    def ex(u, v, w):
        tu = tok.tokenize_role(u, ROLE_FACTOR, is_query=False)
        tv = tok.tokenize_role(v, ROLE_FACTOR, is_query=False)
        return {"factor": np.concatenate([tu, tv]),
                "query": tok.tokenize_role(w, ROLE_QUERY, is_query=True),
                "glob": tok.global_tokens(8), "positive": 1, "coefficient": 1,
                "tier_id": 0, "product_id": 0, "w_id": 0}
    b = collate([ex((2, 1), (2, 1), (3, 1, 2)), ex((1, 4, 3, 2), (2, 1), (4, 3, 2, 1))])
    assert b["factor_tokens"].shape[0] == 2
    assert b["factor_mask"].dtype == __import__("torch").bool
    # padded rows are masked out
    assert b["factor_mask"].sum().item() > 0
