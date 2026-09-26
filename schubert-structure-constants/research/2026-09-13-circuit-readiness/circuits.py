"""Explicit inference with named intervention sites and native-forward checks.

This module exposes a model's computation. Passing its implementation checks
does not make a behaviorally inaccurate model ready for algorithm extraction.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import campaign as c
from budget import Budget


def explicit(model, x, patches=None, capture=False):
    """Replace full named tensors; callers can construct head/token slices.

    Shapes must agree. Patched patterns must remain probability distributions
    on nonpadding destination keys. Downstream operations are recomputed.
    Captures contain the tensors after any intervention. Evaluation only.
    """
    if model.training:
        raise ValueError('Circuit inference requires model.eval()')
    patches = {} if patches is None else patches
    cache = {}
    used = set()

    def site(name, value):
        if name in patches:
            replacement = patches[name]
            if replacement.shape != value.shape:
                raise ValueError(f'{name}: shape mismatch')
            value = replacement
            used.add(name)
        if capture:
            cache[name] = value
        return value

    padding = x.abs().sum(-1) == 0
    features = site('features', model.feature_map(x))
    relations = site('order_comparisons', model.comparisons(x)) if hasattr(model, 'order_bias') else None
    h = site('embedding', model.input(features))
    for number, layer in enumerate(model.encoder.layers):
        prefix = f'L{number}'
        h = site(f'{prefix}.resid_pre', h)
        batch, tokens, dim = h.shape
        heads = layer.self_attn.num_heads
        dh = dim//heads
        attn_input = site(f'{prefix}.attention_input', layer.norm1(h) if layer.norm_first else h)
        qkv = F.linear(attn_input, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.reshape(batch, tokens, 3, heads, dh).permute(2, 0, 3, 1, 4)
        q = site(f'{prefix}.q', q)
        k = site(f'{prefix}.k', k)
        v = site(f'{prefix}.v', v)
        logits = (q @ k.transpose(-2, -1))/math.sqrt(dh)
        if relations is not None:
            bias = site(f'{prefix}.order_bias', torch.einsum('bqkr,hr->bhqk', relations, model.order_bias[number]))
            logits = logits+bias
        logits = site(f'{prefix}.attention_logits', logits)
        pattern = logits.masked_fill(padding[:, None, None, :], -torch.inf).softmax(-1)
        pattern = site(f'{prefix}.pattern', pattern)
        if f'{prefix}.pattern' in patches:
            if not torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=2e-6, rtol=0):
                raise ValueError('A patched pattern must have unit row sums')
            if (pattern < 0).any() or (pattern.masked_select(padding[:, None, None, :].expand_as(pattern)).abs() > 1e-7).any():
                raise ValueError('A patched pattern violates the destination key mask')
        head_outputs = site(f'{prefix}.head_outputs', pattern @ v)
        transported = head_outputs.transpose(1, 2).reshape(batch, tokens, dim)
        output = site(f'{prefix}.attention_output', layer.self_attn.out_proj(transported))
        h = site(f'{prefix}.resid_after_attention', h+output if layer.norm_first else layer.norm1(h+output))
        mlp_input = site(f'{prefix}.mlp_input', layer.norm2(h) if layer.norm_first else h)
        # This is the fused activation used by native PyTorch 2.6 inference.
        if relations is not None:
            hidden = F.gelu(layer.linear1(mlp_input), approximate='tanh')
        else:
            hidden = torch._addmm_activation(layer.linear1.bias,
                        mlp_input.reshape(batch*tokens, dim), layer.linear1.weight.t(), use_gelu=True).reshape(batch, tokens, -1)
        hidden = site(f'{prefix}.mlp_hidden', hidden)
        mlp = site(f'{prefix}.mlp_output', layer.linear2(hidden))
        h = site(f'{prefix}.resid_post', h+mlp if layer.norm_first else layer.norm2(h+mlp))
    scores = site('scores', model.head(h).squeeze(-1))
    if used != set(patches):
        raise ValueError(f'Unknown intervention sites: {sorted(set(patches)-used)}')
    return (scores, cache) if capture else scores


def check(path, output, budget):
    model = c.model_from(path).eval()
    data = c.load('matched_pairs')
    max_native, decisions, examples = 0., 0, 0
    with torch.inference_mode():
        for start in range(0, len(data['x']), 256):
            budget.check()
            x = torch.as_tensor(data['x'][start:start+256], device='cuda')
            native = model(x)
            actual = explicit(model, x)
            max_native = max(max_native, float((native-actual).abs().max()))
            mask = torch.as_tensor(data['mask'][start:start+256], device='cuda')
            decisions += int((native.masked_fill(~mask, -1e9).argmax(1) == actual.masked_fill(~mask, -1e9).argmax(1)).sum())
            examples += len(x)
        assert max_native < 2e-5 and decisions == examples
        source = torch.as_tensor(data['x'][:64:2], device='cuda')
        dest = torch.as_tensor(data['x'][1:64:2], device='cuda')
        assert torch.equal(source.abs().sum(-1) == 0, dest.abs().sum(-1) == 0)
        ss, sc = explicit(model, source, capture=True)
        ds, dc = explicit(model, dest, capture=True)
        identity_by_site = {name: float((explicit(model, dest, patches={name: value})-ds).abs().max())
                            for name, value in dc.items()}
        identity = max(identity_by_site.values())
        restore_patches = {'embedding': sc['embedding']}
        if 'order_comparisons' in sc:
            restore_patches['order_comparisons'] = sc['order_comparisons']
        restore = float((explicit(model, dest, patches=restore_patches)-ss).abs().max())
        assert identity == 0. and restore == 0.
        qk_checks = {}
        for layer in range(len(model.encoder.layers)):
            key = f'L{layer}'
            qk_patches = {f'{key}.q': sc[f'{key}.q'], f'{key}.k': sc[f'{key}.k']}
            if f'{key}.order_bias' in sc:
                qk_patches[f'{key}.order_bias'] = sc[f'{key}.order_bias']
            qk = explicit(model, dest, patches=qk_patches)
            pat = explicit(model, dest, patches={f'{key}.pattern': sc[f'{key}.pattern']})
            error = float((qk-pat).abs().max())
            assert error < 2e-5
            qk_checks[key] = error
        bad = dc['L0.pattern'].clone()
        bad.zero_()
        try:
            explicit(model, dest, patches={'L0.pattern': bad})
        except ValueError:
            invalid_pattern_rejected = True
        else:
            raise AssertionError('Invalid intervention was accepted')
        shapes = {k: list(v.shape[1:]) for k, v in dc.items()}
    report = dict(checkpoint=str(path), checkpoint_sha256=c.a.sha(path), source_sha256=c.a.sha(Path(__file__)),
                  examples=examples, native_max_error=max_native, decision_agreement=decisions/examples,
                  identity_max_error=identity, embedding_restoration_max_error=restore,
                  identity_by_site=identity_by_site,
                  query_key_pattern_equivalence=qk_checks,
                  order_bias_included_in_restoration=hasattr(model, 'order_bias'),
                  invalid_pattern_rejected=invalid_pattern_rejected,
                  sites=shapes, scope='Implementation controls only; no causal explanation claimed')
    c.write(output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = c.HERE/f'{args.name}_CIRCUIT_CHECKS.json'
    if output.exists():
        raise RuntimeError('Choose a fresh check name')
    budget = Budget(f'{args.name}_circuit_checks', gpu=True)
    status = 'failed_or_partial'
    try:
        check(args.checkpoint, output, budget)
        status = 'completed'
    finally:
        print(json.dumps(budget.finish(status)), flush=True)
